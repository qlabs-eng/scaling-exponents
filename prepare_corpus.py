"""Prepare the paper's FineWeb or FineWeb-Edu training and validation token pools.

GPT-2 tokens are stored as uint16 with document offsets. Validation is the first
10M tokens of sample-10BT; training is a prefix of sample/100BT in sorted shard
order, excluding documents with a matching 64-token validation signature.
FineWeb-Edu instead reserves the first sorted sample/100BT shard for validation
and trains on the remaining shards.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import importlib.metadata
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tiktoken
import torch

SEQUENCE_LENGTH = 2048
SEQUENCE_SIZE = SEQUENCE_LENGTH + 1
BOS_ID = 50256          # GPT-2 <|endoftext|>
SIG_PREFIX = 64         # tokens of a document used as its dedup signature
ENCODE_BATCH = 20_000   # documents per tiktoken batch

PARQUET_SOURCE = {
    "fineweb": ("HuggingFaceFW/fineweb", "sample/100BT/*.parquet"),
    "fineweb-edu": ("HuggingFaceFW/fineweb-edu", "sample/100BT/*.parquet"),
}

DEFAULT_TRAIN_TOKENS = 30_000_000_000
FINEWEB_VAL_TOKENS = 10_000_000     # the paper's FineWeb validation set, to the token
FINEWEB_EDU_VAL_TOKENS = 10_200_000


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def retry(fn, desc, tries=8, base=3.0):
    """Retry a flaky hub call -- an unauthenticated burst gets rate-limited routinely."""
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 -- the network layer raises many types
            last = exc
            wait = base * (i + 1)
            log(f"  {desc}: attempt {i + 1}/{tries} failed ({type(exc).__name__}: "
                f"{str(exc)[:80]}); retrying in {wait:.0f}s")
            time.sleep(wait)
    raise SystemExit(f"{desc}: giving up after {tries} attempts: {last}")


# ---------------------------------------------------------------------------
# packing: documents in, one contiguous uint16 pool out
# ---------------------------------------------------------------------------

class Pool:
    """A preallocated uint16 token pool, filled in place.

    30B tokens is 60GB as uint16; a Python-list accumulator would need well over ten
    times that, so the buffer is allocated once up front and written into.
    """

    def __init__(self, budget: int, label: str):
        self.buf = np.empty(budget, dtype=np.uint16)
        self.budget = budget
        self.doc_starts: list[int] = []
        self.cur = 0
        self.label = label
        self.t0 = time.time()

    @property
    def full(self) -> bool:
        return self.cur >= self.budget

    def add(self, tokens) -> None:
        """Append <bos> + tokens, clipped to the remaining budget."""
        if self.full:
            return
        self.buf[self.cur] = BOS_ID
        self.cur += 1
        self.doc_starts.append(self.cur - 1)
        take = min(len(tokens), self.budget - self.cur)
        if take > 0:
            self.buf[self.cur:self.cur + take] = np.asarray(tokens[:take], dtype=np.uint16)
            self.cur += take

    def progress(self, what: str) -> None:
        rate = self.cur / max(1e-9, time.time() - self.t0)
        log(f"  [{self.label}] {what}: {self.cur:,}/{self.budget:,} "
            f"({100 * self.cur / self.budget:.1f}%), {len(self.doc_starts):,} docs, "
            f"{rate:,.0f} tok/s")

    def finish(self, source_desc: str):
        if not self.full:
            raise SystemExit(f"[{self.label}] ran out of data: {self.cur:,}/{self.budget:,} "
                             f"tokens from {source_desc}; raise --num-shards")
        return self.buf, np.asarray(self.doc_starts, dtype=np.int64)


def array_sha256(array: np.ndarray) -> str:
    """Hash tensor bytes without allocating a second corpus-sized buffer."""
    digest = hashlib.sha256()
    raw = memoryview(array).cast("B")
    for start in range(0, len(raw), 64 << 20):
        digest.update(raw[start:start + (64 << 20)])
    return digest.hexdigest()


def write_pool(path: str, tokens: np.ndarray, doc_starts: np.ndarray, provenance=None) -> None:
    assert doc_starts[0] == 0, "the first document must start at offset 0"
    assert np.all(tokens[doc_starts] == BOS_ID), "every document must start with bos"
    n_seq = tokens.size // SEQUENCE_SIZE
    log(f"writing {path}: {tokens.size:,} tokens, {doc_starts.size:,} docs, {n_seq:,} sequences")
    torch.save({
        "tokens": torch.from_numpy(tokens),
        "doc_starts": torch.from_numpy(doc_starts),
        "bos_id": int(BOS_ID),
        "seq_size": int(SEQUENCE_SIZE),
        "token_count": int(tokens.size),
        "doc_count": int(doc_starts.size),
    }, path)
    if provenance is not None:
        provenance.setdefault("outputs", {})[os.path.basename(path)] = {
            "token_count": int(tokens.size), "doc_count": int(doc_starts.size),
            "tokens_sha256": array_sha256(tokens), "doc_starts_sha256": array_sha256(doc_starts),
        }


def doc_signature(tokens) -> bytes:
    """SHA-1 of up to SIG_PREFIX body tokens, preserving the paper's exclusion.

    Truncation preserves this signature only when at least SIG_PREFIX body tokens
    remain. FineWeb's final stored validation document has only 48 body tokens;
    its signature can differ from the longer source document's signature. That
    document lies outside the paper's retained validation targets.
    """
    return hashlib.sha1(np.asarray(tokens[:SIG_PREFIX], dtype=np.uint32).tobytes()).digest()


def pool_signatures(path: str) -> set[bytes]:
    d = torch.load(path, map_location="cpu", weights_only=False)
    toks, starts, bos = d["tokens"].numpy(), d["doc_starts"].numpy(), int(d["bos_id"])
    sigs = set()
    for i in range(len(starts)):
        a = int(starts[i])
        b = int(starts[i + 1]) if i + 1 < len(starts) else len(toks)
        body = toks[a:b]
        if len(body) and body[0] == bos:
            body = body[1:]                       # signatures use unprefixed encodes
        sigs.add(doc_signature(body.tolist()))
    log(f"val: {len(toks):,} tokens, {len(starts):,} docs -> {len(sigs):,} dedup signatures")
    return sigs


# ---------------------------------------------------------------------------
# shard listing / download
# ---------------------------------------------------------------------------

def list_parquet(corpus: str, num_shards: int, *, revision=None, provenance=None) -> list[str]:
    from huggingface_hub import HfApi, RepoFile
    repo, pattern = PARQUET_SOURCE[corpus]
    files = retry(lambda: list(HfApi().list_repo_tree(
        repo, path_in_repo="sample/100BT", repo_type="dataset", revision=revision)),
                  f"list {repo}/{pattern}")
    files = sorted((f for f in files if isinstance(f, RepoFile) and fnmatch.fnmatch(f.path, pattern)),
                   key=lambda f: f.path)
    if not files:
        raise SystemExit(f"no shards matched {repo}/{pattern}")
    rel = [f.path for f in files]
    if num_shards and len(rel) < num_shards:
        raise SystemExit(f"{repo}: only {len(rel)} shards available, need {num_shards}")
    chosen = files[:num_shards] if num_shards else files
    if provenance is not None:
        provenance["selected_shards"] = [
            {"path": f.path, "size": f.size, "git_blob_id": f.blob_id,
             "lfs_sha256": f.lfs.sha256 if f.lfs else None}
            for f in chosen]
    return [f.path for f in chosen]


def download(repo: str, filenames: list[str], stage_dir: str, workers: int = 8, *, revision=None) -> list[str]:
    """Fetch shards in parallel. Callers download in small batches, never the whole
    listing: a corpus lists far more shards than a token budget needs, and each one is
    a couple of GB."""
    from huggingface_hub import hf_hub_download
    os.makedirs(stage_dir, exist_ok=True)

    def _get(fn):
        return retry(lambda: hf_hub_download(repo, fn, repo_type="dataset", local_dir=stage_dir,
                                             revision=revision),
                     f"download {fn}")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_get, filenames))
    return [os.path.join(stage_dir, fn) for fn in filenames]


def stream_shards(repo: str, filenames: list[str], stage_dir: str, batch: int = 4, workers: int = 8,
                  *, revision=None):
    """Yield local shard paths, downloading `batch` at a time and only as far as the
    consumer actually gets -- so a 30B-token pool costs 30B tokens of download, not the
    corpus."""
    for start in range(0, len(filenames), batch):
        chunk = filenames[start:start + batch]
        log(f"downloading shards {start + 1}-{start + len(chunk)} of <= {len(filenames)} "
            f"from {repo}")
        for path in download(repo, chunk, stage_dir, workers=workers, revision=revision):
            yield path


# ---------------------------------------------------------------------------
# per-corpus readers
# ---------------------------------------------------------------------------

def parquet_texts(path: str):
    import pyarrow.parquet as pq
    return pq.read_table(path, columns=["text"]).column("text").to_pylist()


def fill_from_texts(pool: Pool, paths, enc, threads: int, *, skip_sigs=None,
                    stream_reader=None, delete_after=False, provenance=None, stage_dir=None) -> None:
    """Tokenize documents from `paths`, in order, into `pool` until it is full.

    `paths` may be a generator, in which case shards are pulled only as far as the
    budget needs them.
    """
    n_skipped = 0
    for i, path in enumerate(paths):
        if pool.full:
            break
        texts = stream_reader(path) if stream_reader else parquet_texts(path)
        batch = []

        def flush():
            nonlocal n_skipped
            for toks in enc.encode_ordinary_batch(batch, num_threads=threads):
                if skip_sigs is not None and doc_signature(toks) in skip_sigs:
                    n_skipped += 1
                    continue
                pool.add(toks)
                if pool.full:
                    return

        for text in texts:
            batch.append(text)
            if len(batch) >= ENCODE_BATCH:
                flush()
                batch = []
                if pool.full:
                    break
        if batch and not pool.full:
            flush()
        if provenance is not None:
            provenance.setdefault("consumed_shards", {}).setdefault(pool.label, []).append({
                "path": os.path.relpath(path, stage_dir) if stage_dir else path,
                "pool_tokens_after": pool.cur, "pool_documents_after": len(pool.doc_starts),
                "skipped_validation_signatures_so_far": n_skipped,
            })
        if delete_after:
            try:
                os.remove(path)
            except OSError:
                pass
        pool.progress(f"shard {i + 1} [{os.path.basename(path)}]"
                      + (f", {n_skipped} deduped" if skip_sigs is not None else ""))


def build_fineweb(args, enc, out_dir: str) -> None:
    """The one corpus whose validation set is not a held-out shard: it is the first
    `val_tokens` of sample-10BT (the yardstick every FineWeb number in the paper is
    measured on), so train -- a prefix of the DIFFERENT sample-100BT -- is deduped
    against it document by document."""
    val_path = os.path.join(out_dir, "val.pt")
    revision, provenance = getattr(args, "revision", None), getattr(args, "provenance", None)
    if os.path.exists(val_path):
        log(f"val exists, reusing {val_path}")
    else:
        from datasets import load_dataset
        log(f"val <- streaming HuggingFaceFW/fineweb sample-10BT ({args.val_tokens:,} tokens)")
        stream = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT",
                              split="train", streaming=True, revision=revision)
        pool = Pool(args.val_tokens, "val")
        for doc in stream:
            pool.add(enc.encode_ordinary(doc["text"]))
            if pool.full:
                break
        del stream
        write_pool(val_path, *pool.finish("sample-10BT"), provenance=provenance)
        del pool

    sigs = pool_signatures(val_path)
    shards = list_parquet("fineweb", args.num_shards, revision=revision, provenance=provenance)
    pool = Pool(args.train_tokens, "train")
    fill_from_texts(pool, stream_shards(PARQUET_SOURCE["fineweb"][0], shards, args.stage_dir, revision=revision),
                    enc, args.threads, skip_sigs=sigs, delete_after=not args.keep_staged,
                    provenance=provenance, stage_dir=args.stage_dir)
    write_pool(os.path.join(out_dir, "train.pt"), *pool.finish(f"{len(shards)} listed shards"), provenance=provenance)


def build_fineweb_edu(args, enc, out_dir: str) -> None:
    """Hold out shard 0 for validation; pack training shards 1 onward in order."""
    revision, provenance = getattr(args, "revision", None), getattr(args, "provenance", None)
    shards = list_parquet("fineweb-edu", args.num_shards, revision=revision, provenance=provenance)
    if len(shards) < 2:
        raise SystemExit("FineWeb-Edu needs at least two shards: one validation and one training")
    repo = PARQUET_SOURCE["fineweb-edu"][0]
    log(f"val <- {shards[0]}; train <- remaining {len(shards) - 1} shards")

    pool = Pool(args.val_tokens, "val")
    fill_from_texts(pool, download(repo, shards[:1], args.stage_dir, revision=revision), enc, args.threads,
                    delete_after=not args.keep_staged, provenance=provenance, stage_dir=args.stage_dir)
    write_pool(os.path.join(out_dir, "val.pt"), *pool.finish("validation shard 0"), provenance=provenance)
    del pool

    pool = Pool(args.train_tokens, "train")
    fill_from_texts(pool, stream_shards(repo, shards[1:], args.stage_dir, revision=revision), enc, args.threads,
                    delete_after=not args.keep_staged, provenance=provenance, stage_dir=args.stage_dir)
    write_pool(os.path.join(out_dir, "train.pt"), *pool.finish(f"{len(shards) - 1} training shards"), provenance=provenance)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", choices=PARQUET_SOURCE, default="fineweb")
    ap.add_argument("--revision", default="main", help="Hugging Face revision; resolved once to an immutable commit")
    ap.add_argument("--out", required=True, help="output dir; pass it to train.py --data-dir")
    ap.add_argument("--train-tokens", type=int, default=DEFAULT_TRAIN_TOKENS)
    ap.add_argument("--val-tokens", type=int, default=None,
                    help="default: 10,000,000 for FineWeb; 10,200,000 for FineWeb-Edu")
    ap.add_argument("--num-shards", type=int, default=0,
                    help="shards to consume, including FineWeb-Edu's validation shard (0 = all available)")
    ap.add_argument("--stage-dir", default=None, help="raw-shard staging dir (default <out>/staged)")
    ap.add_argument("--keep-staged", action="store_true",
                    help="keep raw shards after packing (default: delete as they are consumed)")
    ap.add_argument("--threads", type=int, default=min(96, (os.cpu_count() or 8)),
                    help="tiktoken encode threads")
    args = ap.parse_args()

    if args.val_tokens is None:
        args.val_tokens = FINEWEB_VAL_TOKENS if args.corpus == "fineweb" else FINEWEB_EDU_VAL_TOKENS
    if args.train_tokens < 1 or args.val_tokens < 1 or args.threads < 1 or args.num_shards < 0:
        ap.error("token budgets and threads must be positive; num-shards must be nonnegative")
    if args.stage_dir is None:
        args.stage_dir = os.path.join(args.out, "staged")
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.stage_dir, exist_ok=True)

    # Every invocation has an independently attributable output. Raw download
    # caches can still be shared with --stage-dir; prepared pools cannot.
    metadata_path = os.path.join(args.out, "corpus.json")
    if any(os.path.exists(os.path.join(args.out, name)) for name in
           ("corpus.json", "train.pt", "val.pt", "fineweb_train.pt", "fineweb_val.pt")):
        ap.error("output directory already contains prepared data or provenance; use a separate --out")
    from huggingface_hub import HfApi
    repo = PARQUET_SOURCE[args.corpus][0]
    requested_revision = args.revision
    args.revision = retry(lambda: HfApi().dataset_info(repo, revision=requested_revision).sha,
                          f"resolve {repo}@{requested_revision}")
    if not args.revision:
        ap.error("Hugging Face returned no immutable dataset revision")
    metadata = {
        "schema_version": 1, "status": "preparing", "corpus": args.corpus,
        "repo_id": repo, "requested_revision": requested_revision, "revision": args.revision,
        "train_tokens": args.train_tokens, "val_tokens": args.val_tokens,
        "tokenizer": {"encoding": "gpt2", "method": "encode_ordinary", "bos_id": BOS_ID},
        "packages": {name: importlib.metadata.version(name) for name in
                     ("tiktoken", "numpy", "torch", "datasets", "pyarrow", "huggingface-hub")},
        "split_policy": ("sample-10BT validation prefix; sample/100BT training with 64-token validation-signature exclusion"
                         if args.corpus == "fineweb" else "sorted sample/100BT shard 0 validation; remaining shards training"),
        "selected_shards": [], "consumed_shards": {},
    }
    args.provenance = metadata

    def save_metadata():
        temporary = metadata_path + ".tmp"
        with open(temporary, "w") as handle:
            json.dump(metadata, handle, indent=2)
            handle.write("\n")
        os.replace(temporary, metadata_path)

    save_metadata()

    log(f"corpus={args.corpus} train={args.train_tokens:,} val={args.val_tokens:,} -> {args.out}")
    enc = tiktoken.get_encoding("gpt2")
    builder = build_fineweb if args.corpus == "fineweb" else build_fineweb_edu
    try:
        builder(args, enc, args.out)
    except BaseException:
        metadata["status"] = "failed"
        raise
    else:
        metadata["status"] = "complete"
    finally:
        save_metadata()
    log(f"DONE {args.corpus} -> {args.out}  (train.pt + val.pt)")


if __name__ == "__main__":
    main()
