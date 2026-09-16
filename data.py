"""Memory-mapped uint16 token pools with deterministic document and window shuffling.

Documents are concatenated with BOS separators, split into 2049-token windows,
and shifted to form 2048-token inputs and targets. Windows can begin inside a
document. Packing is cut to a multiple of loader batch size × cutoff devices
before distributing rows across ranks (256 sequences normally, 512 for d26).
"""
import ctypes
import ctypes.util
import os

import torch

from models.common import print0

BOS_ID = 50256              # GPT-2 <|endoftext|>
_PREFAULT_CHUNK_TOKENS = 1 << 27  # 128M tokens = 256 MiB per sequential prefault chunk
_CAT_FLUSH_TOKENS = 1 << 28       # flush the epoch-stream cat every ~512 MiB of output


def _ram_guard(clip_tokens, n_docs, world_size):
    """Refuse to build an epoch that would OOM the node.

    This lives here rather than in the trainer because the loader is the only thing
    that knows the resolved clip/world/local-rank numbers, and it is kept despite the
    mmap+page-release path because the failure it prevents is not local: the epoch cat
    materializes clip_tokens x 2 B per rank, times the ranks on THIS node, minutes into
    a run. On a shared node the kernel's OOM killer then picks victims across other
    people's jobs. One mis-clipped cell (a missing token limit against a 30B pool is
    ~60 GiB per rank) can wedge a node mid-campaign. Escape hatch: LOOP_DISABLE_RAM_GUARD=1.
    """
    if os.environ.get("LOOP_DISABLE_RAM_GUARD") == "1":
        return
    local_ranks = int(os.environ.get("LOCAL_WORLD_SIZE", world_size) or 1)
    per_rank = 2 * clip_tokens + 2 * clip_tokens / max(world_size, 1) + 512 * n_docs + (2 << 30)
    need = per_rank * local_ranks
    try:
        with open("/proc/meminfo") as handle:
            available = next(int(l.split()[1]) * 1024 for l in handle
                             if l.startswith("MemAvailable:"))
    except (OSError, StopIteration):
        return                                    # no /proc: nothing to guard against
    if need > 0.6 * available:
        raise MemoryError(
            f"epoch build needs ~{need / 2**30:.0f} GiB on this node "
            f"({local_ranks} local ranks x {per_rank / 2**30:.0f} GiB) but only "
            f"{available / 2**30:.0f} GiB is available. Lower --train-token-limit, or set "
            f"LOOP_DISABLE_RAM_GUARD=1 to proceed anyway."
        )


def _madvise_dontneed(region):
    """Drop the resident pages of a file-mmapped CPU tensor (best-effort).

    Only ever called on views of the torch.load(mmap=True) token mapping: those
    pages are file-backed and never written, so MADV_DONTNEED merely unmaps them
    from this process; the next access re-faults byte-identical data from page
    cache / disk. Aligned INWARD so only pages fully inside the region can be
    dropped. Pure footprint hint - returns quietly on any failure.
    """
    try:
        import ctypes
        import mmap as _mmap
        if region is None or region.numel() == 0 or not region.is_contiguous():
            return
        page = _mmap.PAGESIZE
        addr = region.data_ptr()
        end = (addr + region.numel() * region.element_size()) // page * page  # align down
        start = -(-addr // page) * page                                       # align up
        if end <= start:
            return
        libc = ctypes.CDLL(None, use_errno=True)
        libc.madvise(ctypes.c_void_p(start), ctypes.c_size_t(end - start), 4)  # MADV_DONTNEED
    except Exception:
        pass


def _prefault_release(region):
    """Sequentially fault the mapped token region into page cache, dropping this
    process's mapping as it goes (transient RSS ~256 MiB). Turns the epoch-stream
    cat's cold random 4 KiB faults (x local ranks) into one streaming read. uint16
    has no reductions in this torch build, so chunks are bit-viewed as int16 (the
    reduction value is discarded; it only forces the read).
    """
    if region is None or os.environ.get("LOOP_DATALOADER_NO_PREFAULT") == "1":
        return
    try:
        for ofs in range(0, region.numel(), _PREFAULT_CHUNK_TOKENS):
            chunk = region[ofs:ofs + _PREFAULT_CHUNK_TOKENS]
            chunk.view(torch.int16).max()
            _madvise_dontneed(chunk)
    except Exception:
        pass


def _cat_with_page_release(doc_tokens, mapped_region):
    """torch.cat(doc_tokens) with bounded residency.

    Writes straight into the preallocated epoch buffer in ~_CAT_FLUSH_TOKENS
    groups and drops the mapped source pages after each flush, so file-backed
    RSS stays ~2 GiB instead of the whole clip staying resident on top of the
    output buffer. Output is bit-identical to torch.cat(doc_tokens).
    """
    total = 0
    for d in doc_tokens:
        total += d.numel()
    out = torch.empty(total, dtype=doc_tokens[0].dtype if doc_tokens else torch.uint16)
    group, gsz, ofs = [], 0, 0
    for d in doc_tokens:
        group.append(d)
        gsz += d.numel()
        if gsz >= _CAT_FLUSH_TOKENS:
            torch.cat(group, out=out[ofs:ofs + gsz])
            ofs += gsz
            group, gsz = [], 0
            _madvise_dontneed(mapped_region)
    if group:
        torch.cat(group, out=out[ofs:ofs + gsz])
    _madvise_dontneed(mapped_region)
    return out


class DataLoader:
    """Loads flat tokens and chunks them into fixed-length batches.

    shuffle_docs=True: shuffles documents before chunking every epoch, including epoch 1.
    shuffle_seqs=True: shuffles fixed-length sequence rows every epoch.
    Validation passes both as False so it preserves prepared file order.
    token_limit=N keeps only the first N tokens from the prepared file before
    chunking, matching the stream produced by preparing a smaller token budget.

    Always yields (x, y, epoch).
    """

    def __init__(
        self,
        filepath,
        B,
        T,
        device="cuda",
        shuffle_docs=False,
        shuffle_seqs=False,
        seed=0,
        token_limit=None,
        token_limit_cutoff_devices=0,
        rank=0,
        world_size=1,
    ):
        # mmap the token file rather than reading it, and keep tokens uint16 with no
        # eager int64 copy: a plain torch.load + .long() would materialize the whole
        # pool and then a 4x copy of it in EVERY rank, which OOMs a node on the 30B
        # pools. Every structural op here (slice/cat/index/gather) and the .to()/.long()
        # at batch time work natively on uint16; pages fault in as the epoch is built.
        data = torch.load(filepath, weights_only=True, mmap=True)
        all_tokens = data["tokens"]  # uint16, mmap-backed view (no copy)
        raw_doc_starts = data["doc_starts"].long()
        bos_id = int(data["bos_id"])
        assert bos_id == BOS_ID, f"data bos_id {bos_id} != expected {BOS_ID}"
        self.source_tokens = all_tokens.numel()
        self.rank, self.world_size = rank, world_size

        def apply_token_limit(tokens, doc_starts, limit, label):
            if limit is None:
                return tokens, doc_starts, None
            limit = int(limit)
            if limit <= 0:
                raise ValueError(f"{label} must be positive when provided")
            resolved = min(limit, tokens.numel())
            tokens = tokens[:resolved]  # view, no materialization (was .contiguous())
            doc_starts = doc_starts[doc_starts < resolved].contiguous()
            if doc_starts.numel() == 0:
                raise ValueError(f"{label}={limit} excludes all document starts")
            return tokens, doc_starts, resolved

        def build_doc_tokens(tokens, doc_starts):
            doc_ends = torch.cat([doc_starts[1:], torch.tensor([tokens.numel()], dtype=doc_starts.dtype)])
            return [tokens[s:e] for s, e in zip(doc_starts.tolist(), doc_ends.tolist())]

        all_tokens, raw_doc_starts, _ = apply_token_limit(all_tokens, raw_doc_starts, token_limit, "token_limit")
        self.source_pool_tokens = all_tokens.numel()
        self.loaded_tokens = all_tokens.numel()
        _ram_guard(self.loaded_tokens, raw_doc_starts.numel(), self.world_size)
        self.doc_tokens = build_doc_tokens(all_tokens, raw_doc_starts)

        # The clipped mapped region every doc view points into. Kept so the epoch
        # cat can drop its source pages as it consumes them, and prefaulted once
        # so the first build streams the file instead of random-faulting it.
        self._mmap_region = all_tokens
        _prefault_release(self._mmap_region)

        self.device = device
        self.B = B
        self.T = T
        self.seq_size = T + 1
        self.shuffle_docs = shuffle_docs
        self.shuffle_seqs = shuffle_seqs
        self.seed = int(seed)
        self.token_limit = token_limit
        self.token_limit_cutoff_devices = int(token_limit_cutoff_devices or 0)
        if self.token_limit_cutoff_devices < 0:
            raise ValueError("token_limit_cutoff_devices must be non-negative")
        if self.token_limit is not None and self.token_limit_cutoff_devices > 0:
            if self.token_limit_cutoff_devices % self.world_size != 0:
                raise ValueError(
                    "token_limit_cutoff_devices must be divisible by the current world size "
                    f"to keep packed token cutoffs consistent; got "
                    f"token_limit_cutoff_devices={self.token_limit_cutoff_devices}, "
                    f"world_size={self.world_size}"
                )
        self.epoch = 1
        self._build_batches()

    def _epoch_generator(self, stream):
        g = torch.Generator()
        epoch_seed = (self.seed + self.epoch * 1_000_003 + stream * 101) % (2**63 - 1)
        g.manual_seed(epoch_seed)
        return g

    def _build_batches(self):
        doc_tokens = self.doc_tokens
        if self.shuffle_docs:
            g = self._epoch_generator(stream=0)
            perm = torch.randperm(len(doc_tokens), generator=g)
            doc_tokens = [doc_tokens[i] for i in perm.tolist()]

        # Materialize the epoch token stream ONCE (uint16, 2 B/token). doc_tokens are
        # mmap-backed views; this cat is the only full-stream copy, and it releases the
        # mapped source pages as it goes so they never sit resident on top of it.
        tokens = _cat_with_page_release(doc_tokens, self._mmap_region)
        num_seqs = len(tokens) // self.seq_size
        if num_seqs == 0:
            raise ValueError(
                f"not enough tokens for one sequence: have {len(tokens):,}, need at least {self.seq_size:,}"
            )
        all_seqs = tokens[:num_seqs * self.seq_size].view(num_seqs, self.seq_size)
        # Sequence permutation. Every rank draws the SAME permutation (shared seed,
        # dedicated generator stream) and then gathers only its own rows, so a rank
        # materializes its 1/world share rather than the whole shuffled pool.
        if self.shuffle_seqs:
            g = self._epoch_generator(stream=1)
            perm = torch.randperm(num_seqs, generator=g)
        else:
            perm = torch.arange(num_seqs)
        seqs_per_step = self.B * self.world_size
        cutoff_devices = (
            self.token_limit_cutoff_devices
            if self.token_limit is not None and self.token_limit_cutoff_devices > 0
            else self.world_size
        )
        seqs_per_cutoff = self.B * cutoff_devices
        cutoff_units = len(all_seqs) // seqs_per_cutoff
        if cutoff_units == 0:
            raise ValueError(
                f"not enough sequences for one cutoff unit: have {len(all_seqs):,}, "
                f"need {seqs_per_cutoff:,}"
            )
        usable = cutoff_units * seqs_per_cutoff
        num_steps = usable // seqs_per_step
        if num_steps == 0:
            raise ValueError(
                f"not enough sequences for one global step: have {len(all_seqs):,}, need {seqs_per_step:,}"
            )
        # Row at (step s, rank r, batch b) = all_seqs[perm[s*world*B + r*B + b]];
        # gather only the rows where r == self.rank.
        idx = perm[:usable].view(num_steps, self.world_size, self.B)[:, self.rank].reshape(-1)
        self.rank_data = all_seqs[idx].view(num_steps, self.B, self.seq_size)
        self.num_steps = num_steps
        self.total_tokens = usable * self.T
        self.packed_sequences = usable
        self.cutoff_devices = cutoff_devices
        self.pos = 0

    def __iter__(self):
        return self

    def _next_epoch(self):
        self.epoch += 1
        print0(f"Starting epoch {self.epoch}")
        if self.shuffle_docs or self.shuffle_seqs:
            self._build_batches()
        else:
            self.pos = 0

    def __next__(self):
        if self.pos >= self.num_steps:
            self._next_epoch()
        # rank_data is uint16 (2 B/token over PCIe), cast to int64 on-device; the
        # conversion is exact over the whole 0..65535 range.
        batch = self.rank_data[self.pos].to(self.device, non_blocking=True).long()
        self.pos += 1
        return batch[:, :-1].contiguous(), batch[:, 1:].contiguous(), self.epoch
