"""
CORE metric evaluation for base language models.

This is adapted from karpathy/nanochat's DCLM CORE evaluator, but kept
standalone so this repo does not need the full nanochat package.
"""

import csv
import hashlib
import math
import json
import os
import random
import shutil
import tempfile
import time
import urllib.request
import zipfile

import torch
import torch.distributed as dist
import torch.nn.functional as F


EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"
# The bundle is fetched from an unversioned URL, so its CONTENT is fingerprinted instead:
# sha256 over (relative path, file sha256) for all 77 files, sorted. This is the bundle
# used for the paper. Refuse a mismatch so evaluation cannot silently change suites.
EVAL_BUNDLE_FINGERPRINT = "e908e4e0e145860290d46fb37c5d1bad52bbf22da843073ff0e6b6f624250e52"

# The example-order shuffle is a fixed constant upstream (nanochat, and sandyresearch/parcae's
# receval port), which is what decides WHICH examples a --max-per-task subsample keeps. Sweeping
# it is off by default so the paper's numbers reproduce exactly; `vary_subsample=True` derives a
# per-seed shuffle from it instead, so each seed scores a DIFFERENT subsample as well as drawing
# different few-shot examples. That is the honest error bar when the metric is subsampled -- with
# the subsample frozen, the seed spread only measures few-shot variance and understates the total.
SHUFFLE_SEED = 1337
# Few-shot draw seed. `seeds=` in evaluate_core sweeps THIS one; 1234 is the single-seed default
# every earlier study in this repo ran with, so leaving it alone reproduces them bit-for-bit.
DEFAULT_FEWSHOT_SEED = 1234


def _rank():
    return dist.get_rank() if dist.is_initialized() else 0


def print0(s="", **kwargs):
    if _rank() == 0:
        print(s, **kwargs)


class EncoderCoreTokenizer:
    """Small adapter for the tokenizer API expected by the CORE evaluator.

    Wraps the GPT-2 tiktoken encoder used during training.
    """

    def __init__(self, encoder):
        self.encoder = encoder
        self.bos_id = encoder._special_tokens["<|endoftext|>"]

    def get_bos_token_id(self):
        return self.bos_id

    def __call__(self, texts, prepend=None):
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        out = []
        for text in texts:
            tokens = self.encoder.encode_ordinary(text)
            if prepend is not None:
                prepended = [prepend] if isinstance(prepend, int) else self.encoder.encode_ordinary(prepend)
                tokens = prepended + tokens
            out.append(tokens)
        return out[0] if single else out


# Back-compat alias (the class used to be tiktoken-specific).
TiktokenCoreTokenizer = EncoderCoreTokenizer


def _download_eval_bundle(eval_bundle_dir, bundle_url):
    parent_dir = os.path.dirname(os.path.abspath(eval_bundle_dir)) or "."
    os.makedirs(parent_dir, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=parent_dir) as tmpdir:
        zip_path = os.path.join(tmpdir, "eval_bundle.zip")
        urllib.request.urlretrieve(bundle_url, zip_path)
        extract_dir = os.path.join(tmpdir, "extract")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            for member in zip_ref.infolist():
                dest = os.path.abspath(os.path.join(extract_dir, member.filename))
                if not dest.startswith(os.path.abspath(extract_dir) + os.sep):
                    raise RuntimeError(f"unsafe path in eval bundle: {member.filename}")
            zip_ref.extractall(extract_dir)
        extracted_bundle = os.path.join(extract_dir, "eval_bundle")
        if not os.path.isdir(extracted_bundle):
            raise RuntimeError("eval bundle zip did not contain eval_bundle/")
        staging_dir = eval_bundle_dir + f".tmp.{os.getpid()}"
        if os.path.exists(staging_dir):
            shutil.rmtree(staging_dir)
        shutil.move(extracted_bundle, staging_dir)
        if os.path.exists(eval_bundle_dir):
            shutil.rmtree(staging_dir)
        else:
            os.replace(staging_dir, eval_bundle_dir)


def bundle_fingerprint(eval_bundle_dir):
    """Content digest of an extracted eval bundle (see EVAL_BUNDLE_FINGERPRINT)."""
    digest = hashlib.sha256()
    for root, dirs, files in os.walk(eval_bundle_dir):
        dirs.sort()
        for name in sorted(files):
            path = os.path.join(root, name)
            with open(path, "rb") as fh:
                file_digest = hashlib.sha256(fh.read()).hexdigest()
            digest.update(os.path.relpath(path, eval_bundle_dir).encode())
            digest.update(file_digest.encode())
    return digest.hexdigest()


def check_bundle_fingerprint(eval_bundle_dir):
    actual = bundle_fingerprint(eval_bundle_dir)
    if actual != EVAL_BUNDLE_FINGERPRINT:
        raise ValueError(f"CORE bundle fingerprint mismatch at {eval_bundle_dir}: "
                         f"got {actual}, expected {EVAL_BUNDLE_FINGERPRINT}")
    return actual


def ensure_eval_bundle(eval_bundle_dir="eval_bundle", bundle_url=EVAL_BUNDLE_URL):
    """Download and unpack the CORE eval bundle once, with rank 0 doing I/O."""
    eval_bundle_dir = os.path.abspath(eval_bundle_dir)
    if os.path.exists(eval_bundle_dir):
        if _rank() == 0:
            check_bundle_fingerprint(eval_bundle_dir)
        return eval_bundle_dir

    rank = _rank()
    if dist.is_initialized():
        if rank == 0 and not os.path.exists(eval_bundle_dir):
            print0(f"Downloading CORE eval bundle to {eval_bundle_dir}")
            _download_eval_bundle(eval_bundle_dir, bundle_url)
        dist.barrier()
    elif not os.path.exists(eval_bundle_dir):
        print0(f"Downloading CORE eval bundle to {eval_bundle_dir}")
        _download_eval_bundle(eval_bundle_dir, bundle_url)

    if not os.path.exists(eval_bundle_dir):
        raise RuntimeError(f"CORE eval bundle was not found at {eval_bundle_dir}")
    if rank == 0:
        check_bundle_fingerprint(eval_bundle_dir)
    return eval_bundle_dir


def render_prompts_mc(item, continuation_delimiter, fewshot_examples=None):
    fewshot_examples = fewshot_examples or []
    prefix = [
        f"{example['query']}{continuation_delimiter}{example['choices'][example['gold']]}"
        for example in fewshot_examples
    ]
    return [
        "\n\n".join(prefix + [f"{item['query']}{continuation_delimiter}{choice}"]).strip()
        for choice in item["choices"]
    ]


def render_prompts_schema(item, continuation_delimiter, fewshot_examples=None):
    fewshot_examples = fewshot_examples or []
    prefix = [
        f"{example['context_options'][example['gold']]}{continuation_delimiter}{example['continuation']}"
        for example in fewshot_examples
    ]
    return [
        "\n\n".join(prefix + [f"{context}{continuation_delimiter}{item['continuation']}"]).strip()
        for context in item["context_options"]
    ]


def render_prompts_lm(item, continuation_delimiter, fewshot_examples=None):
    fewshot_examples = fewshot_examples or []
    prefix = [
        f"{example['context'].strip()}{continuation_delimiter}{example['continuation']}"
        for example in fewshot_examples
    ]
    prompt_prefix = "\n\n".join(prefix + [f"{item['context'].strip()}{continuation_delimiter}"])
    prompt_without = prompt_prefix.strip()
    prompt_with = prompt_prefix + item["continuation"]
    return [prompt_without, prompt_with]


def continuation_start(tokenizer, full_prompt, continuation):
    """Locate the full gold continuation in a tokenized prompt.

    Tokenizing ``context`` and ``context + continuation`` separately is not
    generally prefix-stable for BPE tokenizers: the token straddling the join
    can change.  The longest common token prefix is therefore the conservative
    boundary used by ranked-classification evaluators.  It includes that
    straddling token in the continuation loss when necessary.
    """
    if not continuation or not full_prompt.endswith(continuation):
        raise ValueError("rendered prompt does not end in a non-empty continuation")
    context = full_prompt[:-len(continuation)]
    full_tokens = tokenizer(full_prompt, prepend=tokenizer.get_bos_token_id())
    context_tokens = tokenizer(context, prepend=tokenizer.get_bos_token_id())
    start = find_common_length([full_tokens, context_tokens], direction="left")
    if start >= len(full_tokens):
        raise ValueError("continuation has no scoreable tokens")
    return start


def continuation_bpb(tokenizer, token_ids, token_losses):
    """Bits per UTF-8 byte for one continuation, excluding special tokens."""
    encoder = tokenizer.encoder
    total_nats, total_bytes = 0.0, 0
    for token_id, loss in zip(token_ids.tolist(), token_losses.tolist()):
        if token_id in getattr(encoder, "_special_tokens", {}).values():
            continue
        token_bytes = encoder.decode_single_token_bytes(int(token_id))
        total_nats += float(loss)
        total_bytes += len(token_bytes)
    return total_nats / (math.log(2) * total_bytes) if total_bytes else float("nan")


def find_common_length(token_sequences, direction="left"):
    min_len = min(len(seq) for seq in token_sequences)
    indices = {
        "left": range(min_len),
        "right": range(-1, -min_len - 1, -1),
    }[direction]
    for i, idx in enumerate(indices):
        token = token_sequences[0][idx]
        if not all(seq[idx] == token for seq in token_sequences):
            return i
    return min_len


def stack_sequences(tokens, pad_token_id):
    batch_size, seq_len = len(tokens), max(len(x) for x in tokens)
    input_ids = torch.full((batch_size, seq_len), pad_token_id, dtype=torch.long)
    for i, x in enumerate(tokens):
        input_ids[i, :len(x)] = torch.tensor(x, dtype=torch.long)
    return input_ids


def batch_sequences_mc(tokenizer, prompts):
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    answer_start_idx = find_common_length(tokens, direction="left")
    start_indices = [answer_start_idx] * len(prompts)
    end_indices = [len(x) for x in tokens]
    return tokens, start_indices, end_indices


def batch_sequences_schema(tokenizer, prompts):
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    suffix_length = find_common_length(tokens, direction="right")
    end_indices = [len(x) for x in tokens]
    start_indices = [ei - suffix_length for ei in end_indices]
    return tokens, start_indices, end_indices


def batch_sequences_lm(tokenizer, prompts):
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    tokens_without, tokens_with = tokens
    start_idx, end_idx = len(tokens_without), len(tokens_with)
    assert start_idx < end_idx, "prompt without is supposed to be a prefix of prompt with"
    assert tokens_without == tokens_with[:start_idx], "prompt without is supposed to be a prefix of prompt with"
    return [tokens_with], [start_idx], [end_idx]


def _max_seq_len(model):
    if getattr(model, "max_seq_len", None) is not None:
        return model.max_seq_len
    config = getattr(model, "config", None)
    return getattr(config, "sequence_len", None)


@torch.no_grad()
def forward_model(model, input_ids, autocast_ctx=None):
    batch_size, seq_len = input_ids.size()
    if autocast_ctx is None:
        outputs = model(input_ids)
    else:
        with autocast_ctx:
            outputs = model(input_ids)
    target_ids = torch.roll(input_ids, shifts=-1, dims=1)
    losses = F.cross_entropy(
        outputs.reshape(batch_size * seq_len, -1),
        target_ids.reshape(batch_size * seq_len),
        reduction="none",
    ).view(batch_size, seq_len)
    losses[:, -1] = float("nan")
    predictions = outputs.argmax(dim=-1)
    return losses, predictions


@torch.no_grad()
def evaluate_example(idx, model, tokenizer, data, device, task_meta, autocast_ctx=None,
                     fewshot_seed=DEFAULT_FEWSHOT_SEED):
    item = data[idx]
    task_type = task_meta["task_type"]
    num_fewshot = task_meta["num_fewshot"]
    continuation_delimiter = task_meta["continuation_delimiter"]

    fewshot_examples = []
    if num_fewshot > 0:
        rng = random.Random(fewshot_seed + idx)
        available_indices = [i for i in range(len(data)) if i != idx]
        fewshot_count = min(num_fewshot, len(available_indices))
        fewshot_indices = rng.sample(available_indices, fewshot_count)
        fewshot_examples = [data[i] for i in fewshot_indices]

    if task_type == "multiple_choice":
        prompts = render_prompts_mc(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_mc(tokenizer, prompts)
        gold_idx = int(item["gold"])
        gold_full_start_idx = continuation_start(
            tokenizer, prompts[gold_idx], item["choices"][gold_idx])
    elif task_type == "schema":
        prompts = render_prompts_schema(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_schema(tokenizer, prompts)
        gold_idx = int(item["gold"])
        gold_full_start_idx = continuation_start(
            tokenizer, prompts[gold_idx], item["continuation"])
    elif task_type == "language_modeling":
        prompts = render_prompts_lm(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_lm(tokenizer, prompts)
        gold_idx = 0
        gold_full_start_idx = start_idxs[0]
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    max_tokens = _max_seq_len(model)
    if max_tokens is not None:
        new_tokens, new_start_idxs, new_end_idxs = [], [], []
        gold_crop = 0
        for prompt_idx, (t, s, e) in enumerate(zip(tokens, start_idxs, end_idxs)):
            if len(t) > max_tokens:
                num_to_crop = len(t) - max_tokens
                new_tokens.append(t[-max_tokens:])
                new_start_idxs.append(s - num_to_crop)
                new_end_idxs.append(e - num_to_crop)
                assert s - num_to_crop >= 0, "continuation was cropped away"
                assert e - num_to_crop >= 0, "continuation was cropped away"
                if prompt_idx == gold_idx:
                    gold_crop = num_to_crop
            else:
                new_tokens.append(t)
                new_start_idxs.append(s)
                new_end_idxs.append(e)
        tokens, start_idxs, end_idxs = new_tokens, new_start_idxs, new_end_idxs

        gold_full_start_idx -= gold_crop
        if gold_full_start_idx <= 0:
            raise ValueError("gold continuation boundary was cropped away")

    pad_token_id = tokenizer.get_bos_token_id()
    input_ids = stack_sequences(tokens, pad_token_id).to(device)
    losses, predictions = forward_model(model, input_ids, autocast_ctx=autocast_ctx)

    if task_type == "language_modeling":
        si = gold_full_start_idx
        ei = end_idxs[0]
        gold_token_losses = losses[0, si - 1:ei - 1]
        gold_loss = gold_token_losses.mean().item()
        gold_bpb = continuation_bpb(
            tokenizer, input_ids[0, si:ei], gold_token_losses)
        predicted_tokens = predictions[0, si - 1:ei - 1]
        actual_tokens = input_ids[0, si:ei]
        return torch.all(predicted_tokens == actual_tokens).item(), gold_loss, gold_bpb
    if task_type in ("multiple_choice", "schema"):
        mean_losses = [
            losses[i, si - 1:ei - 1].mean().item()
            for i, (si, ei) in enumerate(zip(start_idxs, end_idxs))
        ]
        pred_idx = mean_losses.index(min(mean_losses))
        ei = end_idxs[gold_idx]
        gold_token_losses = losses[gold_idx, gold_full_start_idx - 1:ei - 1]
        gold_loss = gold_token_losses.mean().item()
        gold_bpb = continuation_bpb(
            tokenizer, input_ids[gold_idx, gold_full_start_idx:ei], gold_token_losses)
        return pred_idx == gold_idx, gold_loss, gold_bpb
    raise ValueError(f"Unsupported task type: {task_type}")


def evaluate_task(model, tokenizer, data, device, task_meta, autocast_ctx=None,
                  fewshot_seed=DEFAULT_FEWSHOT_SEED):
    rank = _rank()
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    correct = torch.zeros(len(data), dtype=torch.float32, device=device)
    gold_losses = torch.zeros(len(data), dtype=torch.float32, device=device)
    gold_bpbs = torch.zeros(len(data), dtype=torch.float32, device=device)
    for idx in range(rank, len(data), world_size):
        is_correct, gold_loss, gold_bpb = evaluate_example(
            idx, model, tokenizer, data, device, task_meta,
            autocast_ctx=autocast_ctx, fewshot_seed=fewshot_seed)
        correct[idx] = float(is_correct)
        gold_losses[idx] = float(gold_loss)
        gold_bpbs[idx] = float(gold_bpb)
    if world_size > 1:
        dist.barrier()
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(gold_losses, op=dist.ReduceOp.SUM)
        dist.all_reduce(gold_bpbs, op=dist.ReduceOp.SUM)
    return correct.mean().item(), gold_losses.mean().item(), gold_bpbs.mean().item()


def evaluate_core(
    model,
    tokenizer,
    device,
    max_per_task=-1,
    eval_bundle_dir="eval_bundle",
    bundle_url=EVAL_BUNDLE_URL,
    autocast_ctx=None,
    seeds=None,
    vary_subsample=False,
):
    """
    Evaluate a base model on the DCLM CORE benchmark.

    Returns raw and centered task accuracies, full-gold-continuation NLL/BPB,
    per-seed diagnostics, and their task-macro aggregates.

    `seeds` sweeps the FEW-SHOT DRAW seed (sandyresearch/parcae's eval_configs/eval-core.yaml
    runs `seeds: [1234, 2345, 3456]`). The top-level keys stay the seed-MEAN, so a caller that
    passes nothing gets exactly the historical single-seed-1234 numbers; `per_seed` and the
    `*_std` keys carry the spread. Zero-shot tasks never draw a few-shot example, so they are
    evaluated ONCE and the identical result is reused for every seed -- that is not an
    approximation, `evaluate_example` provably never touches the rng when num_fewshot == 0.
    """
    import yaml

    if max_per_task != -1 and max_per_task <= 0:
        raise ValueError("max_per_task must be -1 or positive")
    seeds = [DEFAULT_FEWSHOT_SEED] if seeds is None else [int(s) for s in seeds]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be nonempty and distinct")
    eval_bundle_dir = ensure_eval_bundle(eval_bundle_dir, bundle_url)
    config_path = os.path.join(eval_bundle_dir, "core.yaml")
    data_base_path = os.path.join(eval_bundle_dir, "eval_data")
    eval_meta_data = os.path.join(eval_bundle_dir, "eval_meta_data.csv")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    tasks = config["icl_tasks"]

    random_baselines = {}
    with open(eval_meta_data, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            random_baselines[row["Eval Task"]] = float(row["Random baseline"])

    def _center(label, accuracy):
        random_baseline = random_baselines[label]
        return (accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline)

    def _mean(xs):
        return sum(xs) / len(xs)

    def _std(xs):
        if len(xs) < 2:
            return 0.0
        mu = _mean(xs)
        return (sum((x - mu) ** 2 for x in xs) / len(xs)) ** 0.5

    per_seed = {
        s: {"results": {}, "losses": {}, "bpbs": {}, "centered_results": {}}
        for s in seeds
    }
    results, losses, bpbs, centered_results = {}, {}, {}, {}
    task_counts = {}
    results_std, losses_std, bpbs_std = {}, {}, {}
    for task in tasks:
        start_time = time.time()
        label = task["label"]
        task_meta = {
            "task_type": task["icl_task_type"],
            "dataset_uri": task["dataset_uri"],
            "num_fewshot": task["num_fewshot"][0],
            "continuation_delimiter": task.get("continuation_delimiter", " "),
        }
        print0(f"Evaluating: {label} ({task_meta['num_fewshot']}-shot, type: {task_meta['task_type']})... ", end="")

        data_path = os.path.join(data_base_path, task_meta["dataset_uri"])
        with open(data_path, "r", encoding="utf-8") as f:
            data = [json.loads(line.strip()) for line in f]

        if not data:
            raise ValueError(f"empty CORE task: {label}")
        task_counts[label] = min(len(data), max_per_task) if max_per_task > 0 else len(data)

        def subsample(seed):
            ordered = list(data)
            random.Random(f"{SHUFFLE_SEED}:{seed}" if vary_subsample else SHUFFLE_SEED).shuffle(ordered)
            return ordered[:max_per_task] if max_per_task > 0 else ordered

        # With a frozen subsample a zero-shot task is seed-independent, so it is evaluated once
        # and fanned out. Once the subsample varies, every task genuinely differs per seed.
        subsampled = max_per_task > 0 and len(data) > max_per_task
        seed_independent = task_meta["num_fewshot"] == 0 and not (vary_subsample and subsampled)
        run_seeds = seeds[:1] if seed_independent else seeds
        for seed in run_seeds:
            accuracy, loss, bpb = evaluate_task(
                model, tokenizer, subsample(seed), device, task_meta,
                autocast_ctx=autocast_ctx, fewshot_seed=seed)
            for s in (seeds if seed_independent else [seed]):
                per_seed[s]["results"][label] = accuracy
                per_seed[s]["losses"][label] = loss
                per_seed[s]["bpbs"][label] = bpb
                per_seed[s]["centered_results"][label] = _center(label, accuracy)

        accs = [per_seed[s]["results"][label] for s in seeds]
        lsss = [per_seed[s]["losses"][label] for s in seeds]
        task_bpbs = [per_seed[s]["bpbs"][label] for s in seeds]
        results[label] = _mean(accs)
        losses[label] = _mean(lsss)
        bpbs[label] = _mean(task_bpbs)
        centered_results[label] = _center(label, results[label])
        results_std[label] = _std(accs)
        losses_std[label] = _std(lsss)
        bpbs_std[label] = _std(task_bpbs)
        elapsed = time.time() - start_time
        print0(f"accuracy: {results[label]:.4f} (sd {results_std[label]:.4f}) | "
               f"gold NLL: {losses[label]:.6f} | gold BPB: {bpbs[label]:.6f} | "
               f"centered: {centered_results[label]:.4f} | "
               f"time: {elapsed:.2f}s")

    core_metric_per_seed = {
        s: sum(per_seed[s]["centered_results"].values()) / len(per_seed[s]["centered_results"])
        for s in seeds
    }
    # A partial-task NLL is not the paper's 22-task metric.
    dropped = sorted({label for s in seeds for label, v in per_seed[s]["losses"].items()
                      if v is None or not math.isfinite(v)})
    if dropped:
        raise ValueError(f"non-finite CORE answer NLL for tasks: {', '.join(dropped)}")
    core_loss_per_seed = {}
    for s in seeds:
        finite = [v for label, v in per_seed[s]["losses"].items()
                  if label not in dropped and v is not None and math.isfinite(v)]
        core_loss_per_seed[s] = sum(finite) / len(finite) if finite else float("nan")
    return {
        "task_counts": task_counts,
        "num_examples": sum(task_counts.values()),
        "results": results,
        "losses": losses,
        "bpbs": bpbs,
        "centered_results": centered_results,
        # centering is affine in accuracy, so mean-of-centered == centered-of-mean; both the
        # per-seed spread and the pooled number below are therefore self-consistent.
        "core_metric": sum(centered_results.values()) / len(centered_results),
        "core_loss": (lambda f: sum(f) / len(f) if f else float("nan"))(
            [v for label, v in losses.items() if label not in dropped
             and v is not None and math.isfinite(v)]),
        "core_bpb": (lambda f: sum(f) / len(f) if f else float("nan"))(
            [v for v in bpbs.values() if v is not None and math.isfinite(v)]),
        "seeds": seeds,
        "shuffle_seed": SHUFFLE_SEED,
        "core_loss_dropped_tasks": dropped,
        "vary_subsample": vary_subsample,
        "per_seed": {str(s): per_seed[s] for s in seeds},
        "results_std": results_std,
        "losses_std": losses_std,
        "bpbs_std": bpbs_std,
        "core_metric_per_seed": {str(s): v for s, v in core_metric_per_seed.items()},
        "core_loss_per_seed": {str(s): v for s, v in core_loss_per_seed.items()},
        "core_metric_std": _std(list(core_metric_per_seed.values())),
    }
