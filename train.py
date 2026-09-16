"""Train a Transformer with a fixed or growing core.

Run with torchrun. Each run writes its configuration, loss measurements, and
optional final checkpoint under runs/<run-name>/.
"""
import argparse
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("WANDB_MODE", "offline")
import gc
import hashlib
import math
import time
import json
import sys
import shutil
from contextlib import nullcontext

def build_parser():
    parser = argparse.ArgumentParser(description="Train a fixed or growing Transformer")
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data/fineweb"))
    parser.add_argument("--train-token-limit", type=int)
    parser.add_argument("--val-token-limit", type=int, default=10_000_000)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--device-batch-size", type=int, default=4)
    parser.add_argument("--low-k-device-batch-size", type=int,
                        help="microbatch before growth (default: --device-batch-size)")
    parser.add_argument("--optimizer-transport", choices=("stacked", "owner_bucketed"), default="stacked",
                        help="gradient distribution; d26 uses the paper's memory-bounded owner buckets")
    parser.add_argument("--log-microbatch-losses", action="store_true",
                        help="write each rank's raw microbatch losses for trajectory diagnostics")
    parser.add_argument("--total-batch-size", type=int, default=524288, help="tokens per optimizer step")
    parser.add_argument("--token-limit-cutoff-devices", type=int, default=64, help="keep device_batch_size * cutoff_devices = 256")

    parser.add_argument("--n_layer", type=int, default=8, help="base depth: prelude + one core + coda")
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_embd", type=int, default=1024)
    parser.add_argument("--depth-scale-mode", choices=("none", "loop", "dep"), default="none", help="plain, tied core, or untied cores")
    parser.add_argument("--core-repetitions", type=int, default=1)
    parser.add_argument("--core-layer-count", type=int)
    parser.add_argument("--prelude-layer-count", type=int)
    parser.add_argument("--coda-layer-count", type=int)
    parser.add_argument("--recurrence-rmsnorm", choices=("none", "recurrence_only"), default="none")
    parser.add_argument("--recurrence-emb-alpha", type=float, default=0)
    parser.add_argument("--decoder-inject", action="store_true")
    parser.add_argument("--core-repetition-schedule", choices=("none", "late"), default="none")
    parser.add_argument("--core-repetition-low", type=int, default=2)
    parser.add_argument("--core-repetition-crossover-fraction", type=float, default=.5)
    parser.add_argument("--lsched-lr-warmup-steps", type=int, default=0)

    parser.add_argument("--matrix-lr", type=float, default=.04)
    parser.add_argument("--embedding-lr", type=float, default=.032)
    parser.add_argument("--unembedding-lr", type=float, default=.001)
    parser.add_argument("--lr_multiplier", type=float, default=1)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--matrix-weight-decay", type=float)
    parser.add_argument("--embedding-weight-decay", type=float)
    parser.add_argument("--unembedding-weight-decay", type=float)
    parser.add_argument("--embedding-wd-skip-norm", type=float, default=0)
    parser.add_argument("--adam-betas", type=float, nargs=2, default=(.8, .95))
    parser.add_argument("--adam-eps", type=float, default=1e-10)
    parser.add_argument("--warmup-steps", type=int, default=40)
    parser.add_argument("--warmdown-ratio", type=float, default=.6)
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--stop-after-steps", type=int, default=os.environ.get("STOP_AFTER_STEPS"),
                        help="stop early without changing the data prefix, LR horizon, or growth schedule")
    parser.add_argument("--lr-schedule-steps", type=int)
    parser.add_argument("--output-multiplier", type=float, default=1)
    parser.add_argument("--residual-branch-multiplier", type=float, default=1)
    parser.add_argument("--wte-init-std", type=float, default=.8)
    parser.add_argument("--uniform-init-scale", type=float, default=3**.5)

    parser.add_argument("--run-name")
    parser.add_argument("--runs-dir", default=os.environ.get("RUNS_DIR", "runs"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-result", default="")
    parser.add_argument("--save-final-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-project", default="loop")
    parser.add_argument("--wandb-group", default=os.environ.get("WANDB_GROUP", "scaling_ladders"))
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    return parser


def get_lr_multiplier_for(it, schedule_steps, warmdown_ratio, warmup_steps, final_lr_frac=0.0):
    """WSD: linear warmup, flat, then linear decay over the final warmdown_ratio
    of the horizon, ending at final_lr_frac of the base LR. Returns the multiplier
    on each group's base LR."""
    warmup = min(max(warmup_steps, 0), schedule_steps)
    if warmup > 0 and it < warmup:
        return (it + 1) / warmup

    warmdown = round(warmdown_ratio * schedule_steps)
    if warmdown <= 0:
        return 1.0
    decay_start = max(warmup, schedule_steps - warmdown)
    if it < decay_start:
        return 1.0

    decay_steps = max(schedule_steps - decay_start, 1)
    if decay_steps == 1:
        progress = 1.0
    else:
        progress = (it - decay_start) / (decay_steps - 1)
    progress = min(max(progress, 0.0), 1.0)
    return 1.0 + (final_lr_frac - 1.0) * progress


def resolve_stop_steps(max_train_steps, stop_after_steps):
    """An execution cap independent of the experiment's optimization schedule."""
    if stop_after_steps is not None and stop_after_steps < 1:
        raise ValueError("--stop-after-steps must be positive")
    return max_train_steps if stop_after_steps is None else min(max_train_steps, stop_after_steps)


def parse_args():
    """Parse settings and validate the two-phase microbatch layout."""
    args = build_parser().parse_args()
    args.adam_betas = tuple(args.adam_betas)
    if float(args.core_repetitions).is_integer():
        args.core_repetitions = int(args.core_repetitions)
    if args.low_k_device_batch_size is None:
        args.low_k_device_batch_size = args.device_batch_size
    if (args.device_batch_size < 1 or args.low_k_device_batch_size < args.device_batch_size
            or args.low_k_device_batch_size % args.device_batch_size):
        raise ValueError("low-K microbatch must be a positive multiple of --device-batch-size")
    return args


args = parse_args()

from corpus_manifest import pool_paths, read_preparation_manifest

# Reject a known incomplete preparation before importing GPU/model libraries.
_preparation_manifest, data_provenance = read_preparation_manifest(args.data_dir)
_data_paths = pool_paths(args.data_dir)

# Keep heavyweight imports after argparse so `train.py --help` stays fast and
# does not initialize ML/library integrations.
from data import DataLoader

import torch
import torch._dynamo
torch._dynamo.config.cache_size_limit = 64
import torch.distributed as dist
import wandb
import tiktoken

from models import GPTConfig, TransformerGPT, dep_stack_grow_init, is_flash_attention_3_available
from models.flash_attention import FLASH_ATTENTION_REVISION

_script_start = time.time()

# =============================================================================
# Resolved configuration
#
# Everything a training step reads is resolved once, here, from the parsed flags.
# Constants that no run in the paper varies are written as constants.
# =============================================================================

DEPTH = args.n_layer
N_EMBD = args.n_embd
N_HEAD = args.n_head
HEAD_DIM = N_EMBD // N_HEAD
DEPTH_SCALE_MODE = args.depth_scale_mode
CORE_REPETITIONS = args.core_repetitions
CORE_REPETITION_SCHEDULE = args.core_repetition_schedule
CORE_REPETITION_CROSSOVER_FRACTION = args.core_repetition_crossover_fraction
CORE_REPETITION_LOW = args.core_repetition_low
DEFAULT_CORE_LAYER_COUNT = 4 if DEPTH_SCALE_MODE != "none" else max(0, DEPTH - 8)
CORE_LAYER_COUNT = args.core_layer_count if args.core_layer_count is not None else DEFAULT_CORE_LAYER_COUNT
# Core residual branches are NOT damped by the loop count: every run in the paper
# fixes this at 1.0 (the --core-residual-branch-scale setting, always
# passed as 1). The residual scale that IS tuned is --residual-branch-multiplier.
CORE_RESIDUAL_BRANCH_SCALE = 1.0
EFFECTIVE_DEPTH = (
    DEPTH - CORE_LAYER_COUNT + round(CORE_LAYER_COUNT * CORE_REPETITIONS)
    if DEPTH_SCALE_MODE != "none" else DEPTH
)
MAX_SEQ_LEN = 2048
EVAL_TOKENS = 10_000_000    # ceiling on the validation pass, whatever the val pool holds
WINDOW_PATTERN = "L"   # full causal attention everywhere (no sliding-window layers)
TOTAL_BATCH_SIZE = args.total_batch_size
DATA_DIR = args.data_dir
BOS_ID = 50256  # <|endoftext|>
RUNS_DIR = args.runs_dir

# Three learning rates, one per parameter class, all scaled by the tuned multiplier.
MATRIX_LR = args.matrix_lr * args.lr_multiplier
EMBEDDING_LR = args.embedding_lr * args.lr_multiplier
UNEMBEDDING_LR = args.unembedding_lr * args.lr_multiplier

WEIGHT_DECAY = args.weight_decay
MATRIX_WEIGHT_DECAY = WEIGHT_DECAY if args.matrix_weight_decay is None else args.matrix_weight_decay
EMBEDDING_WEIGHT_DECAY = WEIGHT_DECAY if args.embedding_weight_decay is None else args.embedding_weight_decay
UNEMBEDDING_WEIGHT_DECAY = WEIGHT_DECAY if args.unembedding_weight_decay is None else args.unembedding_weight_decay
EMBEDDING_WD_SKIP_NORM = args.embedding_wd_skip_norm
args.matrix_weight_decay = MATRIX_WEIGHT_DECAY
args.embedding_weight_decay = EMBEDDING_WEIGHT_DECAY
args.unembedding_weight_decay = UNEMBEDDING_WEIGHT_DECAY
ADAM_BETAS = args.adam_betas
ADAM_EPS = args.adam_eps
WARMUP_STEPS = args.warmup_steps
WARMDOWN_RATIO = args.warmdown_ratio
FINAL_LR_FRAC = 0.0

# =============================================================================
# Utilities
# =============================================================================

def get_dist_info():
    if all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE")):
        return True, int(os.environ['RANK']), int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE'])
    return False, 0, 0, 1

def print0(s="", **kwargs):
    if int(os.environ.get('RANK', 0)) == 0:
        print(s, **kwargs)

class DummyWandb:
    def __init__(self): self.summary = {}
    def log(self, *a, **kw): pass
    def finish(self): pass

class TeeStream:
    """Mirror stdout/stderr into the run's terminal.log."""
    def __init__(self, *streams):
        self.streams = streams
        self.encoding = getattr(streams[0], "encoding", "utf-8")
    def write(self, data):
        for stream in self.streams: stream.write(data)
        return len(data)
    def flush(self):
        for stream in self.streams: stream.flush()
    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)
    def fileno(self):
        return self.streams[0].fileno()

def resolve_run_dir(run_name):
    if run_name:
        return run_name, os.path.join(RUNS_DIR, run_name)
    name = time.strftime('%Y%m%d_%H%M%S')
    return name, os.path.join(RUNS_DIR, name)

# =============================================================================
# Dataloader: shuffled documents and fixed-length windows
# =============================================================================

# =============================================================================
# Loss evaluation
# =============================================================================

@torch.no_grad()
def evaluate_bpb(model, batches, steps, token_bytes, bp_steps=None, active_core_repetitions=None):
    """Compute bits per byte and mean cross-entropy loss on a set of batches."""
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=model.get_device())
    total_bytes = torch.tensor(0, dtype=torch.int64, device=model.get_device())
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=model.get_device())
    total_tokens = torch.tensor(0, dtype=torch.int64, device=model.get_device())
    batch_iter = iter(batches)
    for _ in range(steps):
        x, y, _ = next(batch_iter)
        model_kwargs = {"loss_reduction": "none"}
        if active_core_repetitions is not None:
            model_kwargs["active_core_repetitions"] = active_core_repetitions
        loss2d = model(x, y, **model_kwargs).view(-1)
        y = y.view(-1)
        mask = y != -1
        total_loss += loss2d[mask].sum()
        total_tokens += mask.sum()
        num_bytes2d = token_bytes[y]
        total_nats += (loss2d * (num_bytes2d > 0)).sum()
        total_bytes += num_bytes2d.sum()
    if dist.is_initialized():
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)
    total_nats, total_bytes = total_nats.item(), total_bytes.item()
    total_loss, total_tokens = total_loss.item(), total_tokens.item()
    bpb = total_nats / (math.log(2) * total_bytes) if total_bytes > 0 else float('inf')
    loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    return bpb, loss


# =============================================================================
# Training
# =============================================================================

# Compute init
ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
master_process = ddp_rank == 0
torch.manual_seed(args.seed)

if ddp and torch.cuda.is_available():
    device = torch.device("cuda", ddp_local_rank)
    torch.cuda.set_device(device)
    torch.cuda.manual_seed(args.seed)
    dist.init_process_group(backend="nccl", device_id=device)
    dist.barrier()
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device_type = device.type
autocast_ctx = (
    torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
    if device_type == "cuda"
    else nullcontext()
)
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

# GPU info for MFU
gpu_peak_flops = float('inf')
if device_type == "cuda":
    gpu_name = torch.cuda.get_device_name(0).lower()
    if "h100" in gpu_name: gpu_peak_flops = 989e12
    elif "a100" in gpu_name: gpu_peak_flops = 312e12
    elif "4090" in gpu_name: gpu_peak_flops = 165.2e12

# FA3 status. Every number in the paper was produced with FlashAttention-3 on H100s.
# The kernels come from the HF Hub at import time, and a slow or failed fetch used to
# fall back to PyTorch SDPA SILENTLY -- same loss curve shape, different numerics and
# ~2x the step time, discovered only after a sweep had burned. So: hard error by
# default. Set LOOP_ALLOW_SDPA=1 to run the fallback deliberately (non-Hopper GPUs).
if is_flash_attention_3_available():
    print0("Using Flash Attention 3 (Hopper GPU detected)")
elif os.environ.get("LOOP_ALLOW_SDPA") == "1":
    print0("WARNING: FlashAttention-3 unavailable; running the PyTorch SDPA fallback "
           "(LOOP_ALLOW_SDPA=1). Numerics will NOT match the paper.")
else:
    raise SystemExit(
        "FlashAttention-3 is not available.\n"
        "  It requires a Hopper GPU (H100) and the `kernels` package, which fetches\n"
        "  kernels-community/flash-attn3 from the HF Hub on first use.\n"
        "  Set LOOP_ALLOW_SDPA=1 to run the PyTorch SDPA fallback anyway (different\n"
        "  numerics and much slower -- not what the paper's numbers were produced with)."
    )

# Run / logging paths
run_name, run_dir = resolve_run_dir(args.run_name)
if dist.is_initialized():
    shared = [run_name]
    dist.broadcast_object_list(shared, src=0)
    run_name = shared[0]
    run_dir = os.path.join(RUNS_DIR, run_name)
checkpoints_dir = os.path.join(run_dir, "checkpoints")
terminal_log_path = os.path.join(run_dir, "terminal.log")
stdout_orig = sys.stdout
stderr_orig = sys.stderr
artifacts_log_f = None
result_path = os.path.join(run_dir, "result.json")
existing_run = [master_process and any(os.path.exists(os.path.join(run_dir, name))
                                     for name in ("terminal.log", "result.json", "trajectory.jsonl"))]
if dist.is_initialized():
    dist.broadcast_object_list(existing_run, src=0)
if existing_run[0]:
    raise FileExistsError(f"run already has outputs: {run_dir}; use a new --run-name or --runs-dir")
if args.log_microbatch_losses:
    # Other nodes may use local output directories for their rank diagnostics.
    os.makedirs(run_dir, exist_ok=True)
if master_process:
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "wandb"), exist_ok=True)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    shutil.copy2(__file__, os.path.join(run_dir, "train.py"))
    for name in ("data.py", "corpus_manifest.py"):
        shutil.copy2(os.path.join(script_dir, name), os.path.join(run_dir, name))
    for artifact_dir in ("models",):
        src_dir = os.path.join(script_dir, artifact_dir)
        if os.path.isdir(src_dir):
            shutil.copytree(
                src_dir,
                os.path.join(run_dir, artifact_dir),
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
    dependency_files = []
    for name in ("pyproject.toml", "uv.lock"):
        if os.path.isfile(os.path.join(script_dir, name)):
            shutil.copy2(os.path.join(script_dir, name), os.path.join(run_dir, name))
            dependency_files.append(name)
    source_files = ["train.py", "data.py", "corpus_manifest.py"] + dependency_files + [
        os.path.join("models", name) for name in sorted(os.listdir(os.path.join(script_dir, "models")))
        if name.endswith(".py")
    ]
    source_hashes = {}
    for name in source_files:
        with open(os.path.join(run_dir, name), "rb") as handle:
            source_hashes[name] = hashlib.sha256(handle.read()).hexdigest()
    with open(os.path.join(run_dir, "source_manifest.json"), "w") as handle:
        json.dump(source_hashes, handle, indent=2)
    artifacts_log_f = open(terminal_log_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.stdout, artifacts_log_f)
    sys.stderr = TeeStream(sys.stderr, artifacts_log_f)

print0(f"Data preparation provenance: {json.dumps(data_provenance, sort_keys=True)}")

# wandb
_wandb_kwargs = {"project": args.wandb_project, "name": run_name,
                 "dir": os.path.join(run_dir, "wandb"), "config": dict(vars(args))}
# wandb.init defaults to a 90 s handshake and does NOT read WANDB_INIT_TIMEOUT into
# Settings() itself. When many cells start at once the API rate-limits and the init times
# out, killing an otherwise healthy run, so the env var is honoured explicitly.
_wandb_init_timeout = float(os.environ.get("WANDB_INIT_TIMEOUT", "90"))
if _wandb_init_timeout != 90.0:
    _wandb_kwargs["settings"] = wandb.Settings(init_timeout=_wandb_init_timeout)
if args.wandb_group:
    _wandb_kwargs["group"] = args.wandb_group
wandb_run = DummyWandb() if (args.no_wandb or not master_process) else wandb.init(**_wandb_kwargs)

# Print hyperparameters
print0(f"--- Hyperparameters ---")
print0(f"  n_layer={DEPTH}, effective_depth={EFFECTIVE_DEPTH}, n_embd={N_EMBD}, n_head={N_HEAD}, head_dim={HEAD_DIM}")
print0(
    f"  depth_scale_mode={DEPTH_SCALE_MODE}, core_repetitions={CORE_REPETITIONS}, "
    f"core_layer_count={CORE_LAYER_COUNT}, prelude_layer_count={args.prelude_layer_count}, "
    f"coda_layer_count={args.coda_layer_count}"
)
print0(f"  core_repetition_schedule={CORE_REPETITION_SCHEDULE}, decoder_inject={args.decoder_inject}")
print0(
    f"  core_residual_branch_scale={CORE_RESIDUAL_BRANCH_SCALE}, "
    f"recurrence_rmsnorm={args.recurrence_rmsnorm}, "
    f"recurrence_emb_alpha={args.recurrence_emb_alpha}"
)
print0(f"  magic_norm=True, residual_branch_multiplier_init={args.residual_branch_multiplier}")
print0(f"  seq_len={MAX_SEQ_LEN}, window_pattern={WINDOW_PATTERN}")
print0(f"  total_batch_size={TOTAL_BATCH_SIZE}, device_batch_size={args.device_batch_size}, low_k_device_batch_size={args.low_k_device_batch_size}, optimizer_transport={args.optimizer_transport}")
print0(f"  matrix_lr={MATRIX_LR}, embedding_lr={EMBEDDING_LR}, unembedding_lr={UNEMBEDDING_LR}")
print0(
    f"  weight_decay={WEIGHT_DECAY}, matrix_weight_decay={MATRIX_WEIGHT_DECAY}, "
    f"embedding_weight_decay={EMBEDDING_WEIGHT_DECAY}, "
    f"unembedding_weight_decay={UNEMBEDDING_WEIGHT_DECAY}, adam_betas={ADAM_BETAS}, adam_eps={ADAM_EPS}"
)
print0(f"  embedding_wd_skip_norm={EMBEDDING_WD_SKIP_NORM}")
print0(f"  lr_schedule=wsd, warmup_steps={WARMUP_STEPS}, warmdown_ratio={WARMDOWN_RATIO}, final_lr_frac={FINAL_LR_FRAC}")
print0(f"  output_multiplier={args.output_multiplier}, wte_init_std={args.wte_init_std}, "
       f"uniform_init_scale={args.uniform_init_scale}")
print0(f"  num_epochs={args.num_epochs}, seed={args.seed}")
print0(f"  max_train_steps={args.max_train_steps}, lr_schedule_steps={args.lr_schedule_steps}")
print0(
    f"  train_token_limit={args.train_token_limit}, val_token_limit={args.val_token_limit}, "
    f"token_limit_cutoff_devices={args.token_limit_cutoff_devices}"
)
print0(f"  run={run_name}")
print0(f"  run_dir={run_dir}")
print0(f"  wandb_enabled={not args.no_wandb}")
print0(f"-----------------------")


def build_token_bytes(encoder, device):
    eot_id = encoder._special_tokens['<|endoftext|>']
    token_bytes_list = []
    for token_id in range(encoder.n_vocab):
        if token_id == eot_id:
            token_bytes_list.append(0)
        else:
            token_bytes_list.append(len(encoder.decode_single_token_bytes(token_id)))
    return torch.tensor(token_bytes_list, dtype=torch.int32, device=device)


def build_model_config(vocab_size):
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN,
        vocab_size=vocab_size,
        n_layer=DEPTH,
        n_head=N_HEAD,
        n_kv_head=N_HEAD,
        n_embd=N_EMBD,
        window_pattern=WINDOW_PATTERN,
        output_multiplier=args.output_multiplier,
        wte_init_std=args.wte_init_std,
        uniform_init_scale=args.uniform_init_scale,
        residual_branch_multiplier=args.residual_branch_multiplier,
        depth_scale_mode=args.depth_scale_mode,
        core_repetitions=args.core_repetitions,
        core_layer_count=args.core_layer_count,
        prelude_layer_count=args.prelude_layer_count,
        coda_layer_count=args.coda_layer_count,
        recurrence_rmsnorm=args.recurrence_rmsnorm,
        recurrence_emb_alpha=args.recurrence_emb_alpha,
        decoder_inject=args.decoder_inject,
        matrix_lr=MATRIX_LR,
        embedding_lr=EMBEDDING_LR,
        unembedding_lr=UNEMBEDDING_LR,
        weight_decay=WEIGHT_DECAY,
        matrix_weight_decay=MATRIX_WEIGHT_DECAY,
        embedding_weight_decay=EMBEDDING_WEIGHT_DECAY,
        unembedding_weight_decay=UNEMBEDDING_WEIGHT_DECAY,
        embedding_wd_skip_norm=EMBEDDING_WD_SKIP_NORM,
        adam_betas=ADAM_BETAS,
        adam_eps=ADAM_EPS,
    )


# Load GPT-2 tokenizer and compute token_bytes for BPB evaluation
encoder = tiktoken.get_encoding("gpt2")
vocab_size = encoder.n_vocab  # 50257
print0(f"Vocab size: {vocab_size:,}")
token_bytes = build_token_bytes(encoder, device)

# Build model
config = build_model_config(vocab_size)
with torch.device("meta"):
    model = TransformerGPT(config)
model.to_empty(device=device)
model.init_weights()

param_summary = model.parameter_summary()
num_flops_per_token = model.estimate_flops()
flops_by_core_repetitions = (
    {k: model.estimate_flops(active_core_repetitions=k) for k in range(1, CORE_REPETITIONS + 1)}
    if DEPTH_SCALE_MODE in ("loop", "dep") else {1: num_flops_per_token}
)
estimated_training_flops = 0
parts = ", ".join(f"{name}: {count:,}" for name, count in param_summary["parts"].items())
print0(f"Parameters: {param_summary['total']:,} ({parts})")
print0(f"FLOPs per token: {num_flops_per_token:e}")

# Compile
orig_model = model
if args.no_compile:
    print0("torch.compile disabled")
else:
    model = torch.compile(model, dynamic=False)

# Optimizer
optimizer = orig_model.setup_optimizer(optimizer_transport=args.optimizer_transport)
# Bucket statistics contain collectives; every rank must participate here.
optimizer_distribution = optimizer.zero2_stats() if getattr(optimizer, "uses_zero2", False) else None
if optimizer_distribution is not None:
    print0(f"Optimizer distribution: {json.dumps(optimizer_distribution, sort_keys=True)}")

# Dataloaders: paths were checked before GPU initialization.
train_path = str(_data_paths['train'])
val_path = str(_data_paths['val'])


def build_train_loader():
    return DataLoader(
        train_path,
        args.low_k_device_batch_size,
        MAX_SEQ_LEN,
        device=device,
        shuffle_docs=True,
        shuffle_seqs=True,
        seed=args.seed,
        token_limit=args.train_token_limit,
        token_limit_cutoff_devices=args.token_limit_cutoff_devices,
        rank=ddp_rank,
        world_size=ddp_world_size,
    )


def build_val_loader():
    return DataLoader(
        val_path,
        args.device_batch_size,
        MAX_SEQ_LEN,
        device=device,
        shuffle_docs=False,
        shuffle_seqs=False,
        seed=args.seed,
        token_limit=args.val_token_limit,
        token_limit_cutoff_devices=args.token_limit_cutoff_devices,
        rank=ddp_rank,
        world_size=ddp_world_size,
    )


train_loader = build_train_loader()
if args.train_token_limit is not None and train_loader.source_tokens < args.train_token_limit:
    raise ValueError("training pool is smaller than --train-token-limit; prepare a larger pool")
TOKENS_PER_EPOCH = train_loader.total_tokens
x, y, current_epoch = next(train_loader)
print0(
    f"Train file tokens: {train_loader.source_tokens:,}; "
    f"source-pool train tokens: {train_loader.source_pool_tokens:,}; "
    f"active train tokens before packing: {train_loader.loaded_tokens:,}"
)
print0(
    f"Packed train tokens per epoch: {train_loader.total_tokens:,} "
    f"({train_loader.packed_sequences:,} sequences; cutoff devices={train_loader.cutoff_devices})"
)
_val_info_loader = build_val_loader()
if args.val_token_limit is not None and _val_info_loader.source_tokens < args.val_token_limit:
    raise ValueError("validation pool is smaller than --val-token-limit")
VAL_PACKED_TOKENS = _val_info_loader.total_tokens
VAL_PACKED_SEQUENCES = _val_info_loader.packed_sequences
print0(f"Val file tokens: {_val_info_loader.source_tokens:,}; active val tokens before packing: {_val_info_loader.loaded_tokens:,}")
print0(
    f"Packed val tokens: {VAL_PACKED_TOKENS:,} "
    f"({VAL_PACKED_SEQUENCES:,} sequences; cutoff devices={_val_info_loader.cutoff_devices})"
)
del _val_info_loader

# Training config
tokens_per_fwdbwd = args.device_batch_size * MAX_SEQ_LEN * ddp_world_size
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0, \
    f"total batch size {TOTAL_BATCH_SIZE} is not divisible by {tokens_per_fwdbwd}"
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd
low_k_tokens_per_fwdbwd = args.low_k_device_batch_size * MAX_SEQ_LEN * ddp_world_size
if TOTAL_BATCH_SIZE % low_k_tokens_per_fwdbwd:
    raise ValueError("global batch must be divisible by the low-K microbatch across all ranks")
loader_steps_per_optimizer = TOTAL_BATCH_SIZE // low_k_tokens_per_fwdbwd
estimated_num_iterations = round(TOKENS_PER_EPOCH * args.num_epochs / TOTAL_BATCH_SIZE)
num_iterations = args.lr_schedule_steps if args.lr_schedule_steps is not None else estimated_num_iterations
max_train_steps = args.max_train_steps if args.max_train_steps is not None else estimated_num_iterations
stop_steps = resolve_stop_steps(max_train_steps, args.stop_after_steps)
if max_train_steps > estimated_num_iterations:
    raise ValueError("packed training data cannot cover --max-train-steps")
if CORE_REPETITION_SCHEDULE == "late":
    crossover = int(CORE_REPETITION_CROSSOVER_FRACTION * num_iterations)
    if DEPTH_SCALE_MODE == "none" or not 1 <= CORE_REPETITION_LOW < CORE_REPETITIONS or not 1 <= crossover < max_train_steps:
        raise ValueError("growth requires loop/dep mode, 1 <= low K < final K, and a crossover inside training")
eval_token_budget = min(EVAL_TOKENS, VAL_PACKED_TOKENS)  # cap the eval pass at 10M tokens
print0(f"Batch size: {TOTAL_BATCH_SIZE:,} tokens, grad accum: low-K={loader_steps_per_optimizer}, final-K={grad_accum_steps}")
print0(
    f"Training for {args.num_epochs} epoch(s) "
    f"(estimated_steps={estimated_num_iterations}, schedule_steps={num_iterations}, max_train_steps={max_train_steps})"
)
print0(f"Eval set: {eval_token_budget:,} packed tokens across {max(1, eval_token_budget // tokens_per_fwdbwd):,} step(s)")

# Schedulers
def get_lr_multiplier(it):
    return get_lr_multiplier_for(it, num_iterations, WARMDOWN_RATIO, WARMUP_STEPS, FINAL_LR_FRAC)

# Re-warm scheduled learning rates for 40 steps after the growth crossover.
LSCHED_LR_WARMUP_STEPS = args.lsched_lr_warmup_steps
crossover_step = int(CORE_REPETITION_CROSSOVER_FRACTION * num_iterations)
LSCHED_CHANGE_STEPS = (crossover_step,) if CORE_REPETITION_SCHEDULE == "late" and 0 < crossover_step < num_iterations and CORE_REPETITION_LOW != CORE_REPETITIONS else ()


def get_lsched_lr_mult(it):
    if not LSCHED_CHANGE_STEPS or LSCHED_LR_WARMUP_STEPS <= 0 or it < crossover_step:
        return 1.0
    return min(1.0, (it - crossover_step) / LSCHED_LR_WARMUP_STEPS)


# --- dep grow-init: stacking copy + dormant-core zero-grad injection ------------
# An UNTIED dep model is allocated at peak K (=CORE_REPETITIONS) but runs only its first
# CORE_REPETITION_LOW cores until the late crossover; there the newly-activated cores are
# stacking-copy-initialized from the trained ones (C_j <- C_{j mod low}) and the model runs
# the full K. dep_grow_events records each grow (result.json) so tests can assert the
# crossover actually fired.
dep_grow_events = []
DEP_GROW = DEPTH_SCALE_MODE == "dep" and CORE_REPETITION_SCHEDULE != "none"
if DEP_GROW:
    # Crossover-sanity: a mis-configured smoke where the copy never fires (crossover < 1)
    # or the run ends before growing (crossover >= max_train_steps) is not a physics
    # result -- fail loud at startup rather than silently produce a K2-only run.
    _dep_crossover_step = int(CORE_REPETITION_CROSSOVER_FRACTION * num_iterations)
    if _dep_crossover_step < 1 or _dep_crossover_step >= max_train_steps:
        raise ValueError(
            f"dep grow crossover step {_dep_crossover_step} (= int(scf="
            f"{CORE_REPETITION_CROSSOVER_FRACTION} * num_iterations={num_iterations})) is out of "
            f"range [1, max_train_steps={max_train_steps}); the copy-init would never fire / the "
            "run would end before growing")
    print0(f"dep grow-init: stacking copy scheduled at step {_dep_crossover_step} "
           f"(active {CORE_REPETITION_LOW} -> {CORE_REPETITIONS})")

# Dormant-core params (cores CORE_REPETITION_LOW..K) carry an injected ZERO grad every
# pre-crossover step so each per-shape DistMuonAdamW Muon group's params-with-grad list is
# the FULL fixed list from step 1. This keeps chunk_size / params[0] / row-ordering stable
# across the crossover -- a list that GROWS there crashes the distributed reduce (stale
# chunk_size sliced against more owned rows) or silently wipes the whole group's momentum
# (local shape re-check). C0/C1 momentum is preserved (group membership never changes); the
# dormant rows sit at exactly 0 == the intended reset for the freshly-copied cores.
DEP_DORMANT_PARAMS = (
    [p for r in orig_model.core_recurrence_indices[CORE_REPETITION_LOW:]
     for i in r for p in orig_model.transformer["h"][i].parameters()]
    if DEP_GROW else []
)

def get_muon_momentum(it):
    """Muon momentum ramps 0.85 -> 0.95 over the first 300 steps."""
    return (1 - min(it / 300, 1)) * 0.85 + min(it / 300, 1) * 0.95


steps_per_epoch = num_iterations / args.num_epochs


# Training loop
step = 0
min_val_bpb = float("inf")
min_val_loss = float("inf")
best_val_epoch = None
val_bpb = float("inf")
val_loss = float("inf")
# Per-epoch (train_loss, val_loss) series: one entry per epoch boundary plus the
# final eval. train_loss is the debiased smoothed train loss at that step (the same
# quantity as final_train_loss), so the two curves are directly comparable -- this is
# the train-val gap the data-constrained sections read.
train_val_per_epoch = []


def _debiased_train_loss():
    """Debiased EMA-smoothed train loss at the current step (matches final_train_loss)."""
    return smooth_train_loss / (1 - 0.9 ** step) if step > 0 else float("inf")


last_val_eval_step = None
last_completed_epoch = 0
smooth_train_loss = 0
total_training_time = 0
timed_steps = 0
timing_start_step = 4  # skip first compile + 3 warmup steps
eval_steps = max(1, eval_token_budget // (args.device_batch_size * MAX_SEQ_LEN * ddp_world_size))
last_core_block_residual_rms = None
last_train_mfu = None
last_train_tok_per_sec = None
last_train_step_time = None
core_repetition_sample_history = []
core_repetition_sample_histogram = {}


def sample_active_core_repetitions():
    if CORE_REPETITION_SCHEDULE == "none":
        return None
    value = CORE_REPETITION_LOW if step < crossover_step else CORE_REPETITIONS
    core_repetition_sample_history.append(value)
    core_repetition_sample_histogram[str(value)] = core_repetition_sample_histogram.get(str(value), 0) + 1
    return value


def eval_active_core_repetitions():
    if DEPTH_SCALE_MODE not in ("loop", "dep"):
        return None
    return CORE_REPETITIONS


model.eval()
val_loader = build_val_loader()
with autocast_ctx:
    val_bpb, val_loss = evaluate_bpb(
        model,
        val_loader,
        eval_steps,
        token_bytes,
        active_core_repetitions=eval_active_core_repetitions(),
    )
last_val_eval_step = step
print0(f"Step {step:05d} | Val BPB: {val_bpb:.6f} | Val Loss: {val_loss:.6f}")
wandb_run.log({
    "step": step,
    "val/bpb": val_bpb,
    "val/loss": val_loss,
    "val_bpb/current": val_bpb,
    "val_loss/current": val_loss,
})
model.train()

last_scheduled_core_repetitions = None  # previous step's scheduled active K, for switch logging

while current_epoch <= args.num_epochs and step < stop_steps:
    # Training step
    synchronize()
    t0 = time.time()
    active_core_repetitions = sample_active_core_repetitions()
    if (CORE_REPETITION_SCHEDULE != "none"
            and last_scheduled_core_repetitions is not None
            and active_core_repetitions != last_scheduled_core_repetitions):
        print0(f"core repetition schedule: active K "
               f"{last_scheduled_core_repetitions} -> {active_core_repetitions} at step {step}")
        # At a dep grow step, stacking-copy-initialize the newly-activated
        # untied cores from the trained ones (C_j <- C_{j mod low}). Runs BEFORE this step's
        # forward, and after the previous optimizer.step() all-gathered the params, so every
        # rank copies identical sources (pure local memcpy, no collective).
        if DEP_GROW and active_core_repetitions > last_scheduled_core_repetitions:
            dep_grow_copies = dep_stack_grow_init(
                orig_model, last_scheduled_core_repetitions, active_core_repetitions)
            print0(f"dep grow-init: stacking copy at step {step}: {dep_grow_copies}")
            dep_grow_events.append({
                "step": step,
                "from": int(last_scheduled_core_repetitions),
                "to": int(active_core_repetitions),
                "block_copies": dep_grow_copies,
            })
            wandb_run.summary["dep_grow_step"] = step
            if args.low_k_device_batch_size != args.device_batch_size:
                # Release cached K2 activation workspaces before compiling the K4 graph.
                synchronize()
                torch.cuda.empty_cache()
    last_scheduled_core_repetitions = active_core_repetitions
    active_device_batch_size = (
        args.low_k_device_batch_size
        if active_core_repetitions is not None and active_core_repetitions < CORE_REPETITIONS
        else args.device_batch_size
    )
    splits_per_loader_batch = args.low_k_device_batch_size // active_device_batch_size
    active_grad_accum_steps = loader_steps_per_optimizer * splits_per_loader_batch
    microbatch_losses = []
    for loader_micro_step in range(loader_steps_per_optimizer):
        micro_xs = x.split(active_device_batch_size, dim=0)
        micro_ys = y.split(active_device_batch_size, dim=0)
        if len(micro_xs) != splits_per_loader_batch or len(micro_ys) != splits_per_loader_batch:
            raise RuntimeError("training loader batch does not match the configured microbatch layout")
        for micro_x, micro_y in zip(micro_xs, micro_ys):
            if args.optimizer_transport == "owner_bucketed":
                inactive = DEP_DORMANT_PARAMS if DEP_DORMANT_PARAMS and active_core_repetitions < CORE_REPETITIONS else ()
                optimizer.prepare_backward(inactive_params=inactive)
            with autocast_ctx:
                loss, metrics = model(
                    micro_x,
                    micro_y,
                    active_core_repetitions=active_core_repetitions,
                )
            train_loss = loss.detach()
            microbatch_losses.append(train_loss)
            (loss / active_grad_accum_steps).backward()
            if args.optimizer_transport == "owner_bucketed":
                optimizer.finish_backward()
        x, y, epoch = next(train_loader)

    # Keep the dormant cores' Muon group rows alive with an injected ZERO grad
    # every pre-crossover step (see DEP_DORMANT_PARAMS) so each per-shape group's
    # params-with-grad list is the full fixed list from step 1 -- stable across the grow.
    # model.zero_grad(set_to_none=True) frees these each step, so they are re-created here by
    # design. The AdamW embed/head groups are per-param state and untouched (always live).
    if not getattr(optimizer, "uses_zero2", False) and DEP_DORMANT_PARAMS and active_core_repetitions < CORE_REPETITIONS:
        for p in DEP_DORMANT_PARAMS:
            if p.grad is None:
                p.grad = torch.zeros_like(p)

    # Update optimizer
    lrm = get_lr_multiplier(step)
    # Loop-schedule LR re-warmup: fold the (<=1) re-warmup multiplier into lrm so it
    # scales ONLY the scheduled-LR groups (same groups the WSD lrm scales below). 1.0
    # unless we are inside a re-warmup window after a loop-K change (0.0 at the change).
    lsched_lr_mult = get_lsched_lr_mult(step)
    lrm *= lsched_lr_mult
    if lsched_lr_mult < 1.0:
        print0(f"lsched LR re-warmup: step {step:05d} lr_mult={lsched_lr_mult:.4f}")
    for group in optimizer.param_groups:
        if group.get("schedule_lr", True):
            group["lr"] = group["initial_lr"] * lrm
        else:
            group["lr"] = group["initial_lr"]
        if "initial_wd" not in group:
            group["initial_wd"] = group.get("weight_decay", 0.0)
        group["weight_decay"] = group["initial_wd"]
        if group['kind'] == 'muon':
            group["momentum"] = get_muon_momentum(step)
    optimizer.step()
    model.zero_grad(set_to_none=True)
    train_loss_f = train_loss.item()
    # Keep the paper's rank-0 last-microbatch statistic, and report the global
    # mean separately. The former changes its sampled examples with GPU count.
    global_batch_loss = torch.stack(microbatch_losses).float().mean()
    if dist.is_initialized():
        dist.all_reduce(global_batch_loss, op=dist.ReduceOp.AVG)
    global_batch_loss_f = global_batch_loss.item()
    nonfinite_train_loss = not (math.isfinite(float(train_loss_f)) and math.isfinite(global_batch_loss_f))
    if dist.is_initialized():
        nonfinite_tensor = torch.tensor([1 if nonfinite_train_loss else 0], dtype=torch.int32, device=device)
        dist.all_reduce(nonfinite_tensor, op=dist.ReduceOp.MAX)
        nonfinite_train_loss = bool(nonfinite_tensor.item())
    if nonfinite_train_loss:
        raise RuntimeError(f"non-finite training loss at optimizer step {step + 1}")
    synchronize()
    dt = time.time() - t0

    step += 1

    # Legacy paper metric: EMA of rank 0's last microbatch, not global-batch loss.
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased = smooth_train_loss / (1 - ema_beta**step)
    pct = 100 * step / stop_steps
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
    active_flops_per_token = flops_by_core_repetitions[active_core_repetitions or 1]
    estimated_training_flops += active_flops_per_token * TOTAL_BATCH_SIZE
    mfu = 100 * active_flops_per_token * TOTAL_BATCH_SIZE / dt / (gpu_peak_flops * ddp_world_size)
    last_train_mfu = mfu
    last_train_tok_per_sec = tok_per_sec
    last_train_step_time = dt
    if step >= timing_start_step:
        total_training_time += dt
        timed_steps += 1
    eta_str = f" | eta: {(stop_steps - step) * total_training_time / timed_steps / 60:.1f}m" if timed_steps > 0 else ""
    k_str = (
        f" | core_reps: {active_core_repetitions}"
        if CORE_REPETITION_SCHEDULE != "none"
        else ""
    )
    print0(f"step {step:05d} ({pct:.2f}%) | loss: {debiased:.6f}{k_str} | dt: {dt*1000:.2f}ms | tok/sec: {tok_per_sec:,} | bf16_mfu: {mfu:.2f}%{eta_str}")
    if master_process:
        with open(os.path.join(run_dir, "trajectory.jsonl"), "a") as handle:
            handle.write(json.dumps({"step": step, "legacy_loss": train_loss_f,
                                     "legacy_loss_ema": debiased,
                                     "global_batch_loss": global_batch_loss_f}) + "\n")
    if args.log_microbatch_losses:
        with open(os.path.join(run_dir, f"microbatches_rank{ddp_rank}.jsonl"), "a") as handle:
            handle.write(json.dumps({"step": step, "rank": ddp_rank, "world_size": ddp_world_size,
                                     "loader_batch_size": args.low_k_device_batch_size,
                                     "microbatch_size": active_device_batch_size,
                                     "losses": torch.stack(microbatch_losses).float().tolist()}) + "\n")
    metric_values = {k: v.detach().item() for k, v in metrics.items()}
    log_payload = {
        "step": step,
        "train/loss": debiased,
        "train/global_batch_loss": global_batch_loss_f,
        "train/mfu": mfu,
        "train/lrm": lrm,
    }
    for key, value in metric_values.items():
        if key == "core_block_residual_rms":
            last_core_block_residual_rms = value
            log_payload["rms/core_block_residual"] = value
        elif key.startswith("rms/"):
            # already fully-qualified metric names (rms/res_enc, rms/res_core, ...)
            log_payload[key] = value
        else:
            log_payload[f"train/{key}"] = value
    if CORE_REPETITION_SCHEDULE != "none":
        log_payload["train/core_repetitions"] = active_core_repetitions
    wandb_run.log(log_payload)

    # Synchronize epoch across ranks (different ranks may exhaust data at different steps)
    if ddp:
        epoch_tensor = torch.tensor([epoch], dtype=torch.long, device=device)
        dist.all_reduce(epoch_tensor, op=dist.ReduceOp.MAX)
        epoch = epoch_tensor.item()

    # Epoch boundary: evaluate when the dataloader advances to a new epoch
    if epoch != current_epoch:
        model.eval()
        val_loader = build_val_loader()
        with autocast_ctx:
            val_bpb, val_loss = evaluate_bpb(
                model,
                val_loader,
                eval_steps,
                token_bytes,
                active_core_repetitions=eval_active_core_repetitions(),
            )
        last_val_eval_step = step
        if not math.isfinite(float(val_loss)) or not math.isfinite(float(val_bpb)):
            raise RuntimeError(
                f"non-finite validation metric at epoch {current_epoch}: "
                f"val_bpb={val_bpb} val_loss={val_loss}"
            )
        _epoch_train_loss = _debiased_train_loss()
        print0(f"Step {step:05d} | Epoch {current_epoch} | Train Loss: {_epoch_train_loss:.6f} | "
               f"Val BPB: {val_bpb:.6f} | Val Loss: {val_loss:.6f}")
        train_val_per_epoch.append({
            "epoch": int(current_epoch), "step": int(step),
            "train_loss": float(_epoch_train_loss),
            "val_loss": float(val_loss), "val_bpb": float(val_bpb)})
        wandb_run.log({
            "step": step,
            "epoch": current_epoch,
            "train/loss_epoch": _epoch_train_loss,
            "val/bpb": val_bpb,
            "val/loss": val_loss,
            "val_bpb/current": val_bpb,
            "val_loss/current": val_loss,
        })
        last_completed_epoch = int(current_epoch)
        if val_loss < min_val_loss:
            min_val_bpb = val_bpb
            min_val_loss = val_loss
            best_val_epoch = int(current_epoch)
        model.train()
        current_epoch = epoch

    # One collection after the first step, then freeze the heap: the steady state
    # allocates nothing that needs collecting, and GC pauses show up in step time.
    if step == 1:
        gc.collect(); gc.freeze(); gc.disable()

if last_val_eval_step != step:
    model.eval()
    val_loader = build_val_loader()
    with autocast_ctx:
        val_bpb, val_loss = evaluate_bpb(
            model,
            val_loader,
            eval_steps,
            token_bytes,
            active_core_repetitions=eval_active_core_repetitions(),
        )
    last_val_eval_step = step
    if not math.isfinite(float(val_loss)) or not math.isfinite(float(val_bpb)):
        raise RuntimeError(
            f"non-finite final validation metric at step {step}: "
            f"val_bpb={val_bpb} val_loss={val_loss}"
        )
    epoch_progress = step / steps_per_epoch if steps_per_epoch > 0 else None
    _epoch_train_loss = _debiased_train_loss()
    print0(
        f"Step {step:05d} | Final eval | "
        f"Epoch progress: {epoch_progress:.6f} | Train Loss: {_epoch_train_loss:.6f} | "
        f"Val BPB: {val_bpb:.6f} | Val Loss: {val_loss:.6f}"
    )
    # Final (mid-epoch) endpoint for the per-epoch train-val gap series. Only reached
    # when the run ends off a boundary (last_val_eval_step != step), so no duplicate.
    train_val_per_epoch.append({
        "epoch": int(current_epoch), "step": int(step),
        "train_loss": float(_epoch_train_loss),
        "val_loss": float(val_loss), "val_bpb": float(val_bpb)})
    wandb_run.log({
        "step": step,
        "epoch_progress": epoch_progress,
        "val/bpb": val_bpb,
        "val/loss": val_loss,
        "val_bpb/current": val_bpb,
        "val_loss/current": val_loss,
        "val/final_step_bpb": val_bpb,
        "val/final_step_loss": val_loss,
    })
    if val_loss < min_val_loss:
        min_val_bpb = val_bpb
        min_val_loss = val_loss
        best_val_epoch = epoch_progress
    model.train()

# Summary
total_wall_time = time.time() - _script_start
print0(f"Peak memory: {get_max_memory() / 1024 / 1024:.2f} MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
final_train_loss = smooth_train_loss / (1 - 0.9**step) if step > 0 else float('inf')
print0(f"Final train loss: {final_train_loss:.6f}")
print0(f"Min val BPB: {min_val_bpb:.6f}")
print0(f"Min val Loss: {min_val_loss:.6f}")
if last_core_block_residual_rms is not None:
    print0(f"Core block residual RMS: {last_core_block_residual_rms:.6f}")
if core_repetition_sample_history:
    observed_core_repetitions_mean = sum(core_repetition_sample_history) / len(core_repetition_sample_history)
    observed_core_repetitions_min = min(core_repetition_sample_history)
    observed_core_repetitions_max = max(core_repetition_sample_history)
    print0(
        f"Observed train core repetitions: mean={observed_core_repetitions_mean:.4f}, "
        f"min={observed_core_repetitions_min}, max={observed_core_repetitions_max}"
    )
else:
    observed_core_repetitions_mean = None
    observed_core_repetitions_min = None
    observed_core_repetitions_max = None
wandb_run.summary["final_train_loss"] = final_train_loss
# Record the headline validation loss used by the plotting scripts.
wandb_run.summary["final_val_loss"] = val_loss
wandb_run.summary["best_val_loss"] = min_val_loss
wandb_run.summary["best_val_bpb"] = min_val_bpb
if best_val_epoch is not None:
    wandb_run.summary["best_val_epoch"] = best_val_epoch
wandb_run.summary["total_wall_time_seconds"] = total_wall_time
wandb_run.summary["total_training_time_seconds"] = total_training_time
if last_train_mfu is not None:
    wandb_run.summary["last_train_mfu"] = last_train_mfu
if last_train_tok_per_sec is not None:
    wandb_run.summary["last_train_tok_per_sec"] = last_train_tok_per_sec
if last_core_block_residual_rms is not None:
    wandb_run.summary["rms/core_block_residual"] = last_core_block_residual_rms
if core_repetition_sample_history:
    wandb_run.summary["train_active_core_repetitions_mean"] = observed_core_repetitions_mean
    wandb_run.summary["train_active_core_repetitions_min"] = observed_core_repetitions_min
    wandb_run.summary["train_active_core_repetitions_max"] = observed_core_repetitions_max

final_checkpoint_path = None
if args.save_final_checkpoint:
    final_checkpoint_path = os.path.abspath(os.path.join(checkpoints_dir, "final.pt"))
    if master_process:
        # Parameters only (buffers are config-derived and rebuilt by init_weights),
        # plus the configuration needed to rebuild the model.
        final_ckpt = {
            "model": {name: p.detach().float().cpu() for name, p in orig_model.named_parameters()},
            "args": vars(args),
            "step": step,
            "final_train_loss": final_train_loss,
            "best_val_loss": min_val_loss,
            "final_eval_active_core_repetitions": eval_active_core_repetitions(),
            "head_dim": HEAD_DIM,
        }
        torch.save(final_ckpt, final_checkpoint_path)
        del final_ckpt
    print0(f"Final checkpoint saved to {final_checkpoint_path}")
    wandb_run.summary["final_checkpoint_path"] = final_checkpoint_path

_result_out = args.save_result or result_path
if master_process:
    result = {
        # --- model ---
        "parameter_count": param_summary["total"],
        "parameter_summary": param_summary,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "n_embd": args.n_embd,
        "head_dim": HEAD_DIM,
        "depth_scale_mode": args.depth_scale_mode,
        "core_repetitions": args.core_repetitions,
        "core_layer_count": args.core_layer_count,
        "resolved_core_layer_count": getattr(orig_model, "core_layer_count", None),
        "prelude_layer_count": args.prelude_layer_count,
        "resolved_prelude_layer_count": getattr(orig_model, "prelude_size", None),
        "coda_layer_count": args.coda_layer_count,
        "resolved_coda_layer_count": getattr(orig_model, "coda_size", None),
        "effective_depth": getattr(orig_model, "effective_depth", EFFECTIVE_DEPTH),
        "expected_effective_depth": getattr(orig_model, "expected_effective_depth", float(EFFECTIVE_DEPTH)),
        "recurrence_rmsnorm": args.recurrence_rmsnorm,
        "recurrence_emb_alpha": args.recurrence_emb_alpha,
        "decoder_inject": args.decoder_inject,
        "core_residual_branch_scale": CORE_RESIDUAL_BRANCH_SCALE,
        "resolved_core_residual_branch_scale": getattr(orig_model, "core_residual_scale", None),
        # --- K schedule ---
        "core_repetition_schedule": args.core_repetition_schedule,
        "core_repetition_crossover_fraction": (
            args.core_repetition_crossover_fraction
            if args.core_repetition_schedule == "late" else None
        ),
        "core_repetition_low": args.core_repetition_low,
        "lsched_lr_warmup_steps": args.lsched_lr_warmup_steps,
        "lsched_lr_change_steps": list(LSCHED_CHANGE_STEPS),
        "dep_grow_events": dep_grow_events,
        "target_core_repetitions": CORE_REPETITIONS,
        "final_eval_active_core_repetitions": eval_active_core_repetitions(),
        "observed_train_core_repetitions_mean": observed_core_repetitions_mean,
        "observed_train_core_repetitions_min": observed_core_repetitions_min,
        "observed_train_core_repetitions_max": observed_core_repetitions_max,
        "observed_train_core_repetitions_histogram": (
            core_repetition_sample_histogram if core_repetition_sample_history else None
        ),
        # --- optimization ---
        "matrix_lr": args.matrix_lr,
        "embedding_lr": args.embedding_lr,
        "unembedding_lr": args.unembedding_lr,
        "lr_multiplier": args.lr_multiplier,
        "effective_matrix_lr": MATRIX_LR,
        "effective_embedding_lr": EMBEDDING_LR,
        "effective_unembedding_lr": UNEMBEDDING_LR,
        "weight_decay": args.weight_decay,
        "matrix_weight_decay": MATRIX_WEIGHT_DECAY,
        "embedding_weight_decay": EMBEDDING_WEIGHT_DECAY,
        "unembedding_weight_decay": UNEMBEDDING_WEIGHT_DECAY,
        "embedding_wd_skip_norm": EMBEDDING_WD_SKIP_NORM,
        "adam_betas": list(args.adam_betas),
        "adam_eps": args.adam_eps,
        "lr_schedule": "wsd",
        "warmup_steps": args.warmup_steps,
        "warmdown_ratio": WARMDOWN_RATIO,
        "output_multiplier": args.output_multiplier,
        "residual_branch_multiplier": args.residual_branch_multiplier,
        "wte_init_std": args.wte_init_std,
        "uniform_init_scale": args.uniform_init_scale,
        # --- budget / data ---
        "total_batch_size": args.total_batch_size,
        "device_batch_size": args.device_batch_size,
        "low_k_device_batch_size": args.low_k_device_batch_size,
        "gradient_accumulation_steps": grad_accum_steps,
        "low_k_gradient_accumulation_steps": loader_steps_per_optimizer,
        "optimizer_transport": args.optimizer_transport,
        "zero2": optimizer_distribution,
        "environment": {"python": sys.version, "torch": torch.__version__,
                        "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                        "attention_backend": "flash_attention_3" if is_flash_attention_3_available() else "pytorch_sdpa",
                        "flash_attention_revision": FLASH_ATTENTION_REVISION if is_flash_attention_3_available() else None,
                        "model_compiled": not args.no_compile,
                        "optimizer_compile_step_enabled": all(group.get("compile_step", True)
                                                              for group in optimizer.param_groups)},
        "world_size": ddp_world_size,
        "train_loss_statistic": "rank0_last_microbatch_ema",
        "final_global_batch_loss": global_batch_loss_f,
        "num_epochs": args.num_epochs,
        "estimated_num_iterations": estimated_num_iterations,
        "lr_schedule_steps": args.lr_schedule_steps,
        "resolved_lr_schedule_steps": num_iterations,
        "max_train_steps": args.max_train_steps,
        "resolved_max_train_steps": max_train_steps,
        "stop_after_steps": args.stop_after_steps,
        "resolved_stop_steps": stop_steps,
        "stopped_early": step < max_train_steps,
        "completed_train_steps": step,
        "completed_epochs": last_completed_epoch,
        "seed": args.seed,
        "data_dir": os.path.abspath(DATA_DIR),
        "data_preparation_provenance": data_provenance,
        "train_token_limit": args.train_token_limit,
        "val_token_limit": args.val_token_limit,
        "token_limit_cutoff_devices": args.token_limit_cutoff_devices,
        "active_train_tokens": train_loader.loaded_tokens,
        "packed_train_sequences_per_epoch": train_loader.packed_sequences,
        "packed_train_tokens_per_epoch": train_loader.total_tokens,
        "active_val_packed_sequences": VAL_PACKED_SEQUENCES,
        "active_val_packed_tokens": VAL_PACKED_TOKENS,
        # --- results ---
        "val_loss": val_loss,
        "final_val_loss": val_loss,
        "final_train_loss": final_train_loss,
        "best_val_loss": min_val_loss,
        "best_val_bpb": min_val_bpb,
        "best_val_epoch": best_val_epoch,
        "train_val_per_epoch": train_val_per_epoch,
        "core_block_residual_rms": last_core_block_residual_rms,
        "last_train_mfu": last_train_mfu,
        "estimated_training_flops": estimated_training_flops,
        "flops_per_token_by_active_core_repetitions": flops_by_core_repetitions,
        "last_train_tok_per_sec": last_train_tok_per_sec,
        "last_train_step_time_seconds": last_train_step_time,
        "total_training_time_seconds": total_training_time,
        "total_wall_time_seconds": total_wall_time,
        "final_checkpoint_path": final_checkpoint_path,
        "wandb_url": getattr(wandb_run, "url", None),
    }
    with open(_result_out, "w") as f:
        json.dump(result, f, indent=2)
    print0(f"Result saved to {_result_out}")

print0(f"Total wall time: {total_wall_time:.2f}s ({total_wall_time/60:.2f}m)")

wandb_run.finish()
if dist.is_initialized():
    dist.destroy_process_group()
if artifacts_log_f is not None:
    sys.stdout.flush()
    sys.stderr.flush()
    sys.stdout = stdout_orig
    sys.stderr = stderr_orig
    artifacts_log_f.close()
