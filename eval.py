"""Evaluate a trained checkpoint on the paper's 22-task CORE accuracy and answer NLL.

    python eval.py --checkpoint runs/vanilla_d08/checkpoints/final.pt

Defaults: all 91,037 examples, few-shot seeds 0/1/2, GPT-2 tokenization,
and the fingerprinted paper bundle. See README.md for metric definitions.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import os
import time
from pathlib import Path

import tiktoken
import torch

from core_eval import TiktokenCoreTokenizer, bundle_fingerprint, evaluate_core
from models import GPTConfig, TransformerGPT, is_flash_attention_3_available

# Frozen Vanilla-only filter from the paper; never refit on the evaluated model.
FILTERED_OUT_TASKS = frozenset({
    "bigbench_cs_algorithms", "bigbench_language_identification",
    "bigbench_repeat_copy_logic", "boolq", "commonsense_qa",
})


def build_config(result: dict) -> GPTConfig:
    """Reconstruct the training configuration from result.json."""
    required = {"n_layer", "n_head", "n_embd", "depth_scale_mode", "core_repetitions",
                "output_multiplier", "residual_branch_multiplier", "recurrence_rmsnorm",
                "recurrence_emb_alpha", "decoder_inject"}
    missing = required - result.keys()
    if missing:
        raise ValueError(f"result.json is missing model configuration: {sorted(missing)}")
    fields = {f.name for f in dataclasses.fields(GPTConfig)}
    kwargs = {name: result[name] for name in fields if name in result}
    kwargs["vocab_size"] = 50257  # GPT-2; the model pads to 50304, as in training.
    kwargs["n_kv_head"] = result["n_head"]
    kwargs.setdefault("sequence_len", 2048)
    return GPTConfig(**kwargs)


def load_and_verify(checkpoint_path: Path, result_path: Path, device):
    """Load every learned parameter; only configuration-derived buffers may be absent."""
    result = json.loads(result_path.read_text())
    config = build_config(result)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint["model"] if "model" in checkpoint else checkpoint
    # Wrapped checkpoints also carry the original arguments. Detect accidentally
    # pairing a checkpoint with a same-shaped run that used different wiring/scales.
    saved_args = checkpoint.get("args", {})
    forward_fields = ("n_layer", "n_head", "n_embd", "depth_scale_mode", "core_repetitions",
                      "core_layer_count", "prelude_layer_count", "coda_layer_count",
                      "recurrence_rmsnorm", "recurrence_emb_alpha", "decoder_inject",
                      "output_multiplier", "residual_branch_multiplier", "sequence_len",
                      "window_pattern")
    for name in forward_fields:
        if name in saved_args and name in result and saved_args[name] != result[name]:
            raise ValueError(f"checkpoint/result.json disagree on {name}")

    # Avoid constructing a second full CPU model for large checkpoints.
    with torch.device("meta"):
        model = TransformerGPT(config)
    model.to_empty(device=device)
    model.init_weights()  # Rebuild rotary and residual-scale buffers before loading.
    incompatible = model.load_state_dict(state, strict=False)
    param_names = {name for name, _ in model.named_parameters()}
    buffer_names = {name for name, _ in model.named_buffers()}
    missing_params = param_names - state.keys()
    if missing_params:
        raise RuntimeError(f"checkpoint is missing learned parameters: {sorted(missing_params)[:8]}")
    if incompatible.unexpected_keys:
        raise RuntimeError(f"checkpoint has unexpected keys: {incompatible.unexpected_keys[:8]}")
    if not set(incompatible.missing_keys) <= buffer_names:
        raise RuntimeError(f"unexpected missing keys: {incompatible.missing_keys[:8]}")

    summary = model.parameter_summary()
    expected = result.get("parameter_summary")
    if expected is not None and (summary["total"] != expected["total"]
                                 or summary["parts"] != expected["parts"]):
        raise RuntimeError(f"parameter_summary mismatch: got {summary}, expected {expected}")
    if result.get("parameter_count", summary["total"]) != summary["total"]:
        raise RuntimeError("parameter_count does not match result.json")
    active_k = result.get("final_eval_active_core_repetitions")
    saved_k = checkpoint.get("final_eval_active_core_repetitions", active_k)
    if saved_k != active_k:
        raise ValueError("checkpoint/result.json disagree on final evaluation recurrence")
    return model.eval(), result


class EvaluationModel(torch.nn.Module):
    """Use the recorded final evaluation K, including checkpoints stopped before growth."""

    def __init__(self, model, active_k):
        super().__init__()
        self.model = model
        self.config = model.config
        self.active_k = active_k

    def forward(self, tokens):
        return self.model(tokens, active_core_repetitions=self.active_k)


def filtered_core(scores):
    centered = scores["centered_results"]
    if len(centered) != 22 or not FILTERED_OUT_TASKS <= centered.keys():
        raise ValueError("expected the paper's 22 CORE tasks for the frozen 17-task filter")
    retained = sorted(centered.keys() - FILTERED_OUT_TASKS)
    return sum(centered[task] for task in retained) / len(retained), retained


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="final.pt written by train.py")
    parser.add_argument("--result-json", type=Path,
                        help="default: <checkpoint run directory>/result.json")
    parser.add_argument("--bundle-dir", type=Path, default=Path("eval_bundle"),
                        help="paper CORE bundle; downloaded if absent")
    parser.add_argument("--max-per-task", type=int, default=-1,
                        help="-1 = all examples (paper default); positive values are smoke tests")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                        help="few-shot demonstration seeds (default: 0 1 2)")
    parser.add_argument("--out", type=Path,
                        help="default: <checkpoint run directory>/eval.json")
    args = parser.parse_args(argv)
    if args.max_per_task != -1 and args.max_per_task <= 0:
        parser.error("--max-per-task must be -1 or a positive integer")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must be distinct")
    return args


def main() -> None:
    args = parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise SystemExit("Run eval.py with python, one checkpoint per GPU; do not use torchrun.")
    result_path = args.result_json or args.checkpoint.parent.parent / "result.json"
    output_path = args.out or args.checkpoint.parent.parent / "eval.json"
    if not result_path.is_file():
        raise SystemExit(f"no result.json at {result_path}; pass --result-json explicitly")
    if output_path.resolve() in {args.checkpoint.resolve(), result_path.resolve()}:
        raise SystemExit("--out must not overwrite the checkpoint or result.json")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    fa3 = is_flash_attention_3_available()
    if not fa3:
        if os.environ.get("LOOP_ALLOW_SDPA") != "1":
            raise SystemExit("FlashAttention-3 on H100 is required for paper evaluation. "
                             "Set LOOP_ALLOW_SDPA=1 for a CPU/SDPA smoke test.")
        print("WARNING: using CPU/SDPA; numerics differ from the paper.", flush=True)
    start = time.time()
    model, result = load_and_verify(args.checkpoint, result_path, device)
    active_k = result.get("final_eval_active_core_repetitions")
    eval_model = EvaluationModel(model, active_k).eval()
    tokenizer = TiktokenCoreTokenizer(tiktoken.get_encoding("gpt2"))
    autocast_ctx = (torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if device.startswith("cuda") else contextlib.nullcontext())
    with torch.no_grad():
        scores = evaluate_core(
            eval_model, tokenizer, device, max_per_task=args.max_per_task,
            eval_bundle_dir=str(args.bundle_dir), autocast_ctx=autocast_ctx,
            seeds=args.seeds,
        )
    filtered, retained = filtered_core(scores)
    output = {
        **scores,
        "protocol_id": "downstream-full-v3" if args.max_per_task == -1
                       and args.seeds == [0, 1, 2] and fa3 else "custom",
        "suite_fingerprint": bundle_fingerprint(args.bundle_dir),
        "checkpoint": str(args.checkpoint.resolve()),
        "result_json": str(result_path.resolve()),
        "model_config": dataclasses.asdict(model.config),
        "parameter_count": model.parameter_summary()["total"],
        "final_eval_active_core_repetitions": active_k,
        "val_loss": result.get("val_loss"),  # Recorded pretraining validation, not recomputed.
        "max_per_task": args.max_per_task,
        "core_metric_filtered": filtered,
        "filtered_tasks": retained,
        "device": device,
        "attention_backend": "flash_attention_3" if fa3 else "sdpa",
        "elapsed_seconds": time.time() - start,
    }
    serialized = json.dumps(output, indent=2, allow_nan=False) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(serialized)
    print(json.dumps({key: output[key] for key in (
        "core_metric", "core_loss", "core_metric_filtered", "core_metric_std",
        "elapsed_seconds")}, indent=2))
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
