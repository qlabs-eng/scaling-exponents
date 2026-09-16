"""Model + optimizer configuration.

One dataclass, mirroring train.py's flag surface. Field defaults here are only a
fallback: train.py always passes an explicit value for everything it exposes.
"""
from dataclasses import dataclass


@dataclass
class GPTConfig:
    # --- shape ---
    sequence_len: int = 2048
    vocab_size: int = 50257
    n_layer: int = 12              # PHYSICAL layers: prelude + core + coda
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    head_dim: int | None = None
    window_pattern: str = "L"      # per-layer attention window: S=half context, L=full

    # --- architecture ---
    # "none" = Vanilla (a plain stack), "loop" = the tied loop (one core applied
    # core_repetitions times), "dep" = the untied twin (core_repetitions distinct
    # cores, run in sequence). loop and dep execute the SAME number of blocks per
    # token, so they are FLOPs-matched; dep merely stores more parameters.
    depth_scale_mode: str = "none"
    # Fractional K is allowed in loop mode: K=1.5 on a 2-layer core runs a,b,a
    # (K * core_layer_count must be a whole number). The final pass is then a
    # prefix of the core. K schedules and truncated backprop require an integer K.
    core_repetitions: float = 1
    core_layer_count: int | None = None
    prelude_layer_count: int | None = None
    coda_layer_count: int | None = None

    # --- loop wiring ---
    # No-affine RMSNorm at the recurrence boundary. "recurrence_only" skips the
    # post-prelude norm, so the anchor is the raw prelude output.
    recurrence_rmsnorm: str = "none"
    # Anchor re-injection between core passes: x = norm(x) + alpha * anchor.
    recurrence_emb_alpha: float = 0.0
    # One more anchor injection at the core -> coda boundary.
    decoder_inject: bool = False
    # Truncated backprop through the recurrence: "constant" detaches all but the
    # trailing recurrence_bp_window passes.
    recurrence_truncation_schedule: str = "none"
    recurrence_bp_window: int = 0

    # --- init / output scaling ---
    output_multiplier: float = 1.0        # logits are scaled by this before the softmax
    residual_branch_multiplier: float = 1.0
    wte_init_std: float = 0.8             # std of the normal init on the token embedding
    uniform_init_scale: float = 3 ** 0.5  # c in the uniform init bound s = c * n_embd**-0.5

    # --- optimizer ---
    matrix_lr: float = 0.016              # Muon, dense matrices
    embedding_lr: float = 0.032           # AdamW, token embedding
    unembedding_lr: float = 0.001         # AdamW, lm_head
    weight_decay: float = 0.0
    matrix_weight_decay: float = 0.0
    embedding_weight_decay: float = 0.0
    unembedding_weight_decay: float = 0.0
    # Vocabulary rows whose RMS norm is below this are exempt from embedding WD,
    # so rare tokens are not decayed to zero by updates they never receive.
    embedding_wd_skip_norm: float = 0.03
    adam_betas: tuple[float, float] = (0.8, 0.95)
    adam_eps: float = 1e-10
