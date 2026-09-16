"""The four architectures of the paper, as one module.

All four are the SAME stack -- prelude blocks, a core, coda blocks -- and differ
only in how the core is executed and how its weights are shared:

    depth_scale_mode   core execution                       parameters
    ----------------   ----------------------------------   --------------------------
    "none"   Vanilla   n_layer distinct blocks, once         one block per layer
    "loop"   K1 / K2   ONE core of `core_layer_count`         one core, K applications
                       blocks, applied `core_repetitions`
                       times (weights tied across passes)
    "dep"    Dep       K DISTINCT cores in sequence           K cores stored

loop and dep run the identical number of block applications per token, so they are
FLOPs-matched at a rung; dep merely stores K copies of the core. That is the whole
point of the comparison -- any difference between them is capacity, not compute.

Between core passes the stream is optionally re-normalized (recurrence_rmsnorm) and
the prelude output ("the anchor") is re-injected (recurrence_emb_alpha):

    x <- norm(x) + alpha * anchor

with one extra injection at the core -> coda boundary when decoder_inject is set.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import print0
from .layers import Block, init_block_weights, norm
from .optimizer import build_optimizer


class TransformerGPT(nn.Module):
    """Decoder-only transformer with a repeatable core."""

    # Prelude/coda sizes used by a Vanilla stack purely to LABEL its layers for the
    # per-segment residual metrics; they never change what a Vanilla model computes.
    PRELUDE_LAYERS = 4
    CODA_LAYERS = 4

    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        self.config = config
        self._configure_depth_layout(config)
        padded_vocab = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab}")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab, config.n_embd),
            "h": nn.ModuleList([Block(config) for _ in range(self.physical_depth)]),
        })
        self.lm_head = nn.Linear(config.n_embd, padded_vocab, bias=False)
        self.window_sizes = self._compute_window_sizes(config, self.physical_depth)
        head_dim = config.n_embd // config.n_head
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    # ------------------------------------------------------------------ layout

    def _configure_depth_layout(self, config):
        """Resolve (prelude, core, coda) block indices and the execution order.

        Sets, for every mode:
          physical_depth          how many Blocks are allocated
          effective_depth         how many block APPLICATIONS a token sees
          application_indices     the full execution order, as block indices
          core_recurrence_indices one list of block indices per core pass
          track_core_residual     whether the model has a real prelude/core/coda split
        """
        mode = config.depth_scale_mode
        if mode not in {"none", "loop", "dep"}:
            raise ValueError("depth_scale_mode must be one of: none, loop, dep")

        # K may be fractional in loop mode (K=1.5 on core [a,b] -> a,b,a). An integral
        # K stays an int so the K-indexed logic below is exact.
        repetitions = float(config.core_repetitions)
        fractional_k = not repetitions.is_integer()
        if not fractional_k:
            repetitions = int(repetitions)
        if repetitions < 1:
            raise ValueError("core_repetitions must be >= 1")
        if mode == "none" and repetitions != 1:
            raise ValueError("core_repetitions must be 1 when depth_scale_mode is none")
        if fractional_k and mode != "loop":
            raise ValueError("fractional core_repetitions requires loop depth scaling")

        core_layer_count = config.core_layer_count
        prelude_layer_count = config.prelude_layer_count
        coda_layer_count = config.coda_layer_count
        if mode == "none" and (prelude_layer_count is not None or coda_layer_count is not None):
            raise ValueError("prelude/coda layer counts require depth scaling")
        if mode != "none" and config.n_layer != 12 and core_layer_count is None:
            raise ValueError("core_layer_count is required when depth scaling a non-d12 transformer")
        if config.recurrence_rmsnorm not in {"none", "prelude_and_recurrence", "recurrence_only"}:
            raise ValueError("recurrence_rmsnorm must be one of: none, prelude_and_recurrence, recurrence_only")
        if config.recurrence_emb_alpha != 0.0 and mode not in ("loop", "dep"):
            raise ValueError("recurrence_emb_alpha requires loop or dep depth scaling")
        if config.decoder_inject:
            if mode not in ("loop", "dep"):
                raise ValueError("decoder_inject requires loop or dep depth scaling")
            if config.recurrence_emb_alpha == 0.0:
                raise ValueError("decoder_inject requires recurrence_emb_alpha != 0")
        if config.recurrence_truncation_schedule not in {"none", "constant"}:
            raise ValueError("recurrence_truncation_schedule must be one of: none, constant")
        if config.recurrence_truncation_schedule != "none" and config.recurrence_rmsnorm == "none":
            raise ValueError("recurrence truncation requires an inter-pass norm (recurrence_rmsnorm != none)")

        self.depth_scale_mode = mode
        self.core_repetitions = repetitions
        self.base_depth = config.n_layer
        self.recurrence_rmsnorm = config.recurrence_rmsnorm
        self.recurrence_emb_alpha = float(config.recurrence_emb_alpha)
        self.decoder_inject = bool(config.decoder_inject)

        if mode == "none":
            self.physical_depth = config.n_layer
            self.effective_depth = config.n_layer
            self.expected_effective_depth = float(config.n_layer)
            self.application_indices = list(range(config.n_layer))
            self._configure_middle_core_segments(config.n_layer)
            return

        if core_layer_count is None:
            core_layer_count = config.n_layer - self.PRELUDE_LAYERS - self.CODA_LAYERS
        core_size = int(core_layer_count)
        if core_size <= 0 or core_size > config.n_layer:
            raise ValueError("depth scaling requires at least one core layer")
        remaining = config.n_layer - core_size
        if prelude_layer_count is None and coda_layer_count is None:
            prelude_size = remaining // 2
            coda_size = remaining - prelude_size
        elif prelude_layer_count is None:
            coda_size = int(coda_layer_count)
            prelude_size = remaining - coda_size
        elif coda_layer_count is None:
            prelude_size = int(prelude_layer_count)
            coda_size = remaining - prelude_size
        else:
            prelude_size = int(prelude_layer_count)
            coda_size = int(coda_layer_count)
        if prelude_size < 0 or coda_size < 0 or prelude_size + coda_size != remaining:
            raise ValueError(
                "prelude_layer_count + coda_layer_count must equal n_layer - core_layer_count "
                f"(got prelude={prelude_size}, coda={coda_size}, remaining={remaining})"
            )

        prelude = list(range(prelude_size))
        core = list(range(prelude_size, prelude_size + core_size))
        coda = list(range(prelude_size + core_size, config.n_layer))
        # Total core-block applications. A fractional K must land on a whole block:
        # K=1.5 needs an even core, K=4/3 needs a core divisible by 3, and so on.
        core_applications = round(repetitions * core_size)
        if abs(repetitions * core_size - core_applications) > 1e-6:
            raise ValueError(
                "core_repetitions * core_layer_count must be an integer "
                f"(got {repetitions} * {core_size} = {repetitions * core_size})"
            )
        self.core_applications = core_applications
        self.effective_depth = prelude_size + core_applications + coda_size
        self.expected_effective_depth = float(self.effective_depth)
        self.prelude_size = prelude_size
        self.core_layer_count = core_size
        self.coda_size = coda_size
        self.prelude_indices = prelude
        self.track_core_residual = True

        if mode == "loop":
            # TIED: one physical core, cycled a,b,a,b,... A fractional K ends the
            # last pass early, on a prefix of the core (K=1.5, core [a,b] -> a,b | a).
            self.physical_depth = config.n_layer
            full_passes, partial = divmod(core_applications, core_size)
            self.core_recurrence_indices = [list(core) for _ in range(full_passes)]
            if partial:
                self.core_recurrence_indices.append(list(core[:partial]))
            self.coda_indices = coda
        else:
            # UNTIED: K distinct cores laid out back to back, so the stack is
            # physically as deep as it is effectively.
            self.physical_depth = self.effective_depth
            core_start = prelude_size
            self.core_recurrence_indices = [
                list(range(core_start + r * core_size, core_start + (r + 1) * core_size))
                for r in range(repetitions)
            ]
            coda_start = core_start + core_size * repetitions
            self.coda_indices = list(range(coda_start, coda_start + coda_size))

        self.core_application_indices = [i for rec in self.core_recurrence_indices for i in rec]
        self.application_indices = (
            self.prelude_indices + self.core_application_indices + self.coda_indices
        )

    def _configure_middle_core_segments(self, depth):
        """Label a Vanilla stack's layers as prelude/core/coda for the residual
        metrics only. Below 8 layers there is nothing to label."""
        if depth <= self.PRELUDE_LAYERS + self.CODA_LAYERS:
            self.prelude_indices = []
            self.core_application_indices = []
            self.core_recurrence_indices = []
            self.coda_indices = []
            self.track_core_residual = False
            return
        self.prelude_indices = list(range(self.PRELUDE_LAYERS))
        self.core_application_indices = list(range(self.PRELUDE_LAYERS, depth - self.CODA_LAYERS))
        self.core_recurrence_indices = [list(self.core_application_indices)]
        self.coda_indices = list(range(depth - self.CODA_LAYERS, depth))
        self.track_core_residual = True

    def _compute_window_sizes(self, config, depth):
        pattern = config.window_pattern.upper()
        long_w, short_w = config.sequence_len, config.sequence_len // 2
        char_to_w = {"L": (long_w, 0), "S": (short_w, 0)}
        sizes = [char_to_w[pattern[i % len(pattern)]] for i in range(depth)]
        sizes[-1] = (long_w, 0)      # the last layer always sees the full context
        return sizes

    def _precompute_rotary(self, seq_len, head_dim, base=10000):
        device = self.transformer["wte"].weight.device if hasattr(self, "transformer") else None
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos().bfloat16(), freqs.sin().bfloat16()
        return cos[None, :, None, :], sin[None, :, None, :]

    # ------------------------------------------------------------------- setup

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.normal_(self.transformer["wte"].weight, mean=0.0, std=self.config.wte_init_std)
        torch.nn.init.zeros_(self.lm_head.weight)
        init_block_weights(list(self.transformer["h"]), self.config)
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        if self.transformer["wte"].weight.device.type == "cuda":
            self.transformer["wte"].to(dtype=torch.bfloat16)

    def setup_optimizer(self, optimizer_transport="stacked"):
        """Build the optimizer with the selected paper gradient transport."""
        return build_optimizer(
            list(self.transformer["h"].parameters()),
            list(self.transformer["wte"].parameters()),
            list(self.lm_head.parameters()),
            self.config,
            optimizer_transport=optimizer_transport,
            parameter_order=list(self.parameters()),
        )

    def get_device(self):
        return self.transformer["wte"].weight.device

    # ------------------------------------------------------------- accounting

    def _avg_causal_attended_keys(self, window, seq_len):
        if window < 0 or window >= seq_len - 1:
            return (seq_len + 1) / 2
        max_keys = min(window + 1, seq_len)
        return max_keys - max_keys * (max_keys - 1) / (2 * seq_len)

    def estimate_flops(self, active_core_repetitions=None):
        """Forward+backward FLOPs per token, from the EXECUTED schedule.

        6 * (parameters actually applied to a token) + attention. A looped core is
        counted once per pass, so K2 and Dep come out identical -- which is the
        FLOPs-matching the paper's comparisons rest on. Never derive this from the
        parameter count: for a tied loop the two differ by a factor of K.
        """
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        head_params = sum(p.numel() for p in self.lm_head.parameters())
        application_indices = self.application_indices
        if active_core_repetitions is not None and self.depth_scale_mode in ("loop", "dep"):
            application_indices = (
                self.prelude_indices
                + [i for recurrence in self._resolve_core_recurrences(active_core_repetitions)
                   for i in recurrence]
                + self.coda_indices
            )
        block_params = sum(
            sum(p.numel() for p in self.transformer["h"][i].parameters())
            for i in application_indices
        )
        attn_flops = sum(
            12 * h * q * self._avg_causal_attended_keys(self.window_sizes[i][0], t)
            for i in application_indices
        )
        return 6 * (block_params + head_params) + attn_flops

    def parameter_summary(self):
        """Every parameter counts: blocks + embedding + untied lm_head (the
        Chinchilla convention). A tied loop stores ONE core no matter what K is."""
        block_params = sum(p.numel() for p in self.transformer["h"].parameters())
        lm_head_params = sum(p.numel() for p in self.lm_head.parameters())
        total = sum(p.numel() for p in self.parameters())
        return {
            "total": total,
            "parts": {
                "blocks": block_params,
                "lm_head": lm_head_params,
                "other": total - block_params - lm_head_params,
            },
        }

    # ----------------------------------------------------------------- forward

    def _resolve_core_recurrences(self, active_core_repetitions):
        """The core passes to run THIS step; fewer than K while a grow schedule is
        still in its low phase."""
        if active_core_repetitions is None:
            return self.core_recurrence_indices
        if self.depth_scale_mode not in ("loop", "dep"):
            raise ValueError("active_core_repetitions is only supported for loop/dep depth scaling")
        # Compare before truncating: at a fractional K the eval path passes K itself
        # and int(1.5) == 1 would silently drop the partial pass.
        if float(active_core_repetitions) == float(self.core_repetitions):
            return self.core_recurrence_indices
        active = int(active_core_repetitions)
        if self.depth_scale_mode == "dep":
            # dep: the first `active` UNTIED cores run, the rest stay dormant until
            # the grow step copy-initializes them (see dep_stack_grow_init).
            if float(active_core_repetitions) != float(active) or not 1 <= active <= self.core_repetitions:
                raise ValueError("dep active_core_repetitions must be an integer in [1, core_repetitions]")
            return self.core_recurrence_indices[:active]
        if active < 0:
            raise ValueError("active_core_repetitions must be non-negative")
        core = self.core_recurrence_indices[0] if self.core_recurrence_indices else []
        return [core for _ in range(active)]

    def _apply_block(self, x, block_idx, cos_sin):
        return self.transformer["h"][block_idx](x, cos_sin, self.window_sizes[block_idx])

    def _forward_blocks(self, x, cos_sin, num_core_stop_gradients=0, active_core_repetitions=None):
        if not self.track_core_residual:
            for block_idx in self.application_indices:
                x = self._apply_block(x, block_idx, cos_sin)
            return x, {}

        embed_in = x                                  # pre-prelude stream, for rms/res_enc
        for block_idx in self.prelude_indices:
            x = self._apply_block(x, block_idx, cos_sin)
        if self.recurrence_rmsnorm == "prelude_and_recurrence":
            x = norm(x)
        after_prelude = x
        anchor = after_prelude if self.recurrence_emb_alpha != 0.0 else None

        core_recurrences = self._resolve_core_recurrences(active_core_repetitions)
        core_res_changes = []                         # per-pass ||delta stream||, -> rms/res_core
        for recurrence_idx, recurrence in enumerate(core_recurrences, start=1):
            stream_in = x.detach()
            for block_idx in recurrence:
                x = self._apply_block(x, block_idx, cos_sin)
            core_res_changes.append((x.detach().float() - stream_in.float()).square().mean().sqrt())
            if self.recurrence_rmsnorm in ("prelude_and_recurrence", "recurrence_only"):
                x = norm(x)
            # Anchor re-injection: between passes, and once more at the core -> coda
            # boundary when decoder_inject is set.
            if anchor is not None and (recurrence_idx < len(core_recurrences) or self.decoder_inject):
                x = x + self.recurrence_emb_alpha * anchor
            # Truncated backprop: cut the graph after the first N passes.
            if recurrence_idx <= num_core_stop_gradients:
                x = x.detach()

        before_coda = x
        for block_idx in self.coda_indices:
            x = self._apply_block(x, block_idx, cos_sin)

        metrics = {
            # How far each stage moved the residual stream (detached, fp32).
            "core_block_residual_rms": (
                before_coda.float() - after_prelude.float()).square().mean().sqrt().detach(),
            "rms/res_enc": (after_prelude.float() - embed_in.float()).square().mean().sqrt().detach(),
            "rms/res_core": torch.stack(core_res_changes).mean().detach(),
            "rms/res_dec": (x.float() - before_coda.float()).square().mean().sqrt().detach(),
        }
        return x, metrics

    def _head_logits(self, hidden):
        logits = self.lm_head(hidden)[..., :self.config.vocab_size].float()
        return logits * self.config.output_multiplier

    def forward(
        self,
        idx,
        targets=None,
        loss_reduction="mean",
        num_core_stop_gradients=0,
        active_core_repetitions=None,
    ):
        _, T = idx.size()
        x = norm(self.transformer["wte"](idx)).float()
        cos_sin = (self.cos[:, :T], self.sin[:, :T])
        x, metrics = self._forward_blocks(
            x, cos_sin,
            num_core_stop_gradients=num_core_stop_gradients,
            active_core_repetitions=active_core_repetitions,
        )
        x = norm(x)
        logits = self._head_logits(x)
        if targets is None:
            return logits
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
            reduction=loss_reduction,
        )
        if loss_reduction != "mean":
            return loss
        return loss, {"lm_loss": loss, **metrics}


@torch.no_grad()
def dep_stack_grow_init(model, prev_active, new_active):
    """Stacking copy-init for a dep grow step: newly activated core j copies core
    (j % prev_active), i.e. 'C1 C2' -> 'C1 C2 C1 C2'.

    Weights AND buffers are copied in place; optimizer state is deliberately left
    alone (the dormant cores' Muon rows sit at ~zero, which IS the reset the fresh
    copies want, while the trained cores keep their momentum because the per-shape
    Muon group membership never changes -- see the dormant-core zero-grad injection
    in train.py). Rank-safe without a broadcast: DistMuonAdamW all-gathers updated
    parameters after every step, so every rank copies identical sources.

    Returns the [[dst_idx, src_idx], ...] block copies, recorded in result.json.
    """
    blocks = model.transformer["h"]
    copies = []
    for j in range(prev_active, new_active):
        src_recurrence = model.core_recurrence_indices[j % prev_active]
        dst_recurrence = model.core_recurrence_indices[j]
        for src_idx, dst_idx in zip(src_recurrence, dst_recurrence):
            src, dst = blocks[src_idx], blocks[dst_idx]
            for p_src, p_dst in zip(src.parameters(), dst.parameters()):
                p_dst.data.copy_(p_src.data)
            for b_src, b_dst in zip(src.buffers(), dst.buffers()):
                b_dst.data.copy_(b_src.data)
            copies.append([dst_idx, src_idx])
    return copies
