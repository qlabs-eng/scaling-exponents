"""Transformer block: pre-norm attention + SwiGLU MLP, both on scaled residual branches.

Deviations from a textbook GPT-2 block, all of them load-bearing for the paper:
  * no biases and no LayerNorm gains anywhere -- every norm is a plain RMSNorm with
    no learnable parameters, so depth changes nothing about the parameter budget;
  * QK-norm: q and k are RMS-normed after RoPE, which is what keeps the deep
    unrolled (looped) stacks stable at a shared learning rate;
  * each residual branch is multiplied by a per-block buffer
    `residual_branch_multipliers` initialized from --residual-branch-multiplier;
    this is the completeP-style RM knob the paper tunes against depth;
  * residual stream accumulations are kept in fp32 while branch compute stays in
    bf16 autocast -- a K-pass loop re-reads the same stream K times, and bf16
    accumulation there is visibly lossy.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .flash_attention import flash_attn_func


def norm(x):
    """Parameter-free RMSNorm."""
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, window_size):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)          # QK-norm
        y = flash_attn_func(q, k, v, causal=True, window_size=window_size)
        return self.c_proj(y.contiguous().view(B, T, -1))


class MLP(nn.Module):
    """SwiGLU, hidden width rounded up to a multiple of 256."""

    def __init__(self, config):
        super().__init__()
        hidden = 256 * ((8 * config.n_embd // 3 + 255) // 256)
        self.c_gate = nn.Linear(config.n_embd, hidden, bias=False)
        self.c_fc = nn.Linear(config.n_embd, hidden, bias=False)
        self.c_proj = nn.Linear(hidden, config.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.silu(self.c_gate(x)) * self.c_fc(x))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        # Buffer, not a parameter: the branch scale is a fixed init-time constant
        # (the RM knob), never trained. Buffers are rebuilt by init_weights(), which
        # is why checkpoints only need to carry parameters.
        self.register_buffer("residual_branch_multipliers", torch.ones(2))

    def forward(self, x, cos_sin, window_size):
        attn_alpha, mlp_alpha = self.residual_branch_multipliers
        x = x.float()
        x = x + attn_alpha.float() * self.attn(norm(x), cos_sin, window_size).float()
        return x + mlp_alpha.float() * self.mlp(norm(x)).float()


def init_block_weights(blocks, config):
    """Uniform init on the in-projections, ZERO on every out-projection.

    Zero-init of c_proj (and of lm_head, in the model) means each block starts as
    the identity and each residual branch grows in from nothing -- the property the
    looped arms depend on, since the same core is applied K times.
    """
    s = config.uniform_init_scale * config.n_embd ** -0.5
    for block in blocks:
        block.residual_branch_multipliers.fill_(config.residual_branch_multiplier)
        torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
        torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
        torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
        torch.nn.init.zeros_(block.attn.c_proj.weight)
        torch.nn.init.uniform_(block.mlp.c_gate.weight, -s, s)
        torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
        torch.nn.init.zeros_(block.mlp.c_proj.weight)
