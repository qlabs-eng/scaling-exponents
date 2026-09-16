"""FlashAttention-3, with an explicit -- never silent -- fallback.

FA3 is fetched from the HF Hub by the `kernels` package at import time and needs a
Hopper GPU (compute capability 9.x). When it is unavailable this module still provides
a PyTorch SDPA implementation of the same function, but `train.py` refuses to start on
it unless LOOP_ALLOW_SDPA=1: the two paths differ in numerics and by roughly 2x in step
time, and a fallback that happens quietly is a fallback you discover after the sweep.
"""
import os

import torch
import torch.nn.functional as F

# Immutable snapshot of the FA3 v1 binary/wrappers verified in the release tests.
FLASH_ATTENTION_REVISION = "7cb368cf8278b583132eb72cbf312d54586df2e2"


def _load_fa3():
    if not torch.cuda.is_available():
        return None
    try:
        major, _ = torch.cuda.get_device_capability()
        if major != 9:                      # Hopper only
            return None
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        from kernels import get_kernel
        return get_kernel("kernels-community/flash-attn3", revision=FLASH_ATTENTION_REVISION).flash_attn_interface
    except Exception:                       # noqa: BLE001 -- hub fetch, driver, or import
        return None


_fa3 = _load_fa3()


def is_flash_attention_3_available():
    return _fa3 is not None


def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    """Attention over (B, T, H, D) tensors: FA3 when available, SDPA otherwise."""
    if _fa3 is not None and q.device.type == "cuda":
        return _fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)

    # SDPA fallback: (B, T, H, D) -> (B, H, T, D), expand KV heads if grouped, and
    # build an explicit band mask when a sliding window is requested.
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    if k.size(1) != q.size(1):
        repeat = q.size(1) // k.size(1)
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
    attn_mask, sdpa_causal = None, causal
    if causal and 0 <= window_size[0] < q.size(-2) - 1:
        T = q.size(-2)
        pos = torch.arange(T, device=q.device)
        attn_mask = (pos[None, :] <= pos[:, None]) & ((pos[:, None] - pos[None, :]) <= window_size[0])
        sdpa_causal = False
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=sdpa_causal)
    return y.transpose(1, 2)
