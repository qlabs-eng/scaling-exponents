"""Distributed Muon (dense matrices) + AdamW (embedding, lm_head).

Muon orthogonalizes each update with a Polar-Express Newton-Schulz iteration, so the
step is spectrally normalized and the matrix learning rate does not have to be rescaled
with width. Parameters are sharded ZeRO-2 style: each rank reduce-scatters gradients,
updates the slice it owns, and all-gathers the result.
"""
from collections import deque

import torch
import torch.distributed as dist


def build_optimizer(matrix_params, embed_params, lm_head_params, config,
                    optimizer_transport="stacked", parameter_order=None):
    """Muon on the dense matrices, AdamW on the embedding and the lm_head.

    Three learning rates, one per parameter class, because they scale differently:
    Muon updates are spectrally normalized (so the matrix LR is width-invariant),
    while the embedding and head are ordinary AdamW tensors.

    Muon parameters are grouped BY SHAPE: the distributed step orthogonalizes a
    stacked batch of same-shaped matrices in one Newton-Schulz pass. A tied loop
    core therefore costs exactly one set of rows no matter how many times it runs.
    The paper's d26 run uses owner_bucketed transport to release gradients during
    backward; the other ladders use stacked transport at the optimizer step.
    """
    matrix_params = list(matrix_params)
    embed_params = list(embed_params)
    lm_head_params = list(lm_head_params)

    matrix_weight_decay = config.matrix_weight_decay
    embedding_weight_decay = config.embedding_weight_decay
    unembedding_weight_decay = config.unembedding_weight_decay
    embedding_wd_skip_norm = float(config.embedding_wd_skip_norm)
    if embedding_wd_skip_norm < 0:
        raise ValueError("embedding_wd_skip_norm must be non-negative")

    param_groups = []
    if lm_head_params:
        param_groups.append(dict(kind='adamw', params=lm_head_params, lr=config.unembedding_lr,
                                 betas=config.adam_betas, eps=config.adam_eps,
                                 weight_decay=unembedding_weight_decay))
    if embed_params:
        embed_group = dict(kind='adamw', params=embed_params, lr=config.embedding_lr,
                           betas=config.adam_betas, eps=config.adam_eps,
                           weight_decay=embedding_weight_decay, lr_role='embedding')
        if embedding_wd_skip_norm > 0:
            # Rows whose RMS norm is below this threshold are exempt from decay, so
            # rare tokens are not decayed to zero by updates they never receive.
            embed_group["stable_embedding_wd_eps"] = embedding_wd_skip_norm
        param_groups.append(embed_group)
    for shape in sorted({p.shape for p in matrix_params}):
        param_groups.append(dict(kind='muon', params=[p for p in matrix_params if p.shape == shape],
                                 lr=config.matrix_lr, momentum=0.95, ns_steps=5, beta2=0.95,
                                 weight_decay=matrix_weight_decay))

    if optimizer_transport == "stacked":
        optimizer = DistMuonAdamW(param_groups)
    elif optimizer_transport == "owner_bucketed":
        optimizer = OwnerBucketedMuonAdamW(param_groups, parameter_order=parameter_order)
    else:
        raise ValueError(f"unknown optimizer transport: {optimizer_transport}")
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
    return optimizer


# Polar Express coefficients: a fixed 5-step Newton-Schulz-style iteration that drives
# the singular values of the update toward 1 (Amsel et al., 2025).
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


def _muon_second_momentum_shape(count, shape):
    if shape[-2] >= shape[-1]:
        return (count, *shape[:-1], 1)
    return (count, *shape[:-2], 1, shape[-1])


def _muon_second_momentum_shape_single(shape):
    if shape[-2] >= shape[-1]:
        return (*shape[:-1], 1)
    return (*shape[:-2], 1, shape[-1])


@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    p.add_(exp_avg / ((exp_avg_sq / bias2).sqrt() + eps_t), alpha=-(lr_t / bias1))


def apply_adamw_weight_decay(p, lr, wd, stable_embedding_wd_eps=None):
    if wd == 0:
        return
    if stable_embedding_wd_eps is not None and p.ndim >= 2:
        reduce_dims = tuple(range(1, p.ndim))
        rms = p.float().square().mean(dim=reduce_dims, keepdim=True).sqrt()
        decay_mask = rms >= stable_embedding_wd_eps
        decay = torch.where(decay_mask, p.new_tensor(1 - lr * wd), p.new_tensor(1.0))
        p.mul_(decay)
    else:
        p.mul_(1 - lr * wd)


def adamw_step_plain(p, grad, exp_avg, exp_avg_sq, step, lr, beta1, beta2, eps, wd, stable_embedding_wd_eps=None):
    apply_adamw_weight_decay(p, lr, wd, stable_embedding_wd_eps)
    exp_avg.lerp_(grad, 1 - beta1)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2)
    bias1 = 1 - beta1 ** step
    bias2 = 1 - beta2 ** step
    p.add_(exp_avg / ((exp_avg_sq / bias2).sqrt() + eps), alpha=-(lr / bias1))


@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_lr_t, wd_t, beta2_t, ns_steps, red_dim):
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    # MuonEq-R row normalization
    g /= g.float().norm(dim=-1, keepdim=True).clamp_min(1e-7).to(g.dtype)
    # Polar Express orthogonalization
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            X = a * X + X @ (b * A + c * (A @ A))
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            X = a * X + (b * A + c * (A @ A)) @ X
    g = X
    # Variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    # Weight decay + update
    lr = lr_t.to(g.dtype)
    wd_lr = wd_lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    # Cautious weight decay: decay a weight only where the update agrees in sign
    # with it, so decay never fights an update that is already shrinking the weight.
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + wd_lr * wd * stacked_params * mask)


class DistMuonAdamW(torch.optim.Optimizer):
    """Distributed MuonAdamW with ZeRO-2 style sharding."""
    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        self._adamw_step_t = torch.tensor(0.0)
        self._adamw_lr_t = torch.tensor(0.0)
        self._adamw_beta1_t = torch.tensor(0.0)
        self._adamw_beta2_t = torch.tensor(0.0)
        self._adamw_eps_t = torch.tensor(0.0)
        self._adamw_wd_t = torch.tensor(0.0)
        self._muon_momentum_t = torch.tensor(0.0)
        self._muon_lr_t = torch.tensor(0.0)
        self._muon_wd_lr_t = torch.tensor(0.0)
        self._muon_wd_t = torch.tensor(0.0)
        self._muon_beta2_t = torch.tensor(0.0)

    def _reduce_adamw(self, group, world_size):
        infos = {}
        for p in group['params']:
            grad = p.grad
            if grad is None:
                continue
            if p.numel() < 1024 or p.shape[0] % world_size != 0:
                future = dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                infos[p] = dict(future=future, grad_slice=grad, is_small=True)
            else:
                assert grad.shape[0] % world_size == 0
                rank_size = grad.shape[0] // world_size
                grad_slice = torch.empty_like(grad[:rank_size])
                future = dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                infos[p] = dict(future=future, grad_slice=grad_slice, is_small=False)
        return dict(param_infos=infos)

    def _reduce_muon(self, group, world_size):
        params = [p for p in group['params'] if p.grad is not None]
        if not params:
            return dict(skip=True)
        chunk_size = (len(params) + world_size - 1) // world_size
        padded = chunk_size * world_size
        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype
        stacked_grads = torch.empty(padded, *shape, dtype=dtype, device=device)
        for index, param in enumerate(params):
            stacked_grads[index].copy_(param.grad)
        if len(params) < padded:
            stacked_grads[len(params):].zero_()
        grad_chunk = torch.empty(chunk_size, *shape, dtype=dtype, device=device)
        future = dist.reduce_scatter_tensor(grad_chunk, stacked_grads, op=dist.ReduceOp.AVG, async_op=True).get_future()
        return dict(future=future, grad_chunk=grad_chunk, stacked_grads=stacked_grads, chunk_size=chunk_size, params=params)

    def _compute_adamw(self, group, info, gather_list, rank, world_size):
        for p, pinfo in info['param_infos'].items():
            pinfo['future'].wait()
            state = self.state[p]
            if pinfo['is_small']:
                p_slice = p
            else:
                rank_size = p.shape[0] // world_size
                p_slice = p[rank * rank_size:(rank + 1) * rank_size]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p_slice)
                state['exp_avg_sq'] = torch.zeros_like(p_slice)
            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            stable_embedding_wd_eps = group.get("stable_embedding_wd_eps")
            use_stable_embedding_wd = stable_embedding_wd_eps is not None and p_slice.ndim >= 2
            if group.get("compile_step", True) and not use_stable_embedding_wd:
                adamw_step_fused(p_slice, pinfo['grad_slice'], state['exp_avg'], state['exp_avg_sq'],
                               self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                               self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)
            else:
                adamw_step_plain(
                    p_slice, pinfo['grad_slice'], state['exp_avg'], state['exp_avg_sq'],
                    state['step'], group['lr'], group['betas'][0], group['betas'][1],
                    group['eps'], group['weight_decay'],
                    stable_embedding_wd_eps if use_stable_embedding_wd else None,
                )
            if not pinfo['is_small']:
                future = dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future()
                gather_list.append(dict(future=future, params=None))

    def _compute_muon(self, group, info, gather_list, rank):
        if info.get("skip", False):
            return
        info['future'].wait()
        params = info['params']
        chunk_size = info['chunk_size']
        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype
        start_idx = rank * chunk_size
        num_owned = min(chunk_size, max(0, len(params) - start_idx))
        state = self.state[p]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(chunk_size, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            s = _muon_second_momentum_shape(chunk_size, shape)
            state["second_momentum_buffer"] = torch.zeros(s, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        updated = torch.empty(chunk_size, *shape, dtype=dtype, device=device)
        if num_owned > 0:
            owned = torch.stack([params[start_idx + i] for i in range(num_owned)])
            self._muon_momentum_t.fill_(group["momentum"])
            self._muon_beta2_t.fill_(group["beta2"])
            scaled_lr = group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5
            wd_lr = scaled_lr        # decay is applied at the SHAPE-SCALED Muon LR
            self._muon_lr_t.fill_(scaled_lr)
            self._muon_wd_lr_t.fill_(wd_lr)
            self._muon_wd_t.fill_(group["weight_decay"])
            muon_step_fused(info['grad_chunk'][:num_owned], owned,
                          state["momentum_buffer"][:num_owned], state["second_momentum_buffer"][:num_owned],
                          self._muon_momentum_t, self._muon_lr_t, self._muon_wd_lr_t,
                          self._muon_wd_t, self._muon_beta2_t, group["ns_steps"], red_dim,
                          )
            updated[:num_owned].copy_(owned)
        if num_owned < chunk_size:
            updated[num_owned:].zero_()
        stacked_params = info["stacked_grads"]
        future = dist.all_gather_into_tensor(stacked_params, updated, async_op=True).get_future()
        gather_list.append(dict(future=future, stacked_params=stacked_params, params=params))

    def _init_muon_single_state(self, p):
        state = self.state[p]
        if "momentum_buffer" not in state or state["momentum_buffer"].shape != p.shape:
            state["momentum_buffer"] = torch.zeros_like(p)
        second_shape = _muon_second_momentum_shape_single(p.shape)
        if (
            "second_momentum_buffer" not in state
            or state["second_momentum_buffer"].shape != second_shape
        ):
            state["second_momentum_buffer"] = torch.zeros(
                second_shape, dtype=p.dtype, device=p.device
            )
        return state

    def _muon_single_hparams(self, group, p):
        scaled_lr = group["lr"] * max(1.0, p.shape[-2] / p.shape[-1])**0.5
        wd_lr = scaled_lr        # decay is applied at the SHAPE-SCALED Muon LR
        red_dim = -1 if p.shape[-2] >= p.shape[-1] else -2
        return scaled_lr, wd_lr, red_dim

    def _step_adamw_local(self, group):
        for p in group['params']:
            grad = p.grad
            if grad is None:
                continue
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            stable_embedding_wd_eps = group.get("stable_embedding_wd_eps")
            use_stable_embedding_wd = stable_embedding_wd_eps is not None and p.ndim >= 2
            if group.get("compile_step", True) and not use_stable_embedding_wd:
                adamw_step_fused(p, grad, state['exp_avg'], state['exp_avg_sq'],
                                 self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                                 self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)
            else:
                adamw_step_plain(
                    p, grad, state['exp_avg'], state['exp_avg_sq'],
                    state['step'], group['lr'], group['betas'][0], group['betas'][1],
                    group['eps'], group['weight_decay'],
                    stable_embedding_wd_eps if use_stable_embedding_wd else None,
                )

    def _step_muon_local(self, group):
        params = [p for p in group['params'] if p.grad is not None]
        if not params:
            return
        p0 = params[0]
        shape, dtype, device = p0.shape, p0.dtype, p0.device
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack([p.detach() for p in params])
        state = self.state[p0]
        mb_shape = (len(params), *shape)
        if "momentum_buffer" not in state or state["momentum_buffer"].shape != mb_shape:
            state["momentum_buffer"] = torch.zeros(mb_shape, dtype=dtype, device=device)
        s = _muon_second_momentum_shape(len(params), shape)
        if "second_momentum_buffer" not in state or state["second_momentum_buffer"].shape != s:
            state["second_momentum_buffer"] = torch.zeros(s, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"])
        scaled_lr = group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5
        wd_lr = scaled_lr        # decay is applied at the SHAPE-SCALED Muon LR
        self._muon_lr_t.fill_(scaled_lr)
        self._muon_wd_lr_t.fill_(wd_lr)
        self._muon_wd_t.fill_(group["weight_decay"])
        muon_step_fused(stacked_grads, stacked_params,
                        state["momentum_buffer"], state["second_momentum_buffer"],
                        self._muon_momentum_t, self._muon_lr_t, self._muon_wd_lr_t,
                        self._muon_wd_t, self._muon_beta2_t, group["ns_steps"], red_dim,
                        )
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    def _step_local(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw_local(group)
            elif group['kind'] == 'muon':
                self._step_muon_local(group)

    @torch.no_grad()
    def step(self):
        if not dist.is_initialized():
            self._step_local()
            return
        rank, world_size = dist.get_rank(), dist.get_world_size()
        for group in self.param_groups:
            gather_list = []
            if group['kind'] == 'adamw':
                info = self._reduce_adamw(group, world_size)
                self._compute_adamw(group, info, gather_list, rank, world_size)
            elif group['kind'] == 'muon':
                info = self._reduce_muon(group, world_size)
                self._compute_muon(group, info, gather_list, rank)
            for gather_info in gather_list:
                gather_info["future"].wait()
                if gather_info.get("params") is not None:
                    torch._foreach_copy_(
                        gather_info["params"],
                        list(gather_info["stacked_params"][:len(gather_info["params"])].unbind(0)),
                    )


def run_muon_step(group, *args):
    """Use eager Muon in correctness tests and the compiled kernel in training."""
    fn = muon_step_fused
    if not group.get("compile_step", True):
        fn = getattr(muon_step_fused, "_torchdynamo_orig_callable", muon_step_fused)
    fn(*args)

class OwnerBucketedMuonAdamW(DistMuonAdamW):
    """Bounded owner-bucket transport used by the paper's d26 experiment.

    The update formulas and local fallback are shared with the stacked optimizer;
    these methods retain the original backward hooks, reduction order, ownership,
    and synchronization. The trainer brackets each backward with
    prepare_backward()/finish_backward(), declaring dormant parameters up front.
    """
    def __init__(self, param_groups, parameter_order=None, zero2_bucket_mb=256):
        super().__init__(param_groups)
        if zero2_bucket_mb <= 0:
            raise ValueError("zero2_bucket_mb must be positive")

        distributed_available = dist.is_initialized() and dist.get_world_size() > 1
        self._zero2 = distributed_available
        self._zero2_rank = dist.get_rank() if self._zero2 else 0
        self._zero2_world_size = dist.get_world_size() if self._zero2 else 1
        self._zero2_bucket_cap_bytes = int(zero2_bucket_mb) * 1024 * 1024
        self._zero2_buckets = []
        self._zero2_bucket_for_param = {}
        self._zero2_adam_params = set()
        self._zero2_adam_param_order = ()
        self._zero2_adam_pending = {}
        self._zero2_adam_grad_accum = {}
        self._zero2_adam_fired = set()
        self._zero2_hook_handles = []
        self._zero2_inflight = deque()
        self._zero2_next_bucket = 0
        self._zero2_prepared = False

        if self._zero2:
            params = [p for group in self.param_groups for p in group["params"]]
            kind_for_param = {
                p: group["kind"] for group in self.param_groups for p in group["params"]
            }
            self._zero2_adam_params = {
                p for p in params
                if kind_for_param[p] == "adamw"
                and p.numel() >= 1024
                and p.ndim > 0
                and p.shape[0] % self._zero2_world_size == 0
            }
            if parameter_order is None:
                parameter_order = params
            parameter_order = list(parameter_order)
            if set(parameter_order) != set(params) or len(parameter_order) != len(params):
                raise ValueError("parameter_order must contain every optimizer parameter exactly once")
            self._zero2_adam_param_order = tuple(
                p for p in parameter_order if p in self._zero2_adam_params
            )
            bucket_order = [
                p for p in reversed(parameter_order) if p not in self._zero2_adam_params
            ]
            self._build_zero2_buckets(bucket_order)
            for p in params:
                self._zero2_hook_handles.append(
                    p.register_post_accumulate_grad_hook(self._zero2_post_accumulate)
                )


    @property
    def uses_zero2(self):
        return self._zero2


    def _build_zero2_buckets(self, backward_order):
        """Make deterministic, whole-parameter buckets in expected backward order."""
        current = []
        current_bytes = 0
        current_dtype = None

        def flush():
            nonlocal current, current_bytes, current_dtype
            if not current:
                return
            index = len(self._zero2_buckets)
            bucket = {
                "index": index,
                "owner": index % self._zero2_world_size,
                "params": tuple(current),
                "bytes": current_bytes,
                "active": (),
                "offsets": {},
                "numel": 0,
                "fired": set(),
                "buffer": None,
                "grad_accum": None,
                "ready": False,
            }
            self._zero2_buckets.append(bucket)
            for p in current:
                self._zero2_bucket_for_param[p] = bucket
            current = []
            current_bytes = 0
            current_dtype = None

        for p in backward_order:
            size_bytes = p.numel() * p.element_size()
            would_overflow = current and current_bytes + size_bytes > self._zero2_bucket_cap_bytes
            if current and (p.dtype != current_dtype or would_overflow):
                flush()
            current.append(p)
            current_bytes += size_bytes
            current_dtype = p.dtype
            if current_bytes >= self._zero2_bucket_cap_bytes:
                flush()
        flush()


    def zero2_stats(self):
        if not self._zero2:
            return {"enabled": False}
        owned = [b for b in self._zero2_buckets if b["owner"] == self._zero2_rank]
        adam_shard_bytes = sum(
            p.numel() * p.element_size() // self._zero2_world_size
            for p in self._zero2_adam_params
        )
        owned_bytes = sum(b["bytes"] for b in owned) + adam_shard_bytes
        stats_device = self.param_groups[0]["params"][0].device
        owned_min = torch.tensor(owned_bytes, dtype=torch.int64, device=stats_device)
        owned_max = owned_min.clone()
        dist.all_reduce(owned_min, op=dist.ReduceOp.MIN)
        dist.all_reduce(owned_max, op=dist.ReduceOp.MAX)
        return {
            "enabled": True,
            "world_size": self._zero2_world_size,
            "bucket_count": len(self._zero2_buckets) + len(self._zero2_adam_params),
            "muon_bucket_count": len(self._zero2_buckets),
            "row_sharded_adamw_tensor_count": len(self._zero2_adam_params),
            "bucket_cap_mb": self._zero2_bucket_cap_bytes / 1024**2,
            "owned_parameter_bytes": owned_bytes,
            "min_owned_parameter_bytes": owned_min.item(),
            "max_owned_parameter_bytes": owned_max.item(),
            "largest_bucket_bytes": max(b["bytes"] for b in self._zero2_buckets),
        }


    def prepare_backward(self, inactive_params=()):
        """Declare this micro-step's inactive parameters before backward.

        Dep Grow passes the not-yet-active core parameters here. Fully inactive
        buckets issue no collective; mixed boundary buckets pack only their live
        members. Stable bucket ownership keeps existing state pinned across grow.
        """
        if not self._zero2:
            return
        if self._zero2_prepared:
            raise RuntimeError("finish_backward must be called before the next prepare_backward")
        if self._zero2_inflight:
            raise RuntimeError("unfinished ZeRO-2 reductions from the previous backward")
        inactive = set(inactive_params)
        if inactive & self._zero2_adam_params:
            raise RuntimeError("row-sharded AdamW parameters cannot be inactive")
        if self._zero2_adam_pending:
            raise RuntimeError("unfinished AdamW reductions from the previous backward")
        self._zero2_adam_fired = set()
        self._zero2_next_bucket = 0
        for bucket in self._zero2_buckets:
            active = tuple(p for p in bucket["params"] if p not in inactive)
            offsets = {}
            offset = 0
            for p in active:
                offsets[p] = (offset, offset + p.numel())
                offset += p.numel()
            bucket["active"] = active
            bucket["offsets"] = offsets
            bucket["numel"] = offset
            bucket["fired"] = set()
            bucket["buffer"] = None
            bucket["ready"] = not active
        self._zero2_prepared = True
        self._zero2_drain_ready()


    def _zero2_post_accumulate(self, p):
        if not self._zero2_prepared:
            raise RuntimeError("prepare_backward must be called before backward in ZeRO-2 mode")
        if p in self._zero2_adam_params:
            if p in self._zero2_adam_fired:
                raise RuntimeError("a row-sharded AdamW parameter accumulated more than once")
            if p.grad is None:
                raise RuntimeError("post-accumulate hook ran without an AdamW gradient")
            rank_rows = p.shape[0] // self._zero2_world_size
            grad_shard = torch.empty_like(p[:rank_rows])
            work = dist.reduce_scatter_tensor(
                grad_shard, p.grad, op=dist.ReduceOp.SUM, async_op=True
            )
            p.grad = None
            self._zero2_adam_pending[p] = (work, grad_shard)
            self._zero2_adam_fired.add(p)
            return
        bucket = self._zero2_bucket_for_param[p]
        if p not in bucket["offsets"]:
            raise RuntimeError("an inactive ZeRO-2 parameter unexpectedly received a gradient")
        if p in bucket["fired"]:
            raise RuntimeError("a ZeRO-2 parameter accumulated more than once in one backward")
        if p.grad is None:
            raise RuntimeError("post-accumulate hook ran without a gradient")
        if bucket["buffer"] is None:
            bucket["buffer"] = torch.empty(
                bucket["numel"], dtype=p.dtype, device=p.device
            )
        start, end = bucket["offsets"][p]
        bucket["buffer"][start:end].copy_(p.grad.reshape(-1))
        p.grad = None
        bucket["fired"].add(p)
        if len(bucket["fired"]) == len(bucket["active"]):
            bucket["ready"] = True
        self._zero2_drain_ready()


    def _zero2_drain_ready(self):
        while self._zero2_next_bucket < len(self._zero2_buckets):
            bucket = self._zero2_buckets[self._zero2_next_bucket]
            if not bucket["ready"]:
                break
            self._zero2_next_bucket += 1
            if not bucket["active"]:
                continue
            work = dist.reduce(
                bucket["buffer"],
                dst=bucket["owner"],
                op=dist.ReduceOp.SUM,
                async_op=True,
            )
            self._zero2_inflight.append((bucket, work, bucket["buffer"]))
            bucket["buffer"] = None
            # One in-flight bucket overlaps communication with backward while
            # bounding temporary full-gradient storage to roughly two buckets.
            if len(self._zero2_inflight) >= 2:
                self._zero2_finalize_oldest()


    def _zero2_finalize_oldest(self):
        bucket, work, buffer = self._zero2_inflight.popleft()
        work.wait()
        if bucket["owner"] == self._zero2_rank:
            buffer.mul_(1.0 / self._zero2_world_size)
            if bucket["grad_accum"] is None:
                bucket["grad_accum"] = buffer
            else:
                bucket["grad_accum"].add_(buffer)


    def finish_backward(self):
        if not self._zero2:
            return
        if not self._zero2_prepared:
            raise RuntimeError("finish_backward called without prepare_backward")
        self._zero2_drain_ready()
        if self._zero2_next_bucket != len(self._zero2_buckets):
            pending = self._zero2_buckets[self._zero2_next_bucket]
            missing = len(pending["active"]) - len(pending["fired"])
            raise RuntimeError(
                f"ZeRO-2 bucket {pending['index']} is missing {missing} gradients after backward"
            )
        while self._zero2_inflight:
            self._zero2_finalize_oldest()
        if self._zero2_adam_fired != self._zero2_adam_params:
            missing = len(self._zero2_adam_params - self._zero2_adam_fired)
            raise RuntimeError(f"ZeRO-2 backward is missing {missing} AdamW gradients")
        for p, (work, grad_shard) in self._zero2_adam_pending.items():
            work.wait()
            grad_shard.mul_(1.0 / self._zero2_world_size)
            accumulated = self._zero2_adam_grad_accum.get(p)
            if accumulated is None:
                self._zero2_adam_grad_accum[p] = grad_shard
            else:
                accumulated.add_(grad_shard)
        self._zero2_adam_pending = {}
        self._zero2_prepared = False


    def _zero2_grad(self, p):
        if p in self._zero2_adam_params:
            return self._zero2_adam_grad_accum.get(p)
        bucket = self._zero2_bucket_for_param[p]
        if bucket["owner"] != self._zero2_rank or bucket["grad_accum"] is None:
            return None
        bounds = bucket["offsets"].get(p)
        if bounds is None:
            return None
        start, end = bounds
        return bucket["grad_accum"][start:end].view_as(p)


    def _step_adamw_zero2(self, group):
        for p in group["params"]:
            grad = self._zero2_grad(p)
            if grad is None:
                continue
            if p in self._zero2_adam_params:
                rank_rows = p.shape[0] // self._zero2_world_size
                start = self._zero2_rank * rank_rows
                update_param = p[start:start + rank_rows]
            else:
                update_param = p
            state = self.state[p]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(update_param)
                state["exp_avg_sq"] = torch.zeros_like(update_param)
            state["step"] += 1
            self._adamw_step_t.fill_(state["step"])
            self._adamw_lr_t.fill_(group["lr"])
            self._adamw_beta1_t.fill_(group["betas"][0])
            self._adamw_beta2_t.fill_(group["betas"][1])
            self._adamw_eps_t.fill_(group["eps"])
            self._adamw_wd_t.fill_(group["weight_decay"])
            stable_embedding_wd_eps = group.get("stable_embedding_wd_eps")
            use_stable_embedding_wd = stable_embedding_wd_eps is not None and p.ndim >= 2
            if group.get("compile_step", True) and not use_stable_embedding_wd:
                adamw_step_fused(
                    update_param, grad, state["exp_avg"], state["exp_avg_sq"],
                    self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                    self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t,
                )
            else:
                adamw_step_plain(
                    update_param, grad, state["exp_avg"], state["exp_avg_sq"],
                    state["step"], group["lr"], group["betas"][0], group["betas"][1],
                    group["eps"], group["weight_decay"],
                    stable_embedding_wd_eps if use_stable_embedding_wd else None,
                )


    def _step_muon_zero2(self, group):
        params = [p for p in group["params"] if self._zero2_grad(p) is not None]
        if not params:
            return
        grads = [self._zero2_grad(p) for p in params]
        momentum = []
        second_momentum = []
        for p in params:
            state = self._init_muon_single_state(p)
            momentum.append(state["momentum_buffer"])
            second_momentum.append(state["second_momentum_buffer"])

        stacked_grads = torch.stack(grads)
        stacked_params = torch.stack([p.detach() for p in params])
        stacked_momentum = torch.stack(momentum)
        stacked_second_momentum = torch.stack(second_momentum)
        shape = params[0].shape
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"])
        scaled_lr = group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5
        self._muon_lr_t.fill_(scaled_lr)
        self._muon_wd_lr_t.fill_(scaled_lr)
        self._muon_wd_t.fill_(group["weight_decay"])
        run_muon_step(
            group,
            stacked_grads,
            stacked_params,
            stacked_momentum,
            stacked_second_momentum,
            self._muon_momentum_t,
            self._muon_lr_t,
            self._muon_wd_lr_t,
            self._muon_wd_t,
            self._muon_beta2_t,
            group["ns_steps"],
            red_dim,
        )
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))
        torch._foreach_copy_(momentum, list(stacked_momentum.unbind(0)))
        torch._foreach_copy_(second_momentum, list(stacked_second_momentum.unbind(0)))


    def _sync_zero2_parameters(self):
        """Broadcast each active owner bucket with one bounded temporary buffer."""
        for bucket in self._zero2_buckets:
            if not bucket["active"]:
                bucket["grad_accum"] = None
                continue
            if self._zero2_rank == bucket["owner"]:
                buffer = bucket["grad_accum"]
                if buffer is None:
                    raise RuntimeError(f"owner has no reduced gradient for bucket {bucket['index']}")
                for p in bucket["active"]:
                    start, end = bucket["offsets"][p]
                    buffer[start:end].copy_(p.detach().reshape(-1))
            else:
                p0 = bucket["active"][0]
                buffer = torch.empty(bucket["numel"], dtype=p0.dtype, device=p0.device)
            dist.broadcast(buffer, src=bucket["owner"])
            if self._zero2_rank != bucket["owner"]:
                for p in bucket["active"]:
                    start, end = bucket["offsets"][p]
                    p.copy_(buffer[start:end].view_as(p))
            bucket["grad_accum"] = None

        # A Python set's iteration order depends on per-process object addresses.
        # Every rank must issue these collectives in the exact same model order.
        for p in self._zero2_adam_param_order:
            rank_rows = p.shape[0] // self._zero2_world_size
            start = self._zero2_rank * rank_rows
            p_shard = p[start:start + rank_rows]
            dist.all_gather_into_tensor(p, p_shard)
            self._zero2_adam_grad_accum.pop(p, None)


    @torch.no_grad()
    def step(self):
        if not self._zero2:
            self._step_local()
            return
        if self._zero2_prepared or self._zero2_inflight:
            raise RuntimeError("finish_backward must complete before optimizer.step")
        if (not any(bucket["active"] for bucket in self._zero2_buckets)
                and not self._zero2_adam_grad_accum):
            raise RuntimeError("optimizer.step called without an active ZeRO-2 backward plan")
        for group in self.param_groups:
            if group["kind"] == "adamw":
                self._step_adamw_zero2(group)
            elif group["kind"] == "muon":
                self._step_muon_zero2(group)
        self._sync_zero2_parameters()
