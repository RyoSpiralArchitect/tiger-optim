# ============================================================================
#  Project: SpiralReality / Tiger Optimizer
#  Copyright (c) 2025 Ryo ∴ SpiralArchitect and SpiralReality
#
#  This file is part of SpiralReality.
#
#  SpiralReality is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published
#  by the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  SpiralReality is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with SpiralReality.  If not, see <https://www.gnu.org/licenses/>.
# ============================================================================
# Triton kernels: (1) bucket_stats_triton (ussq, pssq, n), (2) fused_apply_updates (experimental)

try:
    import triton
    import triton.language as tl
    import torch
except Exception:
    triton = None

def _concat_views(tensors):
    flats = [t.contiguous().view(-1) for t in tensors]
    if len(flats)==0:
        return None
    return torch.cat(flats, dim=0)

def bucket_stats_triton(updates, params):
    if triton is None:
        raise RuntimeError("Triton not available")
    u = _concat_views(updates)
    p = _concat_views(params)
    if u is None:
        reference = updates[0] if updates else (params[0] if params else None)
        device = reference.device if reference is not None else torch.device("cpu")
        zero = torch.zeros((), dtype=torch.float32, device=device)
        return zero, zero.clone(), zero.clone()

    N = u.numel()
    if p is None or p.numel()==0:
        p = torch.zeros(1, device=u.device, dtype=u.dtype)

    @triton.jit
    def ssq_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        acc = tl.sum(x * x, axis=0)
        tl.store(out_ptr + pid, acc)

    def _ssq(x):
        BLOCK = 1024
        grid = ( (x.numel() + BLOCK - 1)//BLOCK, )
        out = torch.empty(grid[0], device=x.device, dtype=x.dtype)
        ssq_kernel[grid](x, out, x.numel(), BLOCK=BLOCK)
        return out.sum()

    ussq = _ssq(u)
    pssq = _ssq(p)
    n = torch.tensor(float(N), device=u.device, dtype=torch.float32)
    return ussq.to(torch.float32), pssq.to(torch.float32), n

# Experimental: fused apply updates + per-bucket sums in one pass
def fused_apply_updates(params, updates):
    if not params or not updates:
        return False, "empty"
    if len(params) != len(updates):
        raise ValueError("params and updates must have identical lengths")

    target_device = params[0].device
    target_dtype = params[0].dtype

    if target_device.type != "cuda":
        return False, "non_cuda_bucket"

    apply_inputs = []
    total = 0

    for param, update in zip(params, updates):
        if param.device != target_device:
            return False, "param_device_mismatch"
        if param.dtype != target_dtype:
            return False, "param_dtype_mismatch"
        if update.shape != param.shape:
            return False, "shape_mismatch"
        if update.device != target_device:
            return False, "update_device_mismatch"
        if update.dtype != target_dtype:
            return False, "update_dtype_mismatch"

        n = param.numel()
        if n:
            try:
                flat_param = param.view(-1)
            except RuntimeError:
                return False, "param_non_contiguous"
            try:
                flat_update = update.view(-1)
            except RuntimeError:
                return False, "update_non_contiguous"
            apply_inputs.append((flat_param, flat_update, n))
            total += n

    if total == 0:
        return False, "empty"

    if triton is None:
        raise RuntimeError("Triton not available")

    @triton.jit
    def apply_sum_kernel(p_ptr, u_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        p = tl.load(p_ptr + offs, mask=mask, other=0.0)
        u = tl.load(u_ptr + offs, mask=mask, other=0.0)
        p = p + u
        tl.store(p_ptr + offs, p, mask=mask)

    BLOCK = 1024
    for flat_param, flat_update, n in apply_inputs:
        grid = ((n + BLOCK - 1) // BLOCK,)
        apply_sum_kernel[grid](flat_param, flat_update, n, BLOCK=BLOCK)

    # NOTE: This 'fused' path focuses on in-place apply; bucket-wide stats are recommended
    # via bucket_stats_triton to avoid a second global memory pass.
    return True, "ok"
