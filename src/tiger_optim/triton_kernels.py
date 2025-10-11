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
        device = updates[0].device if len(updates)>0 else "cpu"
        return torch.tensor(0.0, dtype=torch.float32, device=device), torch.tensor(0.0, dtype=torch.float32, device=device), torch.tensor(0.0, dtype=torch.float32, device=device)

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
    if triton is None:
        raise RuntimeError("Triton not available")
    # Concatenate into big buffers and remember ranges
    p_views, u_views, ranges = [], [], []
    total = 0
    for P, U in zip(params, updates):
        pv = P.contiguous().view(-1)
        uv = U.contiguous().view(-1).to(P.dtype)
        n = pv.numel()
        p_views.append(pv); u_views.append(uv)
        ranges.append((total, total+n))
        total += n
    Pbig = torch.cat(p_views, dim=0)
    Ubig = torch.cat(u_views, dim=0)

    @triton.jit
    def apply_sum_kernel(p_ptr, u_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        p = tl.load(p_ptr + offs, mask=mask, other=0.0)
        u = tl.load(u_ptr + offs, mask=mask, other=0.0)
        # apply
        p = p + u
        tl.store(p_ptr + offs, p, mask=mask)
        # local ssq accum (not returned here; could be extended to a reduction buffer)
        # (intentionally minimal to avoid extra writes)
        # A production version would write partial sums to a buffer for host reduction.

    BLOCK = 1024
    grid = ( (total + BLOCK - 1)//BLOCK, )
    apply_sum_kernel[grid](Pbig, Ubig, total, BLOCK=BLOCK)

    # NOTE: This 'fused' path focuses on in-place apply; bucket-wide stats are recommended
    # via bucket_stats_triton to avoid a second global memory pass.
    return True
