# ============================================================================
#  Project: SpiralReality / Tiger Optimizer
#  Copyright (c) 2025 Ryo ∴ SpiralArchitect and SpiralReality
#
#  This file is part of SpiralReality.
#
#  SpiralReality is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published
#  by the Free Software Foundation, version 3 of the License.
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


# ============================================================================
# Flat fused candidate/apply path.
#
# Fuses the per-parameter moment/direction math for "simple" parameters
# (fp32 param/grad/state, dense v, softsign blending, positive alpha) onto
# concatenated flat buffers. Sequence, all async:
#   K1 _flat_cand_kernel : cand_m, cand_v, Pbuf, dsign
#   K2 _flat_seg_stats_kernel : per-param amax_m/amax_v/amax_p/amax_dsign/ssq_m
#   K3 _flat_dir_kernel : d = (1-sb)*dsign + sb*(m/rms)*P
#   K4 _flat_seg_stats_kernel : amax_d
#   K5 _flat_apply_kernel : p = p*(1-decay) - lr*eff*d ; M,V <- cand (ok only)
# The host performs one guard sync per group between K4 and K5.
# ============================================================================

if triton is not None:

    @triton.jit
    def _flat_cand_kernel(g_ptr, m_ptr, v_ptr, cm_ptr, cv_ptr, pp_ptr, ds_ptr,
                          N, b1, b2, eps, alpha, sign_tau, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        g = tl.load(g_ptr + offs, mask=mask, other=0.0)
        m = tl.load(m_ptr + offs, mask=mask, other=0.0)
        v = tl.load(v_ptr + offs, mask=mask, other=0.0)
        cm = m * b1 + g * (1.0 - b1)
        cv = v * b2 + g * g * (1.0 - b2)
        pp = tl.sqrt(cv) + eps
        pp = tl.exp(-alpha * tl.log(tl.maximum(pp, 1e-30)))
        pp = tl.minimum(tl.maximum(pp, 1e-8), 1e8)
        ds = cm / (tl.abs(cm) + sign_tau) * pp
        tl.store(cm_ptr + offs, cm, mask=mask)
        tl.store(cv_ptr + offs, cv, mask=mask)
        tl.store(pp_ptr + offs, pp, mask=mask)
        tl.store(ds_ptr + offs, ds, mask=mask)

    @triton.jit
    def _flat_seg_stats_kernel(cm_ptr, cv_ptr, p_ptr, x_ptr, start, end,
                               out_ptr, BLOCK: tl.constexpr):
        # Program j reduces param j's segment [start_j, end_j) passed via
        # two pointer arrays indexed by program id.
        j = tl.program_id(0)
        s = tl.load(start + j)
        e = tl.load(end + j)
        n = e - s
        am_m = 0.0
        am_v = 0.0
        am_p = 0.0
        am_x = 0.0
        for off in range(0, n, BLOCK):
            idx = s + off + tl.arange(0, BLOCK)
            mask = idx < e
            cm = tl.load(cm_ptr + idx, mask=mask, other=0.0)
            cv = tl.load(cv_ptr + idx, mask=mask, other=0.0)
            pp = tl.load(p_ptr + idx, mask=mask, other=0.0)
            xx = tl.load(x_ptr + idx, mask=mask, other=0.0)
            am_m = tl.maximum(am_m, tl.max(tl.abs(cm), axis=0))
            am_v = tl.maximum(am_v, tl.max(tl.abs(cv), axis=0))
            am_p = tl.maximum(am_p, tl.max(tl.abs(pp), axis=0))
            am_x = tl.maximum(am_x, tl.max(tl.abs(xx), axis=0))
        # Second pass: scaled sum-of-squares so finite moments near the FP32
        # limit do not overflow (norm = am_m * sqrt(ssq_scaled / n)).
        ssq_m = 0.0
        if am_m > 0.0:
            inv = 1.0 / am_m
            for off in range(0, n, BLOCK):
                idx = s + off + tl.arange(0, BLOCK)
                mask = idx < e
                cm = tl.load(cm_ptr + idx, mask=mask, other=0.0)
                cs = cm * inv
                ssq_m += tl.sum(tl.where(mask, cs * cs, 0.0), axis=0)
        tl.store(out_ptr + j * 5 + 0, am_m)
        tl.store(out_ptr + j * 5 + 1, am_v)
        tl.store(out_ptr + j * 5 + 2, am_p)
        tl.store(out_ptr + j * 5 + 3, am_x)
        tl.store(out_ptr + j * 5 + 4, ssq_m)

    @triton.jit
    def _flat_seg_ssq_kernel(x_ptr, start, end, out_ptr, BLOCK: tl.constexpr):
        # Per-segment sum of squares, sequential within each segment so the
        # result is deterministic — and, unlike a dense indicator GEMV, a
        # nonfinite element contaminates only its own segment.
        j = tl.program_id(0)
        s = tl.load(start + j)
        e = tl.load(end + j)
        n = e - s
        acc = 0.0
        for off in range(0, n, BLOCK):
            idx = s + off + tl.arange(0, BLOCK)
            mask = idx < e
            x = tl.load(x_ptr + idx, mask=mask, other=0.0)
            acc += tl.sum(tl.where(mask, x * x, 0.0), axis=0)
        tl.store(out_ptr + j, acc)

    @triton.jit
    def _flat_dir_kernel(cm_ptr, pp_ptr, ds_ptr, rms_ptr, seg_ptr, d_ptr,
                         N, sb_ptr, eps, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        # sblend arrives via device pointer so CUDA-graph replays see the
        # current value even when the blend schedule changes every step.
        sblend = tl.load(sb_ptr)
        seg = tl.load(seg_ptr + offs, mask=mask, other=0)
        rms = tl.load(rms_ptr + seg, mask=mask, other=1.0)
        cm = tl.load(cm_ptr + offs, mask=mask, other=0.0)
        pp = tl.load(pp_ptr + offs, mask=mask, other=0.0)
        ds = tl.load(ds_ptr + offs, mask=mask, other=0.0)
        dmag = cm / (rms + eps) * pp
        d = ds * (1.0 - sblend) + dmag * sblend
        tl.store(d_ptr + offs, d, mask=mask)

    @triton.jit
    def _flat_fact_rows_kernel(g_ptr, vr_ptr, cvr_ptr, rstart_ptr, rlen_ptr,
                               b2, BLOCK: tl.constexpr):
        # One program per matrix row across all factored params: reduce the
        # row's g^2 mean and EMA-update the candidate row factor.
        j = tl.program_id(0)
        s = tl.load(rstart_ptr + j)
        n = tl.load(rlen_ptr + j)
        # Divide before summing, matching the eager (g2 / n).sum(dim=1):
        # a finite mean must not overflow in the intermediate sum.
        acc = 0.0
        for off in range(0, n, BLOCK):
            idx = s + off + tl.arange(0, BLOCK)
            mask = off + tl.arange(0, BLOCK) < n
            g = tl.load(g_ptr + idx, mask=mask, other=0.0)
            acc += tl.sum(tl.where(mask, g * g / n, 0.0), axis=0)
        vr = tl.load(vr_ptr + j)
        tl.store(cvr_ptr + j, vr * b2 + acc * (1.0 - b2))

    @triton.jit
    def _flat_fact_cols_kernel(g_ptr, vc_ptr, cvc_ptr, cstart_ptr, cc0_ptr,
                               cr_ptr, cc_ptr, cvc_ptr_off,
                               b2, BLOCK: tl.constexpr):
        # One program per column block: accumulate g^2 down the rows
        # (stride C) and EMA-update the candidate column factor.
        j = tl.program_id(0)
        s = tl.load(cstart_ptr + j)
        c0 = tl.load(cc0_ptr + j)
        R = tl.load(cr_ptr + j)
        C = tl.load(cc_ptr + j)
        vo = tl.load(cvc_ptr_off + j)
        cols = c0 + tl.arange(0, BLOCK)
        cmask = cols < C
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for r in range(0, R):
            g = tl.load(g_ptr + s + r * C + cols, mask=cmask, other=0.0)
            acc += g * g / R
        vc = tl.load(vc_ptr + vo + tl.arange(0, BLOCK), mask=cmask, other=0.0)
        tl.store(cvc_ptr + vo + tl.arange(0, BLOCK),
                 vc * b2 + acc * (1.0 - b2), mask=cmask)

    @triton.jit
    def _flat_fact_cols_partial_kernel(g_ptr, part_ptr, pb_start_ptr, pb_c0_ptr,
                                     pb_r0_ptr, pb_r1_ptr, pb_C_ptr, pb_R_ptr,
                                     BLOCK: tl.constexpr):
        # Phase 1 of the two-phase column reduce for tall matrices: each
        # program covers one (col-block, row-block) pair and writes a scaled
        # partial sum (already divided by R, matching eager (g2/R).sum(dim=0)).
        j = tl.program_id(0)
        s = tl.load(pb_start_ptr + j)
        c0 = tl.load(pb_c0_ptr + j)
        r0 = tl.load(pb_r0_ptr + j)
        r1 = tl.load(pb_r1_ptr + j)
        C = tl.load(pb_C_ptr + j)
        R = tl.load(pb_R_ptr + j)
        cols = c0 + tl.arange(0, BLOCK)
        cmask = cols < C
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for r in range(r0, r1):
            g = tl.load(g_ptr + s + r * C + cols, mask=cmask, other=0.0)
            acc += g * g / R
        tl.store(part_ptr + j * BLOCK + tl.arange(0, BLOCK), acc, mask=cmask)

    @triton.jit
    def _flat_fact_cols_finish_kernel(vc_ptr, cvc_ptr, part_ptr, cbf_vc_ptr,
                                      cbf_p0_ptr, cbf_np_ptr, cbf_rem_ptr,
                                      b2, BLOCK: tl.constexpr):
        # Phase 2: fold a col-block's row-block partials into the candidate
        # column factor EMA update.
        j = tl.program_id(0)
        vo = tl.load(cbf_vc_ptr + j)
        p0 = tl.load(cbf_p0_ptr + j)
        np_ = tl.load(cbf_np_ptr + j)
        rem = tl.load(cbf_rem_ptr + j)
        cmask = tl.arange(0, BLOCK) < rem
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for t in range(0, np_):
            acc += tl.load(part_ptr + (p0 + t) * BLOCK + tl.arange(0, BLOCK),
                           mask=cmask, other=0.0)
        vc = tl.load(vc_ptr + vo + tl.arange(0, BLOCK), mask=cmask, other=0.0)
        tl.store(cvc_ptr + vo + tl.arange(0, BLOCK), vc * b2 + acc * (1.0 - b2),
                 mask=cmask)

    @triton.jit
    def _flat_fact_mean_kernel(cvr_ptr, fvroff_ptr, fr_ptr, fmean_ptr,
                               BLOCK: tl.constexpr):
        # Per factored param: mean of the candidate row factor (scalar).
        j = tl.program_id(0)
        off = tl.load(fvroff_ptr + j)
        R = tl.load(fr_ptr + j)
        # Divide before summing so a finite mean cannot overflow the
        # intermediate accumulator (matches eager (vr / R).sum()).
        acc = 0.0
        for o in range(0, R, BLOCK):
            mask = o + tl.arange(0, BLOCK) < R
            v = tl.load(cvr_ptr + off + o + tl.arange(0, BLOCK),
                        mask=mask, other=0.0)
            acc += tl.sum(tl.where(mask, v / R, 0.0), axis=0)
        tl.store(fmean_ptr + j, acc)

    @triton.jit
    def _flat_fact_vhat_kernel(cm_ptr, cvr_ptr, cvc_ptr, fmean_ptr,
                               pos_ptr, fidx_ptr, fstart_ptr, fc_ptr,
                               fvroff_ptr, fvcoff_ptr,
                               cv_ptr, pp_ptr, ds_ptr,
                               M, eps, alpha, sign_tau, vtiny,
                               BLOCK: tl.constexpr):
        # Elementwise over factored elements only: write the rank-1 vhat into
        # the shared cand_v segment, then recompute pp/dsign from it so the
        # uniform tail (K2-K5) sees the same values as the eager path.
        # Row/col factor indices are derived from the element position and
        # per-param tables, so no per-element index tensors are needed.
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < M
        pos = tl.load(pos_ptr + offs, mask=mask, other=0)
        j = tl.load(fidx_ptr + offs, mask=mask, other=0)
        s = tl.load(fstart_ptr + j, mask=mask, other=0)
        C = tl.load(fc_ptr + j, mask=mask, other=1)
        vro = tl.load(fvroff_ptr + j, mask=mask, other=0)
        vco = tl.load(fvcoff_ptr + j, mask=mask, other=0)
        rel = pos - s
        vr = tl.load(cvr_ptr + vro + rel // C, mask=mask, other=0.0)
        vc = tl.load(cvc_ptr + vco + rel % C, mask=mask, other=0.0)
        fm = tl.load(fmean_ptr + j, mask=mask, other=1.0)
        vh = (vr / tl.maximum(fm, vtiny)) * vc
        tl.store(cv_ptr + pos, vh, mask=mask)
        pp = tl.sqrt(vh) + eps
        pp = tl.exp(-alpha * tl.log(tl.maximum(pp, 1e-30)))
        pp = tl.minimum(tl.maximum(pp, 1e-8), 1e8)
        tl.store(pp_ptr + pos, pp, mask=mask)
        cm = tl.load(cm_ptr + pos, mask=mask, other=0.0)
        tl.store(ds_ptr + pos, cm / (tl.abs(cm) + sign_tau) * pp, mask=mask)

    @triton.jit
    def _flat_fact_commit_kernel(vr_ptr, vc_ptr, cvr_ptr, cvc_ptr,
                                 ok_ptr, rok_ptr, NR, NC, BLOCK: tl.constexpr):
        # Where the owning segment was accepted, commit candidate row/col
        # factors into the persistent flat_vr/flat_vc state buffers.
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        total = NR + NC
        mask = offs < total
        is_r = offs < NR
        ok = tl.load(ok_ptr + tl.load(rok_ptr + offs, mask=mask, other=0),
                     mask=mask, other=0.0) > 0.5
        rmask = mask & is_r & ok
        cmask = mask & (~is_r) & ok
        cvr = tl.load(cvr_ptr + offs, mask=rmask, other=0.0)
        tl.store(vr_ptr + offs, cvr, mask=rmask)
        cvc = tl.load(cvc_ptr + offs - NR, mask=cmask, other=0.0)
        tl.store(vc_ptr + offs - NR, cvc, mask=cmask)

    @triton.jit
    def _flat_apply_kernel(p_ptr, m_ptr, v_ptr, cm_ptr, cv_ptr, d_ptr,
                           eff_ptr, seg_ptr, ok_ptr, upd_ptr,
                           N, decay_ptr, lr_ptr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        seg = tl.load(seg_ptr + offs, mask=mask, other=0)
        ok = tl.load(ok_ptr + seg, mask=mask, other=0.0)
        accepted = mask & (ok > 0.5)
        p = tl.load(p_ptr + offs, mask=mask, other=0.0)
        d = tl.load(d_ptr + offs, mask=mask, other=0.0)
        eff = tl.load(eff_ptr + seg, mask=mask, other=0.0)
        # Scalar loads so CUDA-graph replays see live per-step values.
        decay = tl.load(decay_ptr)
        lr = tl.load(lr_ptr)
        upd = -lr * eff * d
        p = tl.where(accepted, p * (1.0 - decay) + upd, p)
        tl.store(p_ptr + offs, p, mask=mask)
        tl.store(upd_ptr + offs, tl.where(accepted, upd, 0.0), mask=mask)
        cm = tl.load(cm_ptr + offs, mask=mask, other=0.0)
        cv = tl.load(cv_ptr + offs, mask=mask, other=0.0)
        tl.store(m_ptr + offs, cm, mask=accepted)
        tl.store(v_ptr + offs, cv, mask=accepted)


def flat_fused_supported():
    return triton is not None


def _require_triton():
    if triton is None:
        raise RuntimeError("Triton not available")


def _check_flat(*tensors):
    for t in tensors:
        if not t.is_contiguous():
            raise ValueError("flat fused path requires contiguous buffers")
        if t.dtype != torch.float32:
            raise ValueError("flat fused path requires float32 buffers")


def flat_step_candidates(g, m, v, cand_m, cand_v, pp, dsign, *,
                         b1, b2, eps, alpha, sign_tau):
    """K1 launch: fused elementwise candidate moments + preconditioner.

    Requires contiguous fp32 buffers of identical length on one CUDA device.
    """
    _require_triton()
    _check_flat(g, m, v, cand_m, cand_v, pp, dsign)
    if g.device.type != "cuda":
        raise RuntimeError("flat fused path requires CUDA")
    n = g.numel()
    _flat_cand_kernel[((n + 1023) // 1024,)](
        g, m, v, cand_m, cand_v, pp, dsign, n, b1, b2, eps, alpha, sign_tau,
        BLOCK=1024)


def flat_seg_stats(cand_m, cand_v, p_flat, x, starts, ends, out):
    """K2/K4 launch: per-segment [amax_m, amax_v, amax_p, amax_x, ssq_scaled].

    out[j*5+4] holds the *amax-scaled* sum of squares of cand_m, so the
    stable per-segment norm is amax_m * sqrt(ssq_scaled / seg_len) — no
    overflow for finite moments near the FP32 limit.
    """
    _require_triton()
    _check_flat(cand_m, cand_v, p_flat, x, out)
    if starts.numel() != ends.numel():
        raise ValueError("starts/ends length mismatch")
    if out.numel() < 5 * starts.numel():
        raise ValueError("out buffer too small")
    if starts.dtype not in (torch.int32, torch.int64):
        raise ValueError("starts/ends must be integer tensors")
    _flat_seg_stats_kernel[(starts.numel(),)](
        cand_m, cand_v, p_flat, x, starts, ends, out, BLOCK=1024)


def flat_seg_ssq(x, starts, ends, out):
    """Per-segment sum of squares for an arbitrary flat vector.

    Deterministic (sequential per segment) and NaN-localizing: unlike a
    dense indicator-matrix GEMV, a nonfinite element contaminates only
    the segment it belongs to.
    """
    _require_triton()
    _check_flat(x, out)
    if starts.numel() != ends.numel():
        raise ValueError("starts/ends length mismatch")
    if out.numel() < starts.numel():
        raise ValueError("out buffer too small")
    _flat_seg_ssq_kernel[(starts.numel(),)](x, starts, ends, out, BLOCK=1024)


def flat_direction(cand_m, pp, dsign, rms, seg_idx, d, *, sblend, eps):
    """K3 launch: d = (1-sblend)*dsign + sblend*(cm/(rms+eps))*P.

    `rms` holds the raw per-segment root-mean-square; eps is added inside
    the kernel to match the eager denominator (_rms(m)+eps).
    """
    _require_triton()
    _check_flat(cand_m, pp, dsign, rms, d)
    n = cand_m.numel()
    if not torch.is_tensor(sblend):
        sblend = torch.full((), float(sblend), dtype=torch.float32,
                            device=d.device)
    _flat_dir_kernel[((n + 1023) // 1024,)](
        cand_m, pp, dsign, rms, seg_idx, d, n, sblend, eps, BLOCK=1024)


def flat_apply(p_flat, m, v, cand_m, cand_v, d, eff, seg_idx, ok, upd, *,
               decay, lr):
    """K5 launch: where ok[seg], p = p*(1-decay) - lr*eff*d and m,v <- cand.

    Rejected segments (ok=0) are left completely untouched — no decay, no
    update, no moment commit; upd receives 0 there.
    """
    _require_triton()
    _check_flat(p_flat, m, v, cand_m, cand_v, d, eff, ok, upd)
    n = p_flat.numel()
    if not torch.is_tensor(decay):
        decay = torch.full((), float(decay), dtype=torch.float32,
                           device=p_flat.device)
    if not torch.is_tensor(lr):
        lr = torch.full((), float(lr), dtype=torch.float32,
                        device=p_flat.device)
    _flat_apply_kernel[((n + 1023) // 1024,)](
        p_flat, m, v, cand_m, cand_v, d, eff, seg_idx, ok, upd, n, decay, lr,
        BLOCK=1024)


def flat_fact_moments(g, flat_vr, flat_vc, cand_vr, cand_vc, fmean, meta,
                      cand_m, cand_v, pp, dsign, *,
                      b2, eps, alpha, sign_tau, vtiny):
    """Fused factored preamble: candidate vr/vc via segmented row/col
    reduces, per-param row-factor mean, then rank-1 vhat + pp/dsign
    rewritten in the shared flat buffers. All launches are async.

    `meta` is the dict built once by Tiger._flat_group; see it for the
    layout of the row/col-block/fact-element index tensors.
    """
    _require_triton()
    _check_flat(g, flat_vr, flat_vc, cand_vr, cand_vc, cand_v)
    nr = meta["row_start"].numel()
    if nr:
        _flat_fact_rows_kernel[(nr,)](
            g, flat_vr, cand_vr, meta["row_start"], meta["row_len"],
            b2, BLOCK=1024)
    ncb = meta["cb_start"].numel() if "cb_start" in meta else 0
    if ncb:
        _flat_fact_cols_kernel[(ncb,)](
            g, flat_vc, cand_vc, meta["cb_start"], meta["cb_c0"],
            meta["cb_R"], meta["cb_C"], meta["cb_vc"],
            b2, BLOCK=128)
    if "pb_start" in meta:
        # Tall matrices: two-phase column reduce over (col, row) blocks.
        npb = meta["pb_start"].numel()
        if npb:
            _flat_fact_cols_partial_kernel[(npb,)](
                g, meta["part"], meta["pb_start"], meta["pb_c0"],
                meta["pb_r0"], meta["pb_r1"], meta["pb_C"], meta["pb_R"],
                BLOCK=128)
        ncbf = meta["cbf_vc"].numel()
        if ncbf:
            _flat_fact_cols_finish_kernel[(ncbf,)](
                flat_vc, cand_vc, meta["part"], meta["cbf_vc"],
                meta["cbf_p0"], meta["cbf_np"], meta["cbf_rem"],
                b2, BLOCK=128)
    nfact = fmean.numel()
    if nfact:
        _flat_fact_mean_kernel[(nfact,)](
            cand_vr, meta["fact_vroff"], meta["fact_R"], fmean, BLOCK=1024)
    m = meta["elem_pos"].numel()
    if m:
        _check_flat(pp, dsign)
        _flat_fact_vhat_kernel[((m + 1023) // 1024,)](
            cand_m, cand_vr, cand_vc, fmean,
            meta["elem_pos"], meta["elem_fact"],
            meta["fact_start"], meta["fact_C"],
            meta["fact_vroff"], meta["fact_vcoff"],
            cand_v, pp, dsign, m, eps, alpha, sign_tau, vtiny, BLOCK=1024)


def flat_fact_commit(flat_vr, flat_vc, cand_vr, cand_vc, rok, ok):
    """Commit candidate vr/vc into the persistent flat state buffers for
    segments whose ok flag is set. `rok` maps each row entry then each
    column entry to its fast-param index."""
    _require_triton()
    _check_flat(flat_vr, flat_vc, cand_vr, cand_vc, ok)
    nr = flat_vr.numel(); nc = flat_vc.numel()
    total = nr + nc
    if total:
        _flat_fact_commit_kernel[((total + 1023) // 1024,)](
            flat_vr, flat_vc, cand_vr, cand_vc, ok, rok, nr, nc, BLOCK=1024)
