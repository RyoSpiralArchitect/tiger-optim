# CUDA flat fused step: two-phase segmented reduces (Nov 7, 2026)

## Claim

`TIGER_FLAT_STEP=1` reduces Tiger's optimizer-step latency by roughly
**70–72%** on an RTX 5090 (TinyMix d=256/ff=512 model, fp32, batch 16×32
tokens) across all four Tiger modes, versus the eager path.

Follow-up to the same-day apply-graph evidence: the per-segment reduce
kernels (`_flat_seg_stats`, `_flat_seg_ssq`) — ~32% of flat-path GPU
time — now run two-phase (per-1024-element chunk partials + ordered
fold, bitwise-identical to the sequential loop) for segments longer
than `TIGER_SEG2P_MIN` (default 8192). On the host side, `par_ok` moves
via an async pinned copy + CUDA event and static per-membership uploads
are signature-skipped, cutting CPU enqueue time.

## Method

- `benchmarks/bench_compare_optim.py --device cuda --steps 150 --warmup 25
  --seed 0` for each mode, once eager and once `TIGER_FLAT_STEP=1`
  (flat). Same process/GPU state per mode pair; warmup included per run.
- `raw/eager/` and `raw/flat/` hold the untouched per-mode JSON outputs.
- Median of `ms_optimizer` (opt.step + sync) is the headline metric.

## Results (ms_optimizer median)

| mode | eager (ms) | flat (ms) | delta |
|---|---|---|---|
| tiger_v21_full | 6.5237 | 1.9633 | −69.9% |
| tiger_v21_no_ffn | 6.5130 | 1.9564 | −70.0% |
| tiger_v21_no_qkv | 6.2731 | 1.7578 | −72.0% |
| tiger_v13_compat | 5.7786 | 1.6195 | −72.0% |
| adamw (control) | 0.1919 | 0.1867 | −2.7% |

## Correctness

- `pytest tests -x -q`: 207 passed with `TIGER_FLAT_STEP=1`.
- Bitwise parity vs eager on v21_full/lean (params and moments), tall
  (8192×64), and NaN-injection isolation runs — all 0.0 diffs.
- Two-phase seg reduce folds chunk partials in the identical order of
  the sequential loop, so ssq stays bitwise; the stats kernel computes
  inv = 1/am_m after the amax fold so (cm·inv)² rounds identically.
- Wide-row two-phase (C > 16384) unchanged: ~1e-8 fp32 reassociation
  noise, same class as before.
- A pinned-buffer H2D pack for per-step meta uploads was implemented,
  measured slower than signature-skipping on this model, and reverted.

## Caveats

- Diagnostic `bench_robust.py` numbers are for tuning only; this
  directory + `manifest.json` + `SHA256SUMS` is the claim-gate record.
