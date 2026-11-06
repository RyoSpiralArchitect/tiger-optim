# CUDA flat fused + graph-captured optimizer step (Nov 6, 2026)

## Claim

`TIGER_FLAT_STEP=1` reduces Tiger's optimizer-step latency by roughly
**40–45%** on an RTX 5090 (TinyMix d=256/ff=512 model, fp32, batch 16×32
tokens) across all four Tiger modes, with bitwise-identical parameters
and optimizer state versus the eager path.

## Method

- `benchmarks/bench_compare_optim.py --device cuda --steps 150 --warmup 25 --seed 0`
  run twice on the same idle RTX 5090 (torch 2.13.0+cu132, triton 3.7.1,
  Python 3.14, driver clocks reached ~2940 MHz under load):
  - `raw/eager/` — default path (`TIGER_FLAT_STEP` unset)
  - `raw/flat/` — `TIGER_FLAT_STEP=1`
- The flat path concatenates eligible params into persistent flat
  buffers and runs fused Triton kernels; the pre-guard pipeline is
  additionally captured as a CUDA graph after two eager warmup steps.
  `bench_compare_optim`'s 25 warmup steps cover graph capture; the
  medians are over 150 timed steps.

## Results (optimizer-step medians, ms)

| mode | eager | flat | Δ |
|---|---|---|---|
| tiger_v21_full | 6.536 | 3.689 | −43.6% |
| tiger_v21_no_ffn | 6.542 | 3.680 | −43.7% |
| tiger_v21_no_qkv | 6.281 | 3.877 | −38.3% |
| tiger_v13_compat | 5.806 | 3.575 | −38.4% |
| adamw (control) | 0.191 | 0.193 | +1.2% |

Total-step medians: v21_full 8.44 → 4.66 ms (−44.8%).

## Caveats

- The forward/backward timer also dropped (~1.9 → ~0.96 ms). That is
  almost certainly clock ramp (the eager suite ran first on an idle GPU
  at ~180 MHz); the flat suite ran on a warm GPU. It is **not** claimed
  as an optimizer effect — the claim is on `ms_optimizer_median` only,
  and even there the eager numbers are slightly conservative.
- Results apply to this source revision and environment
  (RTX 5090, fp32 TinyMix). Graph capture adds VRAM for per-membership
  buffers; unusual membership churn grows `_flat_cache`.
- Correctness gate: `pytest tests` passes with and without
  `TIGER_FLAT_STEP=1`; shared-gradient runs on v21_full/lean/tall and a
  NaN-injection run are bitwise identical to eager.

## Files

- `raw/eager/`, `raw/flat/` — bench_compare_optim JSON per mode
- `manifest.json` — run metadata and source/env digests
- `SHA256SUMS` — raw file digests
