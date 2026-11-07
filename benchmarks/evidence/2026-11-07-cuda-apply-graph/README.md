# CUDA flat fused + fully graphed optimizer step (Nov 7, 2026)

## Claim

`TIGER_FLAT_STEP=1` reduces Tiger's optimizer-step latency by roughly
**49–56%** on an RTX 5090 (TinyMix d=256/ff=512 model, fp32, batch 16×32
tokens) across all four Tiger modes, versus the eager path.

Follow-up to the Nov 6 evidence: the post-`par_ok` apply tail (eff/AGC,
density accumulators, trust EMA, decay, K5, factored commit, foreach
updates) is now also CUDA-graph captured (graph B), sharing one graph
pool per device, with an LRU bound on flat-buffer membership.

## Method

- `benchmarks/bench_compare_optim.py --device cuda --steps 150 --warmup 25
  --seed 0` for each mode, once eager and once `TIGER_FLAT_STEP=1`
  (flat). Same process/GPU state per mode pair; warmup included per run.
- `raw/eager/` and `raw/flat/` hold the untouched per-mode JSON outputs.
- Median of `ms_optimizer` (opt.step + sync) is the headline metric.

## Results (ms_optimizer median)

| mode | eager (ms) | flat (ms) | delta |
|---|---|---|---|
| tiger_v21_full | 6.6261 | 2.9427 | −55.6% |
| tiger_v21_no_ffn | 6.6233 | 2.9370 | −55.6% |
| tiger_v21_no_qkv | 6.3781 | 3.2567 | −48.9% |
| tiger_v13_compat | 5.9112 | 3.0502 | −48.4% |
| adamw (control) | 0.1922 | 0.1929 | +0.4% |

## Correctness

- `pytest tests -x -q`: 207 passed both eager and `TIGER_FLAT_STEP=1`.
- Bitwise parity vs eager on v21_full/lean (params and moments), tall
  (8192×64), and NaN-injection isolation runs — all 0.0 diffs.
- Membership-churn run (profiler-driven eligibility flips): param
  checksum identical to unmodified main.
- Very wide factored rows (C > 16384) use a two-phase row reduce; the
  partial fold reorders fp32 accumulation (same class as the col
  two-phase), so the gate stays high.

## Caveats

- Diagnostic `bench_robust.py` numbers are for tuning only; this
  directory + `manifest.json` + `SHA256SUMS` is the claim-gate record.
