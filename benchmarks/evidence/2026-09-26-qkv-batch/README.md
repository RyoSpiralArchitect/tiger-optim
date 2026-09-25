# QKV batched FFT Mac check, 2026-09-26

This is an isolated follow-up to the separate `2026-09-26-mac-step25`
diagnostic archive.
The implementation batches equal-length Q/K/V spectral inputs from each fused
tensor into one FFT and uses an explicitly sized MPS output. Unequal-length
inputs retain the existing per-chunk path. These synthetic runs do not
establish training-quality or broad speed advantages.

## Source and setup

The baseline was `origin/main` commit
`a4dd249cbfb0d4b1caed4fa9fd292d6cee83de9f` in a detached worktree.
The candidate was the isolated `codex/tiger-batched-fft-20260926` worktree
before commit. Both used identical accel files and benchmark script. SHA-256:

| File | Baseline | Candidate |
| --- | --- | --- |
| `src/tiger_optim/tiger.py` | `387600e0d5e0550714bd71edeadce69e4ae66f0230d8532e4539e1f5d185216e` | `aced6939c263899502a4e6f8b67af5fbada3aee94c566adc63e0f6a53f949db7` |
| `src/tiger_optim/accel/__init__.py` | `e02a830073e35ad5bf824be5aa071dc6bea1f47a94e101f84e4da0e826e28d3a` | same |
| `src/tiger_optim/accel/torch_backend.py` | `e38bf5af473112c73718ce59b2187ba3a779ba74c02d8f3542b78c4050c82bb5` | same |
| `benchmarks/bench_compare_optim.py` | `c24454b51bfc72f76d992f8c31a823a3c535487888b02b6b14bde7dfd7c199d3` | same |

Source hashes were checked before and after each run, and the profiler
summaries independently record their source tuples. The Mac was Apple M4,
macOS 26.4.1, Python 3.12.6, PyTorch 2.12.1. Each comparison used a fresh
`python3 -S` process, seed 0, batch 16, sequence length 32, width 256, 10
warmup steps, and 30 measured steps. Run order was baseline, candidate,
baseline, candidate. The JSON files in `raw/` are the untouched benchmark
outputs, and adjacent `.log` files preserve stdout and warning text. Global
step 25 is measured-series index 14 because of the 10 warmups.

## Bounded result

Optimizer timing includes MPS synchronization. Milliseconds:

| Run | Step 24 | Step 25 | Step 26 | 30-step median |
| --- | ---: | ---: | ---: | ---: |
| Baseline 1 | 18.36 | 68.46 | 14.20 | 13.81 |
| Candidate 1 | 14.10 | 47.76 | 14.84 | 14.20 |
| Baseline 2 | 13.67 | 44.60 | 20.33 | 13.97 |
| Candidate 2 | 14.36 | 39.70 | 14.55 | 14.17 |

The step-25 spike remains. Its reduction in these two pairs is consistent
with fewer spectral operations, but host drift and two repetitions are not
enough for a reliable wall-time claim. The candidate did not improve the
30-step median here. All four runs ended with identical QKV scales
(`q=0.8988407986929419`, `k=0.8`, `v=1.1001564260217453`) and identical
last loss (`1.0066518783569336`) for this seed.

Fresh-process CPU operator traces around steps 24–26 directly establish the
operation reduction. Both traces have zero FFT calls at steps 24 and 26.

| Step 25 | Baseline | Candidate |
| --- | ---: | ---: |
| `aten::fft_rfft` calls | 6 | 2 |
| FFT output-resize warnings | 6 | 0 |

The compressed Chrome traces and `summary.json` files are under
`raw/profile-baseline/` and `raw/profile-batched/`. Their summaries include
the uncompressed trace SHA-256. The profiler changes timing and is used here
for operation counts only.

The isolated branch's full suite passed `109 passed, 5 skipped`. Two warnings
in that test run came from the untouched baseline single-chunk MPS helper
used as a correctness oracle; the separate finite-guard branch already has
its sized-output fix. Direct helper tests compare contiguous and noncontiguous
equal chunks, unequal-length fallback, and QKV adaptation scale parity.

## Reproduction and integrity

From each source worktree, the four comparison processes used:

```sh
env PYTHONPATH='/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages:src' python3 -S benchmarks/bench_compare_optim.py --device mps --steps 30 --warmup 10 --seed 0 --modes tiger_v21_full
```

`profile_fft_counts.py` accepts `--source-root` and `--output-dir` for a
fresh-process trace of either worktree. Exact commands, source hashes, raw
artifact hashes, and run IDs are in `manifest.json`. From this directory,
`shasum -a 256 -c SHA256SUMS` verifies every file except the checksum file.
