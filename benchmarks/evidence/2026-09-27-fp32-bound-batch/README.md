# MPS FP32 bound batching, 2026-09-27

This iteration starts from [PR #52's trust guard](../2026-09-27-trust-guard/README.md)
at commit `0cd8062dc46d44c2ace817591695e0e7b1916735`. The three ordinary
FP32 update bounds are stacked and checked as a vector against a same-shaped
FP32 limit tensor. The optional norm bounds are stacked in a separate check,
so the existing exact norm fallback still runs when its conservative bound
fails. Both results enter the existing per-parameter finite gate before state
is committed.

The Apple M4 MPS TinyMix fixture (`d=256`, `ff=512`, four heads, seed 0) ran
29 steps on macOS 26.4.1, Python 3.12.6, and PyTorch 2.12.1. The baseline
`tiger.py` SHA-256 is
`fbc5c0811c6a4bf7f5566c98f12cf1bcba4f946f8479d48afbb7b0f000ac25e7`;
the patched SHA-256 is
`e93496b55b3d5b714e24b461d0f094236fc9b72dc8393745d2bc6a54bcb88b0d`.
Both source tuples stayed stable during their runs. Full source hashes and
trace hashes are in `manifest.json`.

## Profiler result

`torch.profiler` recorded **CPU activity** while the model and optimizer
tensors ran on **MPS**. Its marked optimizer steps 24–26 showed:

| Global step | PR #52 `_local_scalar_dense` | Patched `_local_scalar_dense` | `isfinite` both | `fft_rfft` both |
| ---: | ---: | ---: | ---: | ---: |
| 24 | 106 | 46 | 25 | 0 |
| 25 | 106 | 46 | 25 | 2 |
| 26 | 106 | 46 | 25 | 0 |

The reduction is **60 host-side scalar reads per optimizer step** for this
fixture. Across the complete three-step traces, `le > item` fell from 180 to
0; total `_local_scalar_dense` counts fell from 321 to 141. Three reads in
each full trace are the benchmark's `loss.item()` outside optimizer markers.
The traces did not record Python stacks, so attribution to the changed bounds
is an inference from the narrow source diff, event order, and chain counts.

## Correctness and timing boundary

The fresh unprofiled patched run and PR #52's archived run had identical
loss at every step and identical QKV scales before and after every step.
Both had zero warnings. `compare.py` regenerates these zero maximum
differences in `comparison.json`. A CPU/MPS regression test uses the adjacent
FP32 values immediately below, at, and above the bound threshold; it checks
the optimizer's accept/skip decision and verifies the parameter stays intact.
The Mac suite reported **147 passed, 5 skipped**; CUDA was unavailable.

| Unprofiled run | Step 24 | Step 25 | Step 26 | Step 25 minus 24 |
| --- | ---: | ---: | ---: | ---: |
| PR #52 post 1 | 22.35 ms | 71.05 ms | 22.32 ms | +48.70 ms |
| Patched post 1 | 22.07 ms | 70.18 ms | 23.06 ms | +48.11 ms |

These single runs do not establish a wall-time speedup. The step-25 spectral
spike remains. The patched profiler's step-25 marker took 95.81 ms with
profiling overhead; its sole warning was PyTorch's event-clearing notice,
also present in the baseline trace. `SHA256SUMS` covers every new evidence
file except itself.

## Reproduce

From the repository root in the patched checkout:

```sh
env PYTHONPATH='/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages:src' python3 -S benchmarks/evidence/2026-09-26-scalar-sync/diagnose_step25.py --variant full --steps 29 --output /tmp/tiger-bound-post.json
env PYTHONPATH='/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages:src' python3 -S benchmarks/evidence/2026-09-26-scalar-sync/profile_step25.py --variant full --output-dir /tmp/tiger-bound-profile
python3 -S benchmarks/evidence/2026-09-26-scalar-sync/summarize_trace.py /tmp/tiger-bound-profile/trace.json.gz --output /tmp/tiger-bound-profile/trace-summary.json
python3 -S benchmarks/evidence/2026-09-26-scalar-sync/summarize_scalar_chains.py /tmp/tiger-bound-profile/trace.json.gz --output /tmp/tiger-bound-profile/chain-summary.json
python3 -S benchmarks/evidence/2026-09-27-fp32-bound-batch/compare.py
```

From this evidence directory, run `shasum -a 256 -c SHA256SUMS` to verify
the archived files. `raw/post-profile/trace.json.gz` preserves the full
profile, and the baseline remains in PR #52's evidence directory.
