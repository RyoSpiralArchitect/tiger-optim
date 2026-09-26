# MPS trust EMA guard iteration, 2026-09-27

This experiment starts from [PR #51's scaled-norm change](../2026-09-26-scalar-sync/README.md)
at commit `fe2abf2b583de1b522c847e3f850b6d5e0123289`. It changes only the
tensor-valued old trust EMA finite/nonnegative check in `Tiger.step()`: the
value is viewed as one-dimensional before comparison with a same-shaped zero
tensor. The `all()` result still enters the existing per-parameter finite gate
before moments or parameters are committed. Float-valued old trust follows
the existing path.

The MPS TinyMix fixture (`d=256`, `ff=512`, four heads, seed 0) ran for 29
steps on an Apple M4 with macOS 26.4.1, Python 3.12.6, and PyTorch 2.12.1.
Both source tuples were stable during their runs. The baseline `tiger.py`
SHA-256 is `3836c2121d5bd94b9a70ac2e5299d4f7e9e76578425ed2a6d17c2012bed8269b`;
the patched SHA-256 is `fbc5c0811c6a4bf7f5566c98f12cf1bcba4f946f8479d48afbb7b0f000ac25e7`.
The common `torch_backend.py` SHA-256 is
`6c787b4536666446c7a33342c35b7cada1e80bb1c58d7d875db8b97772b4495e`.
Full source hashes are in `manifest.json` and the raw profile.

## Profiler result

`torch.profiler` recorded **CPU activity** while the model and optimizer
tensors ran on **MPS**. Its marked optimizer steps 24–26 showed:

| Global step | PR #51 `_local_scalar_dense` | Patched `_local_scalar_dense` | `isfinite` both | `fft_rfft` both |
| ---: | ---: | ---: | ---: | ---: |
| 24 | 132 | 106 | 25 | 0 |
| 25 | 132 | 106 | 25 | 2 |
| 26 | 132 | 106 | 25 | 0 |

The reduction is **26 host-side scalar reads per step** for this fixture.
Across the complete three-step traces, the enclosing `ge > item` chain fell
from 30 to 0 and `bitwise_and > item` from 48 to 0. The total trace counts
were 399 versus 321; three reads in each trace are the benchmark's
`loss.item()` outside optimizer markers. Because the traces did not record
Python stacks, attributing these internal MPS reads to the changed guard is
an inference from the narrow source diff, event order, and chain counts.

## Output and timing boundary

The fresh unprofiled patched run and PR #51's archived run had identical loss
at every step and identical QKV scales before and after every step. Both had
zero warnings; `compare.py` regenerates the zero maximum differences in
`comparison.json`. CPU/MPS regression tests additionally cover negative,
NaN, and Inf old trust tensors for scalar and QKV vector states, verifying
that parameter, weight decay, and moment updates are skipped.

| Unprofiled run | Step 24 | Step 25 | Step 26 | Step 25 minus 24 |
| --- | ---: | ---: | ---: | ---: |
| PR #51 post 1 | 48.18 ms | 99.16 ms | 56.44 ms | +50.98 ms |
| Patched post 1 | 22.35 ms | 71.05 ms | 22.32 ms | +48.70 ms |

These are different fresh processes under variable host load. They do not
establish a wall-time speedup. The step-25 spectral spike remains. The
patched profiler's step-25 marker took 89.72 ms with profiling overhead;
that time is not compared with the unprofiled runs. Its sole warning was
PyTorch's profiler event-clearing notice, also present in the baseline trace.

The patched Mac test suite reported **141 passed, 5 skipped**. CUDA was
unavailable. `SHA256SUMS` covers every new evidence file except itself.

## Reproduce

From the repository root in the patched checkout:

```sh
env PYTHONPATH='/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages:src' python3 -S benchmarks/evidence/2026-09-26-scalar-sync/profile_step25.py --variant full --output-dir /tmp/tiger-trust-profile
python3 -S benchmarks/evidence/2026-09-26-scalar-sync/summarize_trace.py /tmp/tiger-trust-profile/trace.json.gz --output /tmp/tiger-trust-profile/trace-summary.json
python3 -S benchmarks/evidence/2026-09-27-trust-guard/compare.py
```

From this evidence directory, run `shasum -a 256 -c SHA256SUMS` to verify
the archived files. `raw/post-profile/trace.json.gz` preserves the full
profile, and the baseline remains in PR #51's evidence directory.
