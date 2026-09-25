# MPS scalar host-read diagnosis, 2026-09-26

TinyMix (`d=256`, `ff=512`, four heads, seed 0) ran with Tiger's full
configuration and `skip_if_nonfinite=True`. Model and optimizer tensors were
on **MPS**. `torch.profiler` recorded **CPU activity** during optimizer global
steps 24–26, so the counts below are host-side PyTorch operator events, not
CPU-tensor training results. PyTorch was 2.12.1 on macOS 26.4.1 arm64.

The baseline profile was archived in the preceding final-source diagnosis.
`raw/base/` copies its immutable summary and compressed trace. The patched
profile was captured in a fresh process by `profile_step25.py`; both reported
stable source hashes before and after the run.

| SHA-256 source | Baseline | Patched |
| --- | --- | --- |
| `src/tiger_optim/tiger.py` | `3836c2121d5bd94b9a70ac2e5299d4f7e9e76578425ed2a6d17c2012bed8269b` | same |
| `src/tiger_optim/accel/__init__.py` | `e94689fb383a4e342795b0db111ffbb08e520ddeb3a2997ebcff4f63a1526ef5` | same |
| `src/tiger_optim/accel/torch_backend.py` | `85087acab0a5b06dc11d57f99c7509d91d0d6854a83837f4b49624ad12ae169b` | `6c787b4536666446c7a33342c35b7cada1e80bb1c58d7d875db8b97772b4495e` |
| `benchmarks/bench_compare_optim.py` | `c24454b51bfc72f76d992f8c31a823a3c535487888b02b6b14bde7dfd7c199d3` | same |

| Global step | Baseline `_local_scalar_dense` | Patched `_local_scalar_dense` | Baseline `isfinite` | Patched `isfinite` | `fft_rfft` both |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 24 | 244 | 132 | 53 | 25 | 0 |
| 25 | 244 | 132 | 53 | 25 | 2 |
| 26 | 244 | 132 | 53 | 25 | 0 |

The drop is 112 host reads per step in this profile. The baseline/post
compressed traces and `summarize_scalar_chains.py` attribute all 336 removed
reads across the three steps to operators used by the scaled norm guard: 168 fewer reads inside
`bitwise_and`, 84 inside `gt`, and 84 inside `isfinite`/`ne`. Because only that
guard changed, attributing the reduction to it is an inference from the source
diff and matching operator counts. The guard now
keeps its condition one-dimensional, replaces nonfinite scale values with
zero only for the validity test, and reshapes the final norm back to a scalar.
It still uses scale 1 for zero/Inf/NaN input and retains the original finite
scale otherwise. This leaves Tiger's per-parameter finite gate and its
single commit decision untouched.

`attribute_scalar_reads.py` used `TorchDispatchMode` on the same steps. Its
Python-visible `_local_scalar_dense` count is 15 per step on both sources:
five group finite probes and ten per-parameter commit decisions. This mode
does not see the additional reads made inside MPS operator implementations;
the CPU activity traces above do.

Four fresh-process, **unprofiled** 29-step runs used `diagnose_step25.py` in
base/post/post/base order. All 29 loss values and all QKV scales matched
exactly across runs, with no warnings. The observed step 24/25/26 optimizer
times in milliseconds were:

| Run | 24 | 25 | 26 |
| --- | ---: | ---: | ---: |
| base 1 | 83.57 | 140.19 | 90.84 |
| post 1 | 48.18 | 99.16 | 56.44 |
| post 2 | 48.62 | 132.76 | 57.71 |
| base 2 | 49.77 | 137.03 | 74.60 |

The baseline drifted from 83.57 to 49.77 ms at step 24, and the step-25
spike remained in every run. These timings do not establish a stable speedup.
The profiler also adds substantial overhead, so its elapsed times are not
compared with the unprofiled runs. A profiler event-clearing warning is present
in both traces and is unrelated to Tiger's FFT.

Focused verification on the patched source:
`env PYTHONPATH='/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages:src' python3 -S -m pytest -q tests/test_accel.py tests/test_optimizer_contract.py`
reported **75 passed**. The added reduction test covers CPU float64 and MPS
float32 zero, tiny finite, large finite, Inf, NaN, scalar output shape, and
zero gradients. CPU float32 is also exercised. CUDA was unavailable here.

To reproduce the counts, run `profile_step25.py --variant full --output-dir
<directory>` from the repository root with that Python environment, then
run `summarize_trace.py <directory>/trace.json.gz --output <summary.json>`.
`summarize_scalar_chains.py` produces the enclosing operator breakdown.
`SHA256SUMS` covers every archived file, including both compressed traces.
