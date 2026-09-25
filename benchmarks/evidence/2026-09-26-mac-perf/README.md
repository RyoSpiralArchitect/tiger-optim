# Mac CPU/MPS diagnostic, 2026-09-26

This is a bounded diagnostic of the dirty `66374f21a0ddb00a27dd119c7f616e4d5c74eb27` checkout on Apple M4, macOS 26.4.1 arm64, Python 3.12.6, and PyTorch 2.12.1. Each comparison used one fresh Python process, a fixed synthetic model and input, seed 0, 10 warmup steps, and 30 measured steps. The optimizer configurations and full run identities are in the raw comparison JSON files and `manifest.json`. These runs do not establish convergence or optimizer superiority.

## MPS scalar reads

The PyTorch CPU profiler recorded three Tiger steps per trace. Counts below include only events nested inside `Optimizer.step#Tiger.step`; each trace also has three scalar reads outside the optimizer for loss reporting.

| Phase | `tiger.py` SHA-256 | `aten::_local_scalar_dense` / step | `aten::isfinite` / step | `aten::amax` / step |
| --- | --- | ---: | ---: | ---: |
| After QKV aggregation, before finite-gate consolidation | `ea686bc056867d10a8c5c0b67657a0d61937f02659e0c6d92832644afd497c3a` | 68 | 43 | 0 |
| After one candidate + direction host decision and low-precision fixes | `4198249cd4532b6d9831702caf4d85e202c1c205d53ea4ae97ef47d8e8ad8045` | 58 | 43 | 0 |
| After candidate + direction `amax` reduction | `4ef12be8d55948cbda728fcd3b91cf140e8306e74e94d02b1a1df4ee2987ca93` | 30 | 15 | 48 |
| Final source after FP16 parity and FP32 overflow guard | `387600e0d5e0550714bd71edeadce69e4ae66f0230d8532e4539e1f5d185216e` | 36 | 15 | 60 |

The pre-guard `4ef12be8` trace still assigns 54.37% of profiled self CPU to `_local_scalar_dense`. The final-source trace has six more scalar reads per step than that pre-guard trace. These are operation counts; host-load drift prevents a wall-clock speed claim.
The first three traces precede the later FP16 decay-preview parity fix and focused FP32
output-overflow guard. Their source hashes must not be read as the PR's final
source hash or as a performance measurement of those later changes. The new
`profile-final-387600e0/` phase captures the final source's operation counts.

## Paired MPS smoke

Each row used the same benchmark command, seed, shape, warmup, and measured step count. AdamW and Tiger have different optimizer configurations; see the JSON. Times are milliseconds.

| Phase | AdamW full median | AdamW optimizer median | Tiger full median | Tiger optimizer median | Tiger optimizer at global step 25 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial, source hash unavailable | 4.076 | 1.043 | 40.735 | 35.899 | 630.434 |
| Post-QKV, `ea686bc0` | 3.351 | 0.934 | 20.388 | 18.431 | 73.437 |
| Post-consolidation, `4198249c` | 7.244 | 2.299 | 60.099 | 55.618 | 150.180 |
| Post-`amax`, `4ef12be8` | 3.252 | 0.998 | 27.083 | 23.950 | 99.521 |

The simultaneous AdamW shifts show substantial host-load drift. The recurring Tiger global-step-25 spike remains unresolved. The initial CPU pair is archived under `cpu/`; its source hash was also not captured. The `mps-repeat/` run started while the QKV source was being edited, so it is preserved but excluded from phase comparisons.
No paired smoke was run at source hash `387600e0`; the final-source check was the bounded three-step MPS profile above.

## Commands and artifacts

The comparison command, run from each `benchmarks/results/mac-perf-20260926/<phase>/` directory, was:

```sh
env PYTHONPATH='/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages:/Users/ryospiralarchitect/🌀SpiralReality🌀/tiger-optim/src' python3 -S /Users/ryospiralarchitect/🌀SpiralReality🌀/tiger-optim/benchmarks/bench_compare_optim.py --device mps --steps 30 --warmup 10 --seed 0 --modes adamw tiger_v21_full
```

The CPU command changed only `--device mps` to `--device cpu`. The profile command, including its output paths, is recorded separately for each phase in `manifest.json`. The profile used `--steps 8 --warmup 2 --torch-profiler --profile-steps 3`; its timing should not be compared directly with the paired smoke. The raw Chrome traces are stored as reproducibly compressed `trace.json.gz` files, with compressed and uncompressed hashes in the manifest. `SHA256SUMS` covers every tracked evidence file except itself.

The isolated `amax_probe.py`/`amax_probe.json` check confirms on this MPS/PyTorch version that `abs().amax()` propagates NaN and both infinities for FP32, FP16, and BF16. For four candidate tensors over three calls, `_local_scalar_dense` fell from 15 to 6. The production change also skips `amax` on zero-element tensors to preserve the prior vacuous `isfinite(...).all()` result. Focused optimizer tests passed 57 with 2 skipped; the full CPU suite passed 100 with 5 skipped at source hash `4ef12be8`.

All MPS Tiger runs emitted the PyTorch `torch.fft.rfft` output-resize deprecation warning at `tiger.py:247`. This warning and the adaptation spike were not addressed by the finite-check patch.
