# Contiguous QKV spectral input view, 2026-09-27

This iteration starts from merged `main` commit
`b3ab1e100599d940aa7bbdd73048d9736ee99c9a` (PRs #48–#53 integrated).
When a fused QKV direction is contiguous and split into three equal dim-0
chunks, the spectral helper now views that direction as three rows instead
of copying three chunks into a stack. Other layouts keep the existing path.
The FFT, centering, frequency bands, phase calculation, and QKV adaptation
rules are unchanged. FP32 uses the source storage directly; lower precision
directions still need one FP32 conversion.

The Apple M4 MPS TinyMix fixture (`d=256`, `ff=512`, four heads, seed 0) ran
29 steps on macOS 26.4.1, Python 3.12.6, and PyTorch 2.12.1. The baseline
`tiger.py` SHA-256 is
`6807f7ef007fe250aca43360ccb0354b7a72c8ab2228b405c6d3d48563bc4d60`;
the view-path SHA-256 is
`3d46d4da43b92c7a9914f16f7bcf03b01266d92e8678bbdfc1ac9f81bc679e86`.
Both were stable during each run. The full source tuples are in `manifest.json`.

## Output and profiler evidence

The fresh unprofiled baseline and view-path runs had **identical loss at all
29 steps** and identical QKV scales before and after every step. Both had
zero warnings; `compare.py` regenerates the zero maximum differences. The
Mac suite reported **179 passed, 5 skipped**; CUDA was unavailable. Direct
CPU/MPS tests cover FP32, FP16, BF16, fused weight and bias shapes, NaN/Inf,
and a noncontiguous fallback.

`torch.profiler` recorded **CPU activity** while model and optimizer tensors
ran on **MPS**. At the marked first adaptation step 25, the `stack` event
count fell from 64 to 62. The two FFT calls, 42 `mean` events, and 46
`_local_scalar_dense` events per optimizer step remained. Steps 24 and 26
had 46 `stack` events on both sources. These event counts support the narrow
staging change; they are not GPU kernel timings.
Each profiled trace has PyTorch's event-clearing notice; the unprofiled runs
had no warnings.

## Scoped MPS helper probe

`helper_probe.py` checks exact spectral metric equality and confirms the
FP32 view shares the fused input's storage while the stack allocates separate
storage. It warms both routes, then alternates their order for eight pairs
per shape. The table shows medians in milliseconds from two fresh processes:

| Elements per Q/K/V chunk | Run 1 stack → view | Run 2 stack → view |
| ---: | ---: | ---: |
| 256 | 0.54 → 0.44 | 0.41 → 0.38 |
| 65,536 | 0.82 → 0.95 | 0.86 → 0.74 |
| 262,144 | 1.57 → 1.74 | 1.13 → 1.13 |
| 1,048,576 | 5.41 → 5.12 | 3.64 → 3.26 |
| 4,194,304 | 19.51 → 17.83 | 13.69 → 13.09 |

For the largest shape, the removed stack is an additional **48 MiB** FP32
staging tensor. The large-shape helper medians favored the view in both
runs, while smaller shapes were mixed. This isolated helper result does not
establish an end-to-end optimizer speedup.

The single fresh unprofiled TinyMix run measured step 25 at 84.69 ms for
baseline and 75.48 ms for the view. The profiled markers measured 97.01 ms
and 107.01 ms, respectively. Different-process host load and profiler
overhead prevent a wall-time speed claim. The step-25 spectral spike remains.

## Reproduce

From the repository root, use a checkout at the baseline commit for the
first command and this patched checkout for the remaining commands:

```sh
env PYTHONPATH='/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages:src' python3 -S benchmarks/evidence/2026-09-26-scalar-sync/diagnose_step25.py --variant full --steps 29 --output /tmp/tiger-qkv-view-run.json
env PYTHONPATH='/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages:src' python3 -S benchmarks/evidence/2026-09-26-scalar-sync/profile_step25.py --variant full --output-dir /tmp/tiger-qkv-view-profile
env PYTHONPATH='/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages:src' python3 -S benchmarks/evidence/2026-09-27-qkv-spectral-view/helper_probe.py --output /tmp/tiger-qkv-view-helper.json
python3 -S benchmarks/evidence/2026-09-27-qkv-spectral-view/compare.py
```

Run `shasum -a 256 -c SHA256SUMS` from this evidence directory to verify
every archived file except `SHA256SUMS` itself. Both compressed profiler
traces and all raw helper timings are archived under `raw/`.
