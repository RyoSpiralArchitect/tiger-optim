# Mac MPS QKV step-25 follow-up, 2026-09-26

This is a bounded follow-up to the [original step-25 diagnostic](../2026-09-26-mac-step25/README.md),
on the merged PR #47 source (`origin/main` `98d01260eebe57de9b388f7d21b5625e969e276f`).
It evaluates the residual first QKV spectral adaptation latency. No optimizer
implementation change is proposed by this archive. The model is synthetic
TinyMix, seed 0, Apple M4, macOS 26.4.1, Python 3.12.6, PyTorch 2.12.1.
All 29-step runs used explicit MPS and `python3 -S`. Timings include MPS
synchronization. `tiger.py` SHA-256 was
`3836c2121d5bd94b9a70ac2e5299d4f7e9e76578425ed2a6d17c2012bed8269b`
before and after every optimizer run; the benchmark script SHA-256 was
`c24454b51bfc72f76d992f8c31a823a3c535487888b02b6b14bde7dfd7c199d3`.

## Final-source spectral ablation

Four fresh processes alternated full and no spectral processing. The raw JSON
contains every step, source hashes, QKV scales, loss, and warnings. Values in
milliseconds:

| Run | Step 24 | Step 25 | Step 26 | Step 25 minus 24 |
| --- | ---: | ---: | ---: | ---: |
| Full 1 | 118.22 | 174.88 | 93.13 | +56.66 |
| No spectral 1 | 76.42 | 75.45 | 85.76 | -0.97 |
| Full 2 | 49.75 | 109.02 | 105.07 | +59.27 |
| No spectral 2 | 140.00 | 140.49 | 83.69 | +0.49 |

The spectral path remains the cause of the local step-25 spike in these runs.
Host load varied widely, even between ordinary steps, so cross-process
absolute times are not a controlled speed comparison. There were no warnings.
The last loss was identical (`1.0097911357879639`) in all four runs. Turning
off spectral processing changed the final QKV scale by about `2.28e-8` for Q
and less for K and V in this seed; this single loss does not establish
training-quality equivalence.

## Device and reduction probes

`spectral_device_probe.py` evaluates the existing helper on one fused
`(768, 256)` weight and `(768,)` bias, using either MPS or a diagnostic CPU
round-trip. The two fresh processes reversed the call order, with six calls
per route and shape. Cold weight MPS calls took 35.28/35.25 ms, versus
7.94/6.23 ms for the CPU round-trip. Warm medians were 1.20/1.88 ms on MPS
and 4.07/4.01 ms through CPU. Cold bias MPS calls took 16.68/20.81 ms,
versus 0.93/0.98 ms through CPU. The maximum CPU/MPS metric difference was
`4.84e-7` relative. These isolated calls suggest a cold MPS kernel cost and
a steady-state penalty for CPU transfer; they do not establish optimizer
speed or numerical parity for all inputs.

`reduction_probe.py` compared row `mean` with `sum / length` on eight QKV
spectral shapes, reversing order in fresh processes. First-call totals
shifted strongly with order (mean 194.68/27.09 ms, sum 74.34/43.60 ms),
while warm medians slightly favored `mean` (0.56/0.44 ms versus 0.70/0.99 ms).
This is insufficient evidence to replace `mean`, and `sum / length` can
overflow for finite large inputs where `mean` may remain finite.

## CPU offload diagnostic

`cpu_offload_ablation.py` monkeypatches only the QKV spectral helper. It moves
equal-length chunks to CPU, runs the same helper, and returns the metrics to
MPS; the repository optimizer source remains unchanged. Two more alternating
full/offload pairs used fresh 29-step processes:

| Run | Step 24 | Step 25 | Step 26 | Step 25 minus 24 |
| --- | ---: | ---: | ---: | ---: |
| Full 3 | 85.37 | 180.05 | 97.41 | +94.67 |
| CPU offload 1 | 148.32 | 124.87 | 95.32 | -23.45 |
| Full 4 | 101.05 | 141.59 | 88.05 | +40.54 |
| CPU offload 2 | 77.33 | 101.08 | 85.42 | +23.75 |

The whole-run timing is too noisy for a wall-time speed claim. The final loss
was again identical, and offload changed Q's final scale relative to full by
about `1.7e-12` in this seed. Both routes had zero warnings. Offload reduces
the isolated cold spectral cost but makes each later large-weight spectral
call slower in the helper probe. It also transfers tensors and may scale
poorly for larger models. A default CPU route is therefore not justified.

## Reproduce and verify

From the repository root with the local Python 3.12 site-packages:

```sh
env PYTHONPATH='/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages:src' python3 -S benchmarks/evidence/2026-09-26-mac-step25/diagnose_step25.py --variant full --steps 29 --output /tmp/tiger-step25-full.json
env PYTHONPATH='/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages:src' python3 -S benchmarks/evidence/2026-09-26-step25-followup/cpu_offload_ablation.py --mode cpu_offload --output /tmp/tiger-step25-cpu-offload.json
```

Run the two probe scripts with `--first mps|cpu` and `--first mean|sum`,
respectively, plus `--output`. `manifest.json` records the exact run order,
source hashes, and raw-file hashes. From this directory, run
`shasum -a 256 -c SHA256SUMS` to verify the archive.
