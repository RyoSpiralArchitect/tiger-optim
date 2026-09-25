# Mac MPS QKV step-25 diagnostic, 2026-09-26

This is a bounded diagnosis of the Tiger global-step-25 optimizer latency and
`torch.fft.rfft` output-resize warning. It is a synthetic TinyMix benchmark on
Apple M4, macOS 26.4.1, Python 3.12.6, PyTorch 2.12.1. It does not establish a
training-quality or wall-time advantage. All runs used MPS explicitly and
`python3 -S` to avoid the machine's site customization.

## Causal ablations before the scaled accel-norm change

`diagnose_step25.py` ran 29 global steps per fresh process, seed 0, with the
same model and inputs as `bench_compare_optim.py`. The only setting changed
between variants was the named QKV default. Optimizer times include MPS
synchronization. The run JSON files in `raw/` contain every step, warnings,
QKV scales, and before/after source hashes. All six runs report stable
`tiger.py` hash
`387600e0d5e0550714bd71edeadce69e4ae66f0230d8532e4539e1f5d185216e`
and benchmark hash
`c24454b51bfc72f76d992f8c31a823a3c535487888b02b6b14bde7dfd7c199d3`.
These runs preceded a concurrent scaled accel-norm edit. The corresponding
`origin/main` accel blobs were
`e02a830073e35ad5bf824be5aa071dc6bea1f47a94e101f84e4da0e826e28d3a`
(`accel/__init__.py`) and
`e38bf5af473112c73718ce59b2187ba3a779ba74c02d8f3542b78c4050c82bb5`
(`accel/torch_backend.py`), inferred from Git rather than captured in each
run. Do not interpret these times as measurements of the later source.

| Variant | Step 24 ms | Step 25 ms | Step 26 ms | QKV scale changes | FFT resize warnings |
| --- | ---: | ---: | ---: | --- | ---: |
| Full, run 1 | 13.99 | 66.58 | 19.18 | 25 | 6 |
| Full, run 2 | 14.25 | 50.13 | 15.26 | 25 | 6 |
| No QKV adaptation | 14.32 | 13.82 | 13.70 | none | 0 |
| Adaptation at step 1000 | 14.15 | 14.12 | 14.01 | none | 0 |
| No QKV spectral processing | 14.40 | 19.77 | 18.52 | 25 | 0 |
| Adaptation every 7 steps | 14.15 | 31.33 | 24.41 | 7, 14, 21, 28 | 24 total |

The first adaptation is the large repeatable step-25 event in the full runs.
Moving the interval to 7 moves the first large event to step 7 (58.84 ms),
and later due steps are near 18 ms. Its unrelated step 25 was 31.33 ms in
this one run, so host-load noise remains visible. Disabling only the spectral
work leaves adaptation at step 25 but removes the resize warnings and most of
the local spike. Its later steps also drift upward in the raw run, so the
19.77 ms value is not a stable isolated cost estimate.

## Operator traces on the later scaled accel-norm source

`profile_step25.py` profiled only global steps 24–26 in a fresh process.
Both trace runs captured a stable source tuple: `tiger.py`
`1a976375fd2f1a0f3c309fef3ce5a8c39624314e300b7a91f6dc350530cfdd77`,
`accel/__init__.py`
`c4e5b868eed28aaaa91f335e6a5535f247880dda271d255e5151f7a62e851813`,
`accel/torch_backend.py`
`937155d30df12dd4ef1d780c218a738a36060b6ebb95e3232ba53abfd3522da0`.
The CPU profiler adds substantial overhead, so its elapsed times are not
compared with the unprofiled table. The compressed Chrome traces, CPU tables,
run summaries, and per-step event counts are under `raw/profile-*/`.

| Variant | Step 24 `fft_rfft` | Step 25 `fft_rfft` | Step 26 `fft_rfft` | Step-25 resize warnings |
| --- | ---: | ---: | ---: | ---: |
| Full | 0 | 6 | 0 | 6 |
| No QKV spectral processing | 0 | 0 | 0 | 0 |

The six transforms are Q/K/V from the fused weight and bias. The profiler
also emitted one unrelated warning about clearing events per cycle in both
runs. `summarize_trace.py` derives the counts from the archived Chrome traces.
Each trace summary includes the hash of its uncompressed trace.

## Integrated snapshot before the final finite-guard correction

After the equal-length QKV batching and finite/accel changes were integrated,
one fresh-process 29-step full diagnostic and one 24–26-step profile were run.
The source tuple was stable before and after each run: `tiger.py`
`0eb49c9e3aba2bb79c86fb0733f9c00d13e5732ee6882cf10c3955d476062f1f`,
`accel/__init__.py`
`e94689fb383a4e342795b0db111ffbb08e520ddeb3a2997ebcff4f63a1526ef5`,
`accel/torch_backend.py`
`85087acab0a5b06dc11d57f99c7509d91d0d6854a83837f4b49624ad12ae169b`,
and benchmark script `c24454b51bfc72f76d992f8c31a823a3c535487888b02b6b14bde7dfd7c199d3`.
The unprofiled optimizer took 35.72, 68.59, and 36.00 ms at global steps
24, 25, and 26. No FFT resize warnings occurred. The profiler trace has
`aten::fft_rfft` counts 0, 2, 0 at those steps, confirming the batch route;
its one warning concerns profiler event clearing. The step-25 spike remains.
These preserved pre-fix files are `raw/integrated-pre-fix-0eb49c9e.json` and
`raw/profile-integrated-pre-fix-0eb49c9e/`. The profiler
timing is overhead-inflated and should not be compared to the unprofiled run.

## Second integrated snapshot before final edge-case fixes

The subsequent finite-guard correction changed `tiger.py` to
`92cde1bfc4e7da9d3af2789186b2f9794bca512014f98f27c64149e270d869ed`.
The accel and benchmark hashes remained
`e94689fb383a4e342795b0db111ffbb08e520ddeb3a2997ebcff4f63a1526ef5`,
`85087acab0a5b06dc11d57f99c7509d91d0d6854a83837f4b49624ad12ae169b`,
and `c24454b51bfc72f76d992f8c31a823a3c535487888b02b6b14bde7dfd7c199d3`,
respectively. The source tuple was stable before and after both new runs.
The unprofiled optimizer took 34.63, 60.22, and 35.19 ms at steps 24, 25,
and 26. There were no FFT resize warnings. The profiler trace again has
`aten::fft_rfft` counts 0, 2, 0; its only warning concerns profiler event
clearing. The step-25 spike remains. The raw files are
`raw/pre-final-92cde1bf.json` and `raw/profile-pre-final-92cde1bf/`.
Later edge-case fixes changed the optimizer again, so this is not the final
source snapshot.
Do not compare the two integrated snapshots as an A/B speed experiment:
the source changed for correctness and host load varies.

## Final integrated source check

After the remaining sparse-trust and empty-QKV edge fixes, `tiger.py` had
SHA-256 `3836c2121d5bd94b9a70ac2e5299d4f7e9e76578425ed2a6d17c2012bed8269b`.
The accel and benchmark hashes stayed at
`e94689fb383a4e342795b0db111ffbb08e520ddeb3a2997ebcff4f63a1526ef5`,
`85087acab0a5b06dc11d57f99c7509d91d0d6854a83837f4b49624ad12ae169b`,
and `c24454b51bfc72f76d992f8c31a823a3c535487888b02b6b14bde7dfd7c199d3`.
All four files were checked before and after the 29-step diagnostic; the
profile independently recorded the same stable tuple. The unprofiled
optimizer took 36.17, 77.20, and 36.60 ms at steps 24, 25, and 26.
There were no FFT resize warnings. The step-25 profiler trace has two
`aten::fft_rfft` calls and none at steps 24 or 26; its only warning concerns
profiler event clearing. The spike remains. See `raw/final-3836c212.json`
and `raw/profile-final-3836c212/`. No comparison between historical
snapshots is a controlled speed test of the correctness fixes.

## Isolated FFT probes and candidate

`fft_probe.py` reproduces the resize warning with plain MPS
`torch.fft.rfft` at lengths 256 and 65,536. An explicitly sized complex64
`out` buffer emits no warning and gives exactly the same MPS output in this
probe (maximum absolute difference 0); CPU `rfft` emits no warning.

`fft_batch_probe.py` models one fused weight `(768, 256)` and bias `(768,)`.
It compares the existing three per-chunk transforms to one `[3, N]` transform
per tensor, with per-row mean subtraction and the same energy/phase formulas.
Both use sized output buffers. With five warmups and 20 measured repetitions:

| Tensor | Three sequential median | One batched median | Max relative metric difference |
| --- | ---: | ---: | ---: |
| Weight | 1.040 ms | 0.522 ms | 1.19e-7 |
| Bias | 0.763 ms | 0.301 ms | 1.26e-7 |

These are isolated warm operation timings. The implementation now batches
equal-length QKV chunks and retains the per-chunk fallback for other layouts.
The tiny FP32 metric differences show numerical agreement within tolerance,
not bitwise identity. The separate
[full optimizer A/B archive](../2026-09-26-qkv-batch/README.md) compares the
isolated candidate to its baseline; it found no reliable wall-time advantage.

## Reproduction and integrity

From the repository root, use the local Python 3.12 site-packages and `src` in
`PYTHONPATH`, then invoke each script with `python3 -S`. For example:

```sh
env PYTHONPATH='/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages:src' python3 -S benchmarks/evidence/2026-09-26-mac-step25/diagnose_step25.py --variant full --steps 29 --output /tmp/tiger-step25-full.json
```

The other script commands and arguments are in `manifest.json`. `SHA256SUMS`
covers every file in this evidence directory except itself. Verify with
`shasum -a 256 -c SHA256SUMS` from this directory. Evidence was collected at
multiple exact source hashes while the follow-up branch was being developed;
do not compare the phases as a speed benchmark.
