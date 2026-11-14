# Matched FP32 spectral replication

Sixteen completed RTX 5090 runs, 25,600 updates: one BF16 reproducibility bridge
plus fifteen FP32 runs (five seeds × full/off/fade). All fifteen FP32 runs pass
the original/rebound >=90%, obsolete-answer <=20% binding gate.

The protocol was frozen at `abcf20a9c7a9534a4d57babe4d201c097485dfff` before execution.
Training uses the previous frozen clean source
`3cb4b78ce5f04837ad14d16305e7f5b505bbbe99`, not the runner checkout. The seed-701
BF16 full bridge exactly reproduces its prior initial/final weights and loss/LR
traces. CUDA, PyTorch, device, TF32 setting and CPU threads match the historical
runs; initialization, inputs, implementation hashes and LR match across precision.
Full/fade sampled weights match through step 800 for every FP32 seed.

## Results

Candidate minus reference; negative CE differences favor the candidate.

| Contrast | BF16 mean test CE difference | FP32 mean | FP32 median | Improved BF16 / FP32 |
| --- | ---: | ---: | ---: | ---: |
| Full minus spectral off | +0.00423956 | -0.00185174 | +0.00007102 | 2/5 / 2/5 |
| Fade minus full | -0.00664800 | -0.00035736 | -0.00007191 | 4/5 / 4/5 |

Fade improves four of these five seeds at both precisions, so this favorable
pattern is not confined to BF16. Its mean improvement is much smaller in FP32,
and the adverse seed changes (701 in BF16; 727 in FP32). Full spectral's mean
changes sign across precision while only two seeds improve in either precision;
the mean alone is not a reliable characterization of a typical seed.

The mean FP32-minus-BF16 change in the full/off contrast is -0.00609130 and in
the fade/full contrast is +0.00629064. Seed 709 contributes a large part of the
latter change. Each seed and both precision contrasts are retained in summary.json.

These five seeds were fixed before the FP32 runs, but were already examined in
BF16. This is a matched precision follow-up, not a new independent BF16 sample.
Earlier adverse FP32 results on seeds 631/641 remain valid and are not discarded.
The evidence does not establish a universally beneficial schedule, a causal
mechanism, or language-model generalization. Optimizer defaults remain unchanged.
No throughput claim is made. FP32 disables autocast and TF32; BF16 uses autocast
with FP32 parameter storage, as recorded by the benchmark.

## Reproduce

Use a clean source checkout at the source commit above, a runner checkout at the
protocol commit, and a new output directory. No environment installation is needed
on the recorded Furnace runtime (Python 3.14.4, Torch 2.13.0+cu132, CUDA 13.2).

```sh
python benchmarks/evidence/2026-11-14-spectral-precision/run_suite.py \
  --source-root /path/to/frozen-source --output-dir /path/to/new-output
```

The runner aborts if the bridge fails and refuses to overwrite an output directory.
Raw gzip receipts decompress to original bytes, hashed in index.json. Reproduce
the derived summary using standard Python:

```sh
python benchmarks/evidence/2026-11-14-spectral-precision/summarize.py
python -m unittest discover -s tests -p 'test_spectral*.py'
```

The verifier checks completeness, hashes, source/config, finite traces, paired
inputs/initialization/LR/environment, historical receipts, bridge identity and
pre-fade identity before reporting results.
