# Fresh-seed late spectral replication

Fifteen completed CUDA BF16 runs, 24,000 updates, five new paired seeds
701/709/719/727/733. All 15 pass original/rebound binding accuracy >=90% and
obsolete-answer accuracy <=20%. Initial weights, data, implementation and LR
traces match within pairs. Full/fade sampled model hashes agree through step 800.

The protocol and runner were committed before execution at
`3cb4b78ce5f04837ad14d16305e7f5b505bbbe99`. The task and LR are inherited from
the October development selection, with validation every 50 updates. Native
optimizer execution uses `TIGER_FLAT_STEP=0`. No seed was replaced or excluded.

| Candidate minus reference | Mean test CE difference | Median | Improved seeds | Mean change in validation CE difference, step 800 to 1600 |
| --- | ---: | ---: | ---: | ---: |
| Full minus spectral off | +0.00423956 | +0.00227434 | 2/5 | -0.00215419 |
| Fade minus full | -0.00664800 | -0.00293238 | 4/5 | -0.00340832 |

Negative differences favor the candidate. Spectral full has worse mean final
test CE again, but its mean validation disadvantage *shrinks* from 800 to 1600.
Only seed 709 shows widening full-minus-off validation loss over that interval.
The proposed broad explanation of late widening is therefore unsupported by
this replication. Final test loss and validation interval changes measure
different things and should not be conflated.

Fade improves final test CE for 4/5 seeds here; seed 709 supplies the largest
improvement (-0.02570744). This result does not override the prior mixed BF16
and adverse FP32 fade results, establish a reliable schedule, or change defaults.
This is a small synthetic retrieval task with five BF16 seeds, no new FP32 runs,
no causal isolation of update magnitude, and no throughput claim. Current code
and runtime need not be equivalent to the historical experiment.

Protocol note: `plan.json` retains an inherited historical `scope` sentence
mentioning seven primary and two FP32 seeds. The executable `precisions`,
`controls`, and this report specify the actual five BF16 seeds; no FP32 runs
were planned or executed. The frozen plan bytes have been preserved.

## Reproduce

Inspect `plan.json`, then use a clean checkout of the source commit and a
compatible CUDA/PyTorch runtime:

```sh
python benchmarks/evidence/2026-11-10-spectral-late/run_suite.py --output-dir /path/to/new-output
```

The runner refuses to overwrite an existing index. It records each completed
run's original JSON SHA256. Archived `raw/*.json.gz` files decompress to those
exact original bytes. Environment details are in each receipt and `summary.json`.
To verify and reproduce the derived report with standard Python:

```sh
python benchmarks/evidence/2026-11-10-spectral-late/summarize.py
python -m unittest discover -s tests -p test_spectral_late.py
```

The summary validates completeness, frozen protocol and raw hashes, clean source
identity, finite losses, paired inputs/weights/source/LR, and full/fade pre-fade
identity. It retains every seed's complete sampled validation difference trace.
