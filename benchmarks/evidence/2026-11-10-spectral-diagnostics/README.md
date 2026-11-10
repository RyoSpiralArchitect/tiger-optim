# Spectral trajectory and seed influence diagnostics

This is a post-hoc descriptive analysis of the frozen October 2 CUDA experiment,
not a new training run. Original receipts are unchanged. The tool checks raw
checksums and paired initialization, data, model, source and validation schedules.

Reproduce from the repository root (Python standard library only):

```sh
python benchmarks/diagnose_spectral.py
```

Negative test CE differences favor the first named mode. BF16 has seven seeds;
FP32 has two, which is too small to characterize seed variability reliably.

| BF16 contrast | Mean CE difference | Median | Mean excluding seed 683 |
| --- | ---: | ---: | ---: |
| Full minus spectral off | +0.00270580 | -0.00006255 | -0.00100017 |
| Fade minus full | -0.00225042 | +0.00003319 | +0.00151647 |

Both BF16 mean signs reverse when seed 683 is excluded. Exclusion is an
influence diagnostic, not justification to discard that run. All seeds remain
in the primary result. Leave-one-out values are not confidence intervals.

Every paired contrast reaches sustained sampled validation accuracy of at least
90% at the same observation: step 500, except BF16 seed 677 at step 600.
Validation is sampled every 100 steps; this does not establish identical exact
crossing times or learning trajectories. Fade begins after step 800, so its
shared early attainment is expected. It does not demonstrate faster acquisition.

These results motivate investigating late-training loss changes with new seeds
and a predeclared protocol. They do not identify a causal mechanism or justify a
new default spectral schedule. `summary.json` retains each paired seed result,
all leave-one-out means, and the source archive manifest hash.
