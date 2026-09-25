# CPU held-out quality probe, 2026-09-25

This exploratory probe trains the same 2-64-64-1 SiLU model from the same
initial weights on a noisy 2D regression task. Each optimizer sees the same
precomputed minibatches for 200 updates; evaluation uses 256 held-out,
noise-free examples. The scripts, full traces, stdout, and file hashes are
preserved here. All tensors and parameters were asserted to be on CPU, and
`python3 -S` bypassed host startup that otherwise selects MPS.

The portable replay is `python benchmarks/bench_quality_smoke.py` from the
repository root. It writes a timestamped JSON under the ignored
`benchmarks/results/` directory and records the source hash and run settings.

Final held-out MSE:

| Seed | AdamW, LR 1e-3 | Tiger conservative README settings, LR 2e-4 | Tiger with AGC disabled, LR 2e-4 |
| --- | ---: | ---: | ---: |
| 11 | 0.121787 | 1.503229 | 1.420374 |
| 23 | 0.076303 | 1.686694 | 1.617842 |
| 37 | 0.082212 | 1.261415 | 1.191301 |

On seed 11, the first update's parameter delta L2 was 0.06646 for AdamW,
0.00002006 for Tiger with AGC 0.02, and 0.002008 for Tiger without AGC.
One further sensitivity run on **seed 11 only** yielded Tiger held-out MSE
1.101867 at LR 1e-3 and 0.323873 at LR 3e-3, both without AGC. At fixed LR
1e-2, Tiger learned temporarily but regressed by step 200; LR 3e-2 diverged
sharply on seed 11. A cosine schedule from 1e-2 to 1e-4 was chosen after
reviewing those traces, then tested across all three seeds:

| Seed | Tiger fixed LR 1e-2 | Tiger cosine LR 1e-2 to 1e-4 |
| --- | ---: | ---: |
| 11 | 0.279821 | 0.006240 |
| 23 | 0.661677 | 0.006823 |
| 37 | 0.230837 | 0.002943 |

Every scheduled run remained finite and improved at each recorded checkpoint.
This gives a working recipe for this toy task and supports late regression as
the cause of the fixed-LR result. AdamW used a fixed LR, while Tiger settings
were explored sequentially, so the table does not establish optimizer
superiority.

The optimizers use different settings and the task is small and synthetic.
These results do not establish a general ranking. They do show that the
documented conservative AGC setting can suppress learning on this task; it
should not be presented as a validated quality preset.
