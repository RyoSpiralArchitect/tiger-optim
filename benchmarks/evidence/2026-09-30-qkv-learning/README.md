# QKV feature learning and group-control audit, 2026-09-30

These are 33 finite, 100-update runs of the period-3 causal Transformer task
introduced in the [earlier learning probe](../2026-09-27-toy-transformer-learning/README.md).
Each paired seed shares initial weights, training/held-out data, minibatches,
the Tiger recipe, and the tail cosine schedule. All runs completed on a Mac
with PyTorch 2.12.1. Raw results include source/input hashes and complete traces.
Host load was not controlled; the recorded timings do not establish speedups.

## Feature exploration and confirmation

The ordinary 25-update QKV interval makes three adaptation decisions available
to subsequent updates in a 100-update run. We compared four configurations on
CPU seeds 23/37/41, holding the schedule and all other settings fixed:

| Label | QKV gain | Interval | Spectral |
| --- | ---: | ---: | --- |
| gain-zero | 0 | 25 | off |
| baseline | 0.02 | 25 | off |
| fast | 0.2 | 5 | off |
| fast-spectral | 0.2 | 5 | on |

`gain-zero` retains adaptation bookkeeping but keeps the Q/K/V multipliers
exactly at 0.9/0.8/1.1. It is a learning control, not a timing ablation.

Held-out cross-entropy after update 100:

| Development seed | gain-zero | baseline | fast | fast-spectral |
| ---: | ---: | ---: | ---: | ---: |
| 23 | 0.000480 | 0.000480 | 0.000482 | 0.000482 |
| 37 | 0.000599 | 0.000614 | 0.000490 | 0.000490 |
| 41 | 0.003303 | 0.003376 | 0.002886 | 0.003434 |

More frequent adaptation with a larger gain visibly changed the QKV scales
and improved two development seeds. The fixed `fast` candidate and its
spectral control were then compared against gain-zero on new seeds 61/67/71:

| Confirmation seed | gain-zero | fast | fast-spectral |
| ---: | ---: | ---: | ---: |
| 61 | 0.000474 | 0.000460 | 0.000460 |
| 67 | 0.000776 | 0.000777 | 0.000776 |
| 71 | 0.004061 | 0.004262 | 0.004257 |

The fast candidate improved only one of three confirmation seeds. All three
configurations reached 100% held-out accuracy there, so these small CE changes
also occur near task saturation. The results do not support promoting the
fast configuration to a default. Spectral adaptation affected the trajectories
but did not show a consistent advantage across the development and confirmation
sets. This is one synthetic task, not a general optimizer ranking.

## A concrete control defect and its correction

While tracing the controls, we found that explicit `qkv_lr_scales` bypassed the
group's `lr_scale` in the fused QKV parameter update. Setting the group scale to
0.5 therefore produced exactly the same CPU evaluation traces as the archived
unscaled runs. Setting it to zero still changed the parameters in a direct
regression test.

[PR #57](https://github.com/RyoSpiralArchitect/tiger-optim/pull/57) composes the
group and slice multipliers and aligns the overflow previews with that formula.
We compared the corrected 0.5 group scale against a reference that directly
halves the QKV group's base LR on the old implementation:

| CPU seed | Old group scale 0.5 (ineffective) | Direct QKV LR ×0.5 | Corrected group scale 0.5 |
| ---: | ---: | ---: | ---: |
| 43 | 0.001268 | 0.004551 | 0.004551 |
| 47 | 0.000318 | 0.001728 | 0.001728 |
| 53 | 0.000451 | 0.002425 | 0.002425 |

The corrected and reference runs have identical full evaluation traces and
final model SHA-256 values on all three CPU seeds. A candidate run at the
default group scale 1 also reproduced the archived seed-43 evaluation trace.
MPS seed 43 reached CE 0.004550631 in the reference and 0.004550635 after the
correction; the maximum held-out CE difference across evaluations was 1.08e-6,
with identical accuracies. MPS results are not claimed bitwise equal.

Halving the QKV LR worsened CE here. This experiment checks that the requested
control works; it is not a recommendation to halve that LR. The core fix has
separate tests for staged zero scaling, missing slice multipliers, CPU/MPS
FP32/FP16/BF16 updates, and overflow rejection before moment state is committed.

## Verify and reproduce

From this directory:

```sh
python3 -S compare.py
shasum -a 256 -c SHA256SUMS
```

`compare.py` validates the raw bytes, driver hashes, pinned source hashes,
paired input hashes, complete runs, and CPU trace/model equality. It reads the
earlier September 27 evidence to verify the ineffective old control and the
unchanged default. `comparison.json` is regenerated from the raw records.
The `.json.gz` files preserve the original JSON bytes with deterministic gzip
headers. `manifest.json` lists the decompressed hashes and selection history.

Baseline source: `50f1f5c` (main after PRs #55/#56). Corrected source:
`bbd3d39` (PR #57). Both full commit IDs and source hashes are in the manifest.
`scale_probe.py` loads Tiger from the requested local Git commit; the common
benchmark and accelerator/tagging helpers are the unchanged baseline versions.
The source commit must exist locally, for example after fetching the PR branch.
From the repository root containing these evidence drivers:

```sh
export PYTHONPATH='/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages:src'
python3 -S benchmarks/evidence/2026-09-30-qkv-learning/scale_probe.py --source-ref 50f1f5c --control lr --seeds 43 47 53 --output /tmp/qkv-reference.json
python3 -S benchmarks/evidence/2026-09-30-qkv-learning/scale_probe.py --source-ref bbd3d39 --control group --seeds 43 47 53 --output /tmp/qkv-corrected.json
```

For MPS, add `--device mps --seeds 43` and use a fresh output path. For the
feature exploration, copy `probe.py` to the same relative path in a checkout
at baseline `50f1f5c`, then run it with `--config gain-zero|baseline|fast|fast-spectral`,
`--seeds 23 37 41`, and a fresh `--output`. The confirmation uses seeds
`61 67 71` and the gain-zero/fast/fast-spectral configurations. The Python path
above is specific to the measured Mac; another installation can run the scripts
with its own PyTorch environment. `-S` avoided a local default-device override.
