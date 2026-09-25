# CPU comparison smoke, 2026-09-25

This directory preserves a negative local result. The six raw JSON files are
copies of the outputs named in `manifest.json`; their SHA-256 hashes match the
manifest. Each `repN` pair was produced in a fresh Python process on an Apple
M4 CPU with 50 warmup and 200 measured steps, seed 0, and the default TinyMix
shape. The manifest records the exact command, source hashes, environment,
optimizer settings, pair IDs, and original output paths.

| Mode | Median of three process medians, compute | `opt.step()` | First to last measured training loss |
| --- | ---: | ---: | ---: |
| AdamW | 4.0484 ms (4.0358–4.0763) | 0.7794 ms | 0.764382 → 0.100509 |
| Tiger v2.1 full preset | 8.9731 ms (8.9588–9.0747) | 5.5773 ms | 1.035726 → 0.996406 |

These runs repeat **one seed** and optimize a fixed synthetic training tensor
pair with independently random targets. The loss measures fitting that batch,
not generalization. AdamW ran first in each process, the optimizers used
different settings, and the checkout was dirty. The timer excludes `loss.item()` and
`report_metrics()`. This evidence supports neither a general speed claim nor a
convergence claim. The loss after warmup already differs between modes because
the preceding 50 updates differ.
