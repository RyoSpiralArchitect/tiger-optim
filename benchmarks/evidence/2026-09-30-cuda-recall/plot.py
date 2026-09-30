"""Plot all confirmation seeds; the bands show their range, not confidence intervals."""
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
plan = json.loads((HERE / "plan.json").read_text())
colors = ("#485d73", "#d46a20", "#168a83", "#8a62a6")
fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
for mode, color in zip(plan["confirmation"]["modes"], colors):
    records = [json.loads(gzip.decompress((HERE / "raw" / ("confirm-" + mode + "-" + str(seed) + ".json.gz")).read_bytes()))
               for seed in plan["confirmation"]["seeds"]]
    steps = [point["step"] for point in records[0]["evaluations"]]
    for ax, metric in zip(axes, ("ce", "accuracy")):
        values = np.array([[p["validation"][metric] for p in r["evaluations"]] for r in records])
        ax.plot(steps, values.mean(axis=0), label=mode, color=color)
        ax.fill_between(steps, values.min(axis=0), values.max(axis=0), color=color, alpha=0.12)
        ax.set_xlabel("Completed updates")
        ax.grid(alpha=0.15)
axes[0].set_ylabel("Validation cross-entropy")
axes[0].set_ylim(bottom=0)
axes[1].set_ylabel("Validation query accuracy")
axes[1].set_ylim(0, 1)
axes[0].legend(fontsize=8)
fig.suptitle("CUDA associative recall · 3 confirmation seeds\nLines: mean; bands: min–max across seeds", fontsize=11)
fig.savefig(HERE / "learning.png", dpi=170)
