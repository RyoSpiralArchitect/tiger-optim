"""Render the paired final outcomes and sampled BF16 learning curves."""
import gzip
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
COLORS = {"adamw": "#7e8290", "tiger-full": "#df820d", "tiger-no-spectral": "#24857d", "tiger-spectral-fade": "#6256c5"}
LABELS = {"adamw": "AdamW", "tiger-full": "Tiger full spectral", "tiger-no-spectral": "Tiger no spectral", "tiger-spectral-fade": "Tiger spectral fade"}


def main():
    summary = json.loads((HERE / "summary.json").read_text())
    plan = json.loads((HERE / "plan.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for i, row in enumerate(summary["confirmation"]["bf16"]):
        values = [row["recipes"][mode]["test"]["ce"] for mode in plan["modes"]]
        axes[0].plot(range(4), values, "o-", alpha=0.7, linewidth=1, label=str(row["seed"]))
    axes[0].set_yscale("log")
    axes[0].set_xticks(range(4), ["AdamW", "Full", "Off", "Fade"])
    axes[0].set_ylabel("Final test cross entropy (log scale)")
    axes[0].set_title("Paired BF16 seeds; all outcomes retained")
    axes[0].legend(title="Seed", fontsize=7)
    for mode in plan["modes"]:
        trajectories = []
        for seed in plan["seeds"]:
            raw = gzip.decompress((HERE / "raw" / f"confirm-bf16-{mode}-{seed}.json.gz").read_bytes())
            trajectories.append(json.loads(raw)["evaluations"])
        steps = [p["step"] for p in trajectories[0]]
        medians = [statistics.median(t[i]["validation"]["ce"] for t in trajectories) for i in range(len(steps))]
        axes[1].plot(steps, medians, color=COLORS[mode], label=LABELS[mode])
    axes[1].axvspan(800, 1200, color=COLORS["tiger-spectral-fade"], alpha=0.08, label="Fade window")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Completed updates")
    axes[1].set_ylabel("Median validation CE (log scale)")
    axes[1].set_title("608,256 parameters; 8 pairs + 8 gap tokens")
    axes[1].legend(fontsize=8)
    fig.savefig(HERE / "cuda-spectral.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
