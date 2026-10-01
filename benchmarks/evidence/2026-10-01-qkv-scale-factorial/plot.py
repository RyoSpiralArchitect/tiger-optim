"""Plot all factorial endpoints and actual applied QKV update magnitudes."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def main():
    summary = json.loads((HERE / "summary.json").read_text())
    modes = list(summary["aggregate"])
    labels = ["Asymmetric\nmean 0.933", "Asymmetric\nmean 1", "Uniform\nmean 0.933", "Uniform\nmean 1"]
    colors = ["#7955a3", "#bf681e", "#16837d", "#52657a"]
    markers = ["o", "s", "^", "D", "v"]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.5), constrained_layout=True)
    for i, (mode, color) in enumerate(zip(modes, colors)):
        for j, row in enumerate(summary["confirmation"]):
            score = row["modes"][mode]
            x = i + (j - 2) * 0.09
            axes[0].scatter(x, score["test"]["ce"], color=color, marker=markers[j], s=50)
            axes[1].scatter(x, 1000 * score["applied_update"]["combined_rms_mean"], color=color, marker=markers[j], s=50)
    axes[0].set_yscale("log")
    axes[0].set(ylabel="Final test CE (lower is better)", title="All 5 new paired seed endpoints")
    axes[1].set(ylabel="Mean applied QKV update RMS × 1,000", title="Actual update magnitudes differ")
    for row, marker in zip(summary["confirmation"], markers):
        axes[0].scatter([], [], color="#555555", marker=marker, label=str(row["seed"]))
    axes[0].legend(title="Seed", fontsize=8)
    for axis in axes:
        axis.set_xticks(range(len(modes)), labels)
        axis.grid(alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("QKV allocation × mean LR multiplier · Mac CPU FP32 · 1,200 updates", fontsize=12)
    fig.savefig(HERE / "factorial.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
