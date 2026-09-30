"""Plot every confirmation seed from the frozen Mac QKV experiment."""
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    plan = json.loads((HERE / "plan.json").read_text())
    summary = json.loads((HERE / "summary.json").read_text())
    modes = plan["confirmation"]["modes"]
    labels = ["AdamW", "Full QKV", "No spectral", "Fixed scales", "Uniform scales", "Shared QKV trust"]
    colors = ["#52657a", "#c35511", "#16837d", "#7758b5", "#a87916", "#ae416d"]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), constrained_layout=True)
    for index, (mode, label, color) in enumerate(zip(modes, labels, colors)):
        curves = []
        for seed_index, seed in enumerate(plan["confirmation"]["seeds"]):
            record = json.loads(gzip.decompress((HERE / "raw" / f"confirm-{mode}-{seed}.json.gz").read_bytes()))
            steps = [p["step"] for p in record["evaluations"]]
            curves.append([100 * p["validation"]["accuracy"] for p in record["evaluations"]])
            ce = summary["confirmation"][seed_index]["modes"][mode]["test"]["ce"]
            axes[1].scatter(index + (seed_index - 1) * 0.12, ce, color=color,
                            marker=("o", "s", "^")[seed_index], s=55)
        curves = np.asarray(curves)
        axes[0].plot(steps, curves.mean(axis=0), color=color, label=label, linewidth=1.8)
        axes[0].fill_between(steps, curves.min(axis=0), curves.max(axis=0), color=color, alpha=0.08)
    axes[0].axhline(90, color="#888888", linewidth=0.8, linestyle="--")
    axes[0].set(xlabel="Completed updates", ylabel="Validation accuracy (%)", ylim=(0, 103),
                title="Learning curves: mean and range of 3 seeds")
    axes[0].legend(fontsize=8, loc="lower right")
    axes[1].set_yscale("log")
    axes[1].set_xticks(range(len(modes)), labels, rotation=25, ha="right")
    axes[1].set(ylabel="Final test cross entropy (lower is better)",
                title="Frozen endpoint: all paired seed results")
    for seed, marker in zip(plan["confirmation"]["seeds"], ("o", "s", "^")):
        axes[1].scatter([], [], color="#555555", marker=marker, label=f"Seed {seed}")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Mac CPU FP32 · 102,912 parameters · 4 pairs, no gap · 1,200 updates · LR 0.003", fontsize=12)
    fig.savefig(HERE / "learning.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
