"""Plot all paired endpoints and per-step physical QKV RMS match errors."""
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def main():
    summary = json.loads((HERE / "summary.json").read_text())
    colors = ["#16837d", "#bf681e"]
    labels = ["Matched uniform", "Matched asymmetric"]
    recipes = ["matched-uniform", "matched-asymmetric"]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.5), constrained_layout=True)
    for j, row in enumerate(summary["confirmation"]):
        endpoints = [row["recipes"][r]["test"]["ce"] for r in recipes]
        axes[0].plot([0, 1], endpoints, color="#999999", alpha=0.6, linewidth=1)
        for i, (recipe, color) in enumerate(zip(recipes, colors)):
            axes[0].scatter(i, endpoints[i], color=color, marker=("o", "s", "^", "D", "v")[j], s=55)
            axes[0].annotate(str(row["seed"]), (i, endpoints[i]), xytext=(-29 if i == 0 else 7, 4),
                             textcoords="offset points", fontsize=8)
            r = json.loads(gzip.decompress((HERE / "raw" / f"{recipe}-{row['seed']}.json.gz").read_bytes()))
            axes[1].plot([p["step"] for p in r["steps"]], [100 * p["qkv_rms_match"]["relative_error"] for p in r["steps"]],
                         color=color, alpha=0.25, linewidth=0.8)
    for label, color in zip(labels, colors):
        axes[1].plot([], [], color=color, label=label)
    axes[0].set_yscale("log")
    axes[0].set_xticks([0, 1], labels)
    axes[0].set_xlim(-0.35, 1.35)
    axes[0].set(ylabel="Final test CE (lower is better)", title="Paired allocation effects at matched RMS")
    axes[1].set(xlabel="Completed updates", ylabel="Relative QKV RMS error (%)",
                title="Both arms follow the same reference budget")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Mac CPU FP32 · 5 new seeds · common rescaling wrapper · 1,200 updates", fontsize=12)
    fig.savefig(HERE / "matched-rms.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
