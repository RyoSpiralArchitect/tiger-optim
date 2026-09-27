#!/usr/bin/env python3
"""Render held-out learning curves for development and confirmation seeds."""

import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from compare import _load  # noqa: E402

ROOT = Path(__file__).resolve().parent
PANELS = (
    ("Development seeds 23, 37, 41", (
        ("dev_full_constant", "Tiger + spectral", "#C2511A"),
        ("dev_off_constant", "Tiger without spectral", "#177E89"),
        ("dev_adamw_constant", "AdamW (context)", "#445A9C"),
    )),
    ("Confirmation seeds 43, 47, 53", (
        ("confirm_full_constant", "Tiger constant LR", "#C2511A"),
        ("confirm_full_tail", "Tiger tail cosine", "#445A9C"),
    )),
    ("MPS confirmation seeds 43, 47, 53", (
        ("mps_confirm_full_constant", "Tiger constant LR", "#C2511A"),
        ("mps_confirm_full_tail", "Tiger tail cosine", "#445A9C"),
        ("mps_confirm_off_tail", "Tiger tail, no spectral", "#177E89"),
    )),
)


def main():
    runs = _load()
    figure, axes = plt.subplots(1, 3, figsize=(17.5, 4.8), layout="constrained",
                                sharey=True)
    for axis, (title, styles) in zip(axes, PANELS):
        for key, label, color in styles:
            records = runs[key]["runs"]
            steps = [entry["step"] for entry in records[0]["evaluations"]]
            values = [
                [entry["heldout"]["ce"] for entry in record["evaluations"]]
                for record in records
            ]
            median = [statistics.median(row[index] for row in values)
                      for index in range(len(steps))]
            low = [min(row[index] for row in values)
                   for index in range(len(steps))]
            high = [max(row[index] for row in values)
                    for index in range(len(steps))]
            axis.plot(steps, median, marker="o", linewidth=2.0, markersize=3.2,
                      color=color, label=label)
            axis.fill_between(steps, low, high, color=color, alpha=0.13)
        for boundary in (25, 50, 75, 100):
            axis.axvline(boundary, color="#777777", alpha=0.25, linewidth=0.7)
        axis.set(xlim=(0, 100), ylim=(0.0001, 4), xlabel="Optimizer updates",
                 title=title)
        axis.set_yscale("log")
        axis.grid(axis="y", which="both", alpha=0.16)
        axis.legend(loc="lower left", frameon=False, fontsize=8)
    axes[0].set_ylabel("Held-out next-token cross-entropy")
    figure.suptitle("Toy causal Transformer: period-3 copy task")
    figure.text(0.5, -0.015,
                "Median and range across three seeds. Tail schedule chosen on development seeds. "
                "Synthetic task; not an optimizer ranking.",
                ha="center", fontsize=8)
    output = ROOT / "learning-curves.png"
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
