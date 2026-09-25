#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


_TIMESTAMP_RE = re.compile(r"(\d{8}-\d{6})(?=\.json$)")
_BASELINE_MODE = "adamw"
_DEVICE_COLORS = {
    "cpu": "#355C7D",
    "mps": "#C06C84",
    "cuda": "#6C5B7B",
}


def _path_order_key(path: str) -> Tuple[int, str]:
    match = _TIMESTAMP_RE.search(path)
    if match:
        return (1, match.group(1))
    try:
        return (0, f"{Path(path).stat().st_mtime_ns:020d}")
    except OSError:
        return (0, path)


def _row_key(row: dict) -> Tuple[str, str]:
    return (str(row.get("device", "")), str(row.get("mode", "")))


def _sort_key(row: dict) -> Tuple[str, bool, str]:
    mode = str(row.get("mode", ""))
    return (
        str(row.get("device", "")),
        mode.lower() != _BASELINE_MODE,
        mode,
    )


def load_summaries(pattern: str, *, latest_only: bool = True) -> List[dict]:
    rows: List[dict] = []
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception as exc:
            print("Skip", path, ":", exc)
            continue
        summary = payload.get("summary", {})
        summary["path"] = path
        rows.append(summary)

    if latest_only:
        latest: Dict[Tuple[str, str], dict] = {}
        for row in rows:
            key = _row_key(row)
            current = latest.get(key)
            if current is None or _path_order_key(str(row.get("path", ""))) >= _path_order_key(str(current.get("path", ""))):
                latest[key] = row
        rows = list(latest.values())

    rows.sort(key=_sort_key)
    return rows


def _label(row: dict) -> str:
    return f'{str(row.get("device", "")).upper()} · {row.get("mode", "?")}'


def bar_by_mode(rows: List[dict], out_png: str) -> None:
    labels = [_label(row) for row in rows]
    values = [float(row.get("ms_median", 0.0)) for row in rows]
    colors = [_DEVICE_COLORS.get(str(row.get("device", "")).lower(), "#999999") for row in rows]

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    bars = ax.bar(range(len(values)), values, color=colors)
    ax.set_xticks(range(len(values)), labels, rotation=20, ha="right")
    ax.set_ylabel("Median step time (ms)")
    ax.set_title("Tiger Local Smoke Bench")
    ax.grid(axis="y", linestyle=":", alpha=0.35)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print("Saved plot:", out_png)


def line_loss(rows: List[dict], out_png: str, max_points: int = 200) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    for row in rows:
        path = str(row.get("path", ""))
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        loss = payload.get("series", {}).get("loss", [])
        if not loss:
            continue
        n = len(loss)
        step = max(1, n // max_points)
        xs = list(range(0, n, step))
        ys = [loss[index] for index in xs]
        ax.plot(xs, ys, label=_label(row), linewidth=2.0)

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curves (latest run per device/mode)")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print("Saved plot:", out_png)


def main() -> None:
    parser = argparse.ArgumentParser("Plot Tiger benchmark results")
    parser.add_argument("--pattern", default="benchmarks/results/compare-*.json")
    parser.add_argument("--out-dir", default="benchmarks/plots")
    parser.add_argument("--max-points", type=int, default=200)
    parser.add_argument(
        "--include-history",
        action="store_true",
        help="Plot every matched JSON instead of only the latest file per device/mode.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = load_summaries(args.pattern, latest_only=not args.include_history)
    if not rows:
        print("No results found for", args.pattern)
        return

    bar_by_mode(rows, os.path.join(args.out_dir, "median_step_time.png"))
    line_loss(rows, os.path.join(args.out_dir, "loss_curves.png"), max_points=max(1, args.max_points))


if __name__ == "__main__":
    main()
