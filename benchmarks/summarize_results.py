#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

_TIMESTAMP_RE = re.compile(r"(\d{8}-\d{6})(?=\.json$)")
_DEFAULT_BASELINE_MODE = "adamw"


def _load_rows(pattern: str) -> List[dict]:
    rows: List[dict] = []
    for path in sorted(glob.glob(pattern)):
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        summary = payload.get("summary", {})
        summary["path"] = path
        rows.append(summary)
    return rows


def _row_key(row: dict) -> Tuple[str, str]:
    return (str(row.get("device", "")), str(row.get("mode", "")))


def _path_order_key(path: str) -> Tuple[int, str]:
    match = _TIMESTAMP_RE.search(path)
    if match:
        return (1, match.group(1))
    try:
        return (0, f"{Path(path).stat().st_mtime_ns:020d}")
    except OSError:
        return (0, path)


def _latest_rows(rows: Iterable[dict]) -> List[dict]:
    latest: Dict[Tuple[str, str], dict] = {}
    for row in rows:
        key = _row_key(row)
        current = latest.get(key)
        if current is None or _path_order_key(str(row.get("path", ""))) >= _path_order_key(str(current.get("path", ""))):
            latest[key] = row
    return list(latest.values())


def _validate_rows(rows: Iterable[dict]) -> List[str]:
    errors: List[str] = []
    for row in rows:
        label = f"{row.get('device', '?')}:{row.get('mode', '?')}"
        ms = row.get("ms_median")
        if not isinstance(ms, (int, float)) or not math.isfinite(ms) or ms <= 0:
            errors.append(f"{label} has invalid ms_median={ms!r}")
        for key in ("loss_last", "loss_min", "loss_mean"):
            value = row.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                errors.append(f"{label} has invalid {key}={value!r}")
    return errors


def _load_baseline(path: str | None) -> Dict[Tuple[str, str], dict]:
    if not path:
        return {}
    baseline_path = Path(path)
    if not baseline_path.exists():
        return {}
    with baseline_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    rows = payload if isinstance(payload, list) else payload.get("rows", [])
    result: Dict[Tuple[str, str], dict] = {}
    for row in rows:
        key = (str(row.get("device", "")), str(row.get("mode", "")))
        result[key] = row
    return result


def _format_delta(current: float, baseline: dict | None) -> str:
    if not baseline:
        return "-"
    previous = baseline.get("ms_median")
    if not isinstance(previous, (int, float)) or not math.isfinite(previous) or previous <= 0:
        return "-"
    delta = ((current - previous) / previous) * 100.0
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.1f}%"


def _implicit_baseline(rows: Iterable[dict], baseline_mode: str = _DEFAULT_BASELINE_MODE) -> Dict[Tuple[str, str], dict]:
    by_device: Dict[str, dict] = {}
    collected = list(rows)
    for row in collected:
        if str(row.get("mode", "")).lower() == baseline_mode:
            by_device[str(row.get("device", ""))] = row

    baseline: Dict[Tuple[str, str], dict] = {}
    for row in collected:
        device = str(row.get("device", ""))
        device_baseline = by_device.get(device)
        if device_baseline is not None:
            baseline[_row_key(row)] = device_baseline
    return baseline


def _render_markdown(
    rows: List[dict],
    baseline: Dict[Tuple[str, str], dict],
    delta_label: str,
) -> str:
    lines = [
        f"| Device | Mode | Median ms | Mean ms | Last loss | Delta vs {delta_label} |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        key = _row_key(row)
        lines.append(
            "| {device} | {mode} | {ms_median:.3f} | {ms_mean:.3f} | {loss_last:.6f} | {delta} |".format(
                device=row.get("device", "?"),
                mode=row.get("mode", "?"),
                ms_median=float(row["ms_median"]),
                ms_mean=float(row.get("ms_mean", 0.0)),
                loss_last=float(row.get("loss_last", 0.0)),
                delta=_format_delta(float(row["ms_median"]), baseline.get(key)),
            )
        )
    return "\n".join(lines) + "\n"


def _emit_github_summary(markdown: str) -> None:
    target = os.getenv("GITHUB_STEP_SUMMARY")
    if not target:
        return
    with open(target, "a", encoding="utf-8") as fh:
        fh.write("## Benchmark Smoke Summary\n\n")
        fh.write(markdown)
        fh.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser("Summarize Tiger benchmark result JSON files")
    parser.add_argument("--pattern", default="benchmarks/results/compare-*.json")
    parser.add_argument("--baseline")
    parser.add_argument("--markdown-out")
    parser.add_argument("--emit-github-summary", action="store_true")
    args = parser.parse_args()

    rows = _load_rows(args.pattern)
    if not rows:
        raise SystemExit(f"No benchmark result files matched: {args.pattern}")

    errors = _validate_rows(rows)
    if errors:
        raise SystemExit("\n".join(errors))

    rows = _latest_rows(rows)
    rows.sort(
        key=lambda row: (
            str(row.get("device", "")),
            str(row.get("mode", "")).lower() != _DEFAULT_BASELINE_MODE,
            str(row.get("mode", "")),
        )
    )
    baseline = _load_baseline(args.baseline)
    delta_label = "baseline"
    if not baseline:
        baseline = _implicit_baseline(rows)
        delta_label = "AdamW"
    markdown = _render_markdown(rows, baseline, delta_label)
    print(markdown, end="")

    if args.markdown_out:
        out_path = Path(args.markdown_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(markdown, encoding="utf-8")
    if args.emit_github_summary:
        _emit_github_summary(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
