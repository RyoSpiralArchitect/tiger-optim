#!/usr/bin/env python3
"""Validate and summarize the archived toy Transformer learning runs."""

import gzip
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SPECS = {
    "dev_full_constant": ("v2-full-constant-dev.json", "tiger-full", "constant", (23, 37, 41)),
    "dev_off_constant": ("v2-no-spectral-constant-dev.json", "tiger-no-spectral", "constant", (23, 37, 41)),
    "dev_full_tail": ("v2-full-tail-dev.json", "tiger-full", "tail-cosine", (23, 37, 41)),
    "dev_off_tail": ("v2-no-spectral-tail-dev.json", "tiger-no-spectral", "tail-cosine", (23, 37, 41)),
    "dev_adamw_constant": ("v2-adamw-constant-dev.json", "adamw", "constant", (23, 37, 41)),
    "confirm_full_constant": ("v2-full-constant-confirm.json", "tiger-full", "constant", (43, 47, 53)),
    "confirm_full_tail": ("v2-full-tail-confirm.json", "tiger-full", "tail-cosine", (43, 47, 53)),
    "pilot_full_cosine": ("v2-full-cosine-pilot.json", "tiger-full", "cosine", (37,)),
    "mps_confirm_full_constant": ("v2-full-constant-confirm-mps.json", "tiger-full", "constant", (43, 47, 53)),
    "mps_confirm_full_tail": ("v2-full-tail-confirm-mps.json", "tiger-full", "tail-cosine", (43, 47, 53)),
    "mps_confirm_off_tail": ("v2-no-spectral-tail-confirm-mps.json", "tiger-no-spectral", "tail-cosine", (43, 47, 53)),
}
INITIAL = {
    "dev_full_constant": "tiger-full-cpu.json",
    "dev_off_constant": "tiger-no-spectral-cpu.json",
    "dev_adamw_constant": "adamw-cpu.json",
}


def _read(path):
    with gzip.open(str(path) + ".gz", "rt") as source:
        return json.load(source)


def _load():
    records = {
        key: _read(ROOT / "raw/locked" / filename)
        for key, (filename, _, _, _) in SPECS.items()
    }
    reference = records["dev_full_constant"]
    for key, record in records.items():
        _, mode, schedule, seeds = SPECS[key]
        assert record["mode"] == mode
        assert record["optimizer"]["schedule"] == schedule
        assert record["optimizer"]["lr"] == (0.003 if mode == "adamw" else 0.01)
        is_mps = key.startswith("mps_")
        assert record["environment"]["device"] == ("mps" if is_mps else "cpu")
        assert record["task"]["steps"] == 100
        assert record["task"] == reference["task"]
        assert record["git"]["head"] == reference["git"]["head"]
        assert record["git"]["status"] == (
            "M README.md\n" if is_mps else ""
        ) + "?? benchmarks/evidence/2026-09-27-toy-transformer-learning/"
        assert record["sha256"] == reference["sha256"]
        assert [item["seed"] for item in record["runs"]] == list(seeds)
        if schedule == "tail-cosine":
            assert record["optimizer"]["cosine_min_lr"] == 0.001
        for item in record["runs"]:
            assert item["status"] == "complete"
            assert len(item["steps"]) == 100
            assert item["evaluations"][0]["step"] == 0
            assert item["evaluations"][-1]["step"] == 100
            assert all(math.isfinite(point["heldout"]["ce"])
                       for point in item["evaluations"])
            if mode.startswith("tiger"):
                for step in (25, 50, 75, 100):
                    observation = item["steps"][step - 1]
                    assert observation["qkv_weight_grad_l2"] > 0
                    assert observation["qkv_metrics"]

    for keys in (
        ("dev_full_constant", "dev_off_constant", "dev_full_tail",
         "dev_off_tail", "dev_adamw_constant"),
        ("confirm_full_constant", "confirm_full_tail"),
        ("mps_confirm_full_constant", "mps_confirm_full_tail", "mps_confirm_off_tail"),
    ):
        for index, first in enumerate(records[keys[0]]["runs"]):
            for key in keys[1:]:
                other = records[key]["runs"][index]
                assert other["seed"] == first["seed"]
                for field in ("initial_state_sha256", "corpus_sha256", "minibatches_sha256"):
                    assert other[field] == first[field], (key, first["seed"], field)
                assert other["evaluations"][0] == first["evaluations"][0]

    for cpu_key, mps_key in (
        ("confirm_full_constant", "mps_confirm_full_constant"),
        ("confirm_full_tail", "mps_confirm_full_tail"),
    ):
        for cpu, mps in zip(records[cpu_key]["runs"], records[mps_key]["runs"]):
            assert cpu["seed"] == mps["seed"]
            for field in ("initial_state_sha256", "corpus_sha256", "minibatches_sha256"):
                assert cpu[field] == mps[field]

    for key, filename in INITIAL.items():
        old = _read(ROOT / "raw/initial" / filename)
        new = records[key]
        assert old["mode"] == new["mode"]
        for before, after in zip(old["runs"], new["runs"]):
            assert before["seed"] == after["seed"]
            assert before["evaluations"] == after["evaluations"]
            for field in ("initial_state_sha256", "corpus_sha256", "minibatches_sha256"):
                assert before[field] == after[field]
    return records


def _summary(record):
    per_seed = []
    for item in record["runs"]:
        evals = {point["step"]: point for point in item["evaluations"]}
        regular = [point["optimizer_ms"] for point in item["steps"]
                   if point["step"] % 25]
        boundary = [point["optimizer_ms"] for point in item["steps"]
                    if point["step"] % 25 == 0]
        boundary_points = [point for point in item["steps"]
                           if point["step"] % 25 == 0]
        qkv_summary = None
        if record["mode"].startswith("tiger"):
            qkv_summary = {
                "weight_grad_l2_min": min(
                    point["qkv_weight_grad_l2"] for point in boundary_points),
                "frequency_factor_range": [
                    min(point["qkv_metrics"]["qkv_freq_factor"] for point in boundary_points),
                    max(point["qkv_metrics"]["qkv_freq_factor"] for point in boundary_points),
                ],
                "phase_boost_range": [
                    min(point["qkv_metrics"]["qkv_phase_boost"] for point in boundary_points),
                    max(point["qkv_metrics"]["qkv_phase_boost"] for point in boundary_points),
                ],
            }
        per_seed.append({
            "seed": item["seed"],
            "heldout_ce_by_step": {str(step): evals[step]["heldout"]["ce"]
                                   for step in (0, 25, 50, 75, 99, 100)},
            "final_train": evals[100]["train"],
            "final_heldout": evals[100]["heldout"],
            "final_step_lr": item["steps"][-1]["lr"],
            "optimizer_ms_median_regular": statistics.median(regular),
            "optimizer_ms_median_boundary": statistics.median(boundary),
            "qkv_boundary_summary": qkv_summary,
        })
    return {
        "mode": record["mode"],
        "lr": record["optimizer"]["lr"],
        "schedule": record["optimizer"]["schedule"],
        "per_seed": per_seed,
        "final_heldout_ce_median": statistics.median(
            item["final_heldout"]["ce"] for item in per_seed),
        "final_heldout_accuracy_median": statistics.median(
            item["final_heldout"]["accuracy"] for item in per_seed),
    }


def _deltas(summaries, left, right):
    return [a["final_heldout"]["ce"] - b["final_heldout"]["ce"]
            for a, b in zip(summaries[left]["per_seed"], summaries[right]["per_seed"])]


def main():
    records = _load()
    summaries = {key: _summary(record) for key, record in records.items()}
    comparison = {
        "source_commit": records["dev_full_constant"]["git"]["head"],
        "source_sha256": records["dev_full_constant"]["sha256"],
        "git_status_during_locked_runs": records["dev_full_constant"]["git"]["status"],
        "devices": ["cpu", "mps"],
        "development_seeds": [23, 37, 41],
        "confirmation_seeds": [43, 47, 53],
        "mps_confirmation_seeds": [43, 47, 53],
        "shared_initial_state_corpus_minibatches_within_sets": True,
        "initial_constant_reproduced_by_locked_script": True,
        "runs": summaries,
        "paired_final_heldout_ce_delta": {
            "dev_full_minus_off_constant": _deltas(summaries, "dev_full_constant", "dev_off_constant"),
            "dev_full_minus_off_tail": _deltas(summaries, "dev_full_tail", "dev_off_tail"),
            "dev_tail_minus_constant": _deltas(summaries, "dev_full_tail", "dev_full_constant"),
            "confirm_tail_minus_constant": _deltas(summaries, "confirm_full_tail", "confirm_full_constant"),
            "mps_confirm_tail_minus_constant": _deltas(summaries, "mps_confirm_full_tail", "mps_confirm_full_constant"),
            "mps_confirm_full_minus_off_tail": _deltas(summaries, "mps_confirm_full_tail", "mps_confirm_off_tail"),
        },
        "interpretation": [
            "Both Tiger spectral modes learned the held-out task in all development seeds.",
            "Spectral adaptation had no consistent held-out CE advantage here.",
            "The tail schedule was selected after development results; three new confirmation seeds all favored it at step 100.",
            "The same three seeds favored the tail schedule on MPS; spectral-on and spectral-off MPS tail results were close.",
            "Full-period cosine at the same initial LR underfit exploratory seed 37.",
            "AdamW is contextual because its LR and Tiger's tagged LR scales differ.",
            "Single fresh process per mode and concurrent host work make CPU/MPS timing diagnostic only.",
        ],
    }
    output = ROOT / "comparison.json"
    output.write_text(json.dumps(comparison, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "output": str(output),
        "median_final_heldout_ce": {
            key: value["final_heldout_ce_median"] for key, value in summaries.items()},
        "confirmation_tail_minus_constant_ce":
            comparison["paired_final_heldout_ce_delta"]["confirm_tail_minus_constant"],
        "mps_confirmation_tail_minus_constant_ce":
            comparison["paired_final_heldout_ce_delta"]["mps_confirm_tail_minus_constant"],
    }, indent=2))


if __name__ == "__main__":
    main()
