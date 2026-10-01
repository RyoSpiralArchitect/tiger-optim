"""Validate the frozen QKV scale factorial and derive all paired effects."""
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    plan_bytes = (HERE / "plan.json").read_bytes()
    plan = json.loads(plan_bytes)
    manifest = json.loads((HERE / "manifest.json").read_text())
    parent = HERE.parent / plan["parent_evidence"]
    parent_plan = json.loads((parent / "plan.json").read_text())
    parent_selection = json.loads((parent / "selection.json").read_text())
    assert plan["lr"] == parent_selection["selected"]["tiger-full"]["lr"]
    assert not set(plan["seeds"]) & set(parent_plan["confirmation"]["seeds"] + [parent_plan["development"]["seed"]])
    assert manifest["raw_counts"] == {"cpu": 20, "mps": 2}
    indexed = {}
    for phase in ("cpu", "mps"):
        index = json.loads((HERE / (phase + "-index.json")).read_text())
        assert index["phase"] == phase and index["plan_sha256"] == hashlib.sha256(plan_bytes).hexdigest()
        assert len(index["results"]) == manifest["raw_counts"][phase]
        for row in index["results"]:
            relative = "raw/" + row["file"] + ".gz"
            assert relative not in indexed
            indexed[relative] = row["sha256"]
    assert indexed == manifest["decompressed_raw_sha256"]
    spec = importlib.util.spec_from_file_location("recall", ROOT / "benchmarks/bench_associative_recall.py")
    recall = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recall)
    recall.torch.set_default_device("cpu")
    recall.torch.set_num_threads(plan["cpu_threads"])
    historical_sources = {}
    records = {}
    for relative, expected in indexed.items():
        raw = gzip.decompress((HERE / relative).read_bytes())
        assert hashlib.sha256(raw).hexdigest() == expected
        r = json.loads(raw)
        assert r["status"] == "complete" and r["schema"] == 3
        assert r["git"] == {"head": manifest["source_commit"], "status": ""}
        for path, digest in r["source_sha256"].items():
            if path not in historical_sources:
                historical_sources[path] = subprocess.check_output(["git", "show", manifest["source_commit"] + ":" + path], cwd=ROOT)
            assert hashlib.sha256(historical_sources[path]).hexdigest() == digest
        config = r["config"]
        phase = config["device"]
        mode = config["mode"]
        assert relative == f"raw/{phase}-{mode}-{config['seed']}.json.gz"
        assert mode in plan["cells"] and phase in ("cpu", "mps")
        assert not config["development"] and config["measure_qkv_updates"]
        assert r["model"] == plan["model"] and r["parameter_count"] == 102912
        for key in ("lr", "steps", "batch_size", "eval_interval", "eval_size", "test_size", "precision", "cpu_threads"):
            assert config[key] == plan[key]
        controls = r["optimizer_controls"]
        assert not controls["qkv_lr_autoadapt"] and not controls["qkv_spectral_adapt"]
        assert controls["qkv_trust_split"] and controls["trust_space"] == "precond" and controls["trust_clip"] == 5.0
        assert controls["initial_qkv_scales"] == plan["cells"][mode]["scales"]
        assert math.isclose(statistics.mean(controls["initial_qkv_scales"].values()), plan["cells"][mode]["mean_scale"], rel_tol=1e-14)
        assert [p["step"] for p in r["steps"]] == list(range(1, plan["steps"] + 1))
        assert [p["step"] for p in r["evaluations"]] == list(range(0, plan["steps"] + 1, plan["eval_interval"]))
        for p in r["steps"]:
            assert math.isfinite(p["loss"]) and math.isfinite(p["grad_norm"])
            assert math.isclose(p["lr"], plan["lr"] * recall.lr_factor(p["step"] - 1, plan["steps"]), rel_tol=1e-12)
            update = p["qkv_update"]
            assert set(update["rms_by_slice"]) == {"q", "k", "v"}
            values = list(update["rms_by_slice"].values())
            assert all(math.isfinite(x) and x >= 0 for x in values + [update["combined_rms"]])
            assert math.isclose(update["combined_rms"] ** 2, statistics.mean(x * x for x in values), rel_tol=1e-5)
        for p in r["evaluations"]:
            assert p["qkv_scales"] == controls["initial_qkv_scales"]
            assert math.isfinite(p["validation"]["ce"])
        assert r["binding_check"]["split"] == "test"
        for score in (r["test"], r["binding_check"]["rebound"], r["binding_check"]["old_targets"]):
            assert math.isfinite(score["ce"]) and 0 <= score["accuracy"] <= 1
            assert score["queries"] == plan["test_size"] * plan["model"]["queries"]
        records[(phase, config["seed"], mode)] = r

    # All package code is inherited; only benchmark controls/measurement changed.
    for path, content in historical_sources.items():
        if path.startswith("src/"):
            old = subprocess.check_output(["git", "show", manifest["parent_commit"] + ":" + path], cwd=ROOT)
            assert old == content

    def matches(runs):
        for other in runs[1:]:
            for key in ("initial_state_sha256", "data_sha256", "source_sha256"):
                assert other[key] == runs[0][key]
            assert [p["lr"] for p in other["steps"]] == [p["lr"] for p in runs[0]["steps"]]

    def metrics(r):
        updates = [p["qkv_update"] for p in r["steps"]]
        return {"test": r["test"], "rebound": r["binding_check"]["rebound"],
                "old_target_accuracy": r["binding_check"]["old_targets"]["accuracy"],
                "binding_gate_passed": r["test"]["accuracy"] >= 0.9 and r["binding_check"]["rebound"]["accuracy"] >= 0.9 and r["binding_check"]["old_targets"]["accuracy"] <= 0.2,
                "sampled_validation_ce_mean": statistics.mean(p["validation"]["ce"] for p in r["evaluations"][1:]),
                "first_sampled_step_at_90_percent": next((p["step"] for p in r["evaluations"] if p["validation"]["accuracy"] >= 0.9), None),
                "applied_update": {"combined_rms_mean": statistics.mean(p["combined_rms"] for p in updates),
                                   "slice_rms_mean": {key: statistics.mean(p["rms_by_slice"][key] for p in updates) for key in ("q", "k", "v")},
                                   "first_step": updates[0]}}

    rows = []
    for seed in plan["seeds"]:
        runs = [records[("cpu", seed, mode)] for mode in plan["cells"]]
        matches(runs)
        assert len({r["final_state_sha256"] for r in runs}) == len(runs)
        datasets = {}
        for name, offset, count in (("train", 10000, plan["steps"] * plan["batch_size"]), ("validation", 20000, plan["eval_size"]), ("test", 30000, plan["test_size"])):
            data = recall.corpus(seed + offset, count, **{key: plan["model"][key] for key in ("symbols", "pairs", "gap", "queries")})
            assert recall.tensor_hash(zip(("tokens", "targets"), data)) == runs[0]["data_sha256"][name]
            datasets[name] = data
        contexts = {name: set(map(tuple, data[0][:, :2 * plan["model"]["pairs"]].tolist())) for name, data in datasets.items()}
        assert not contexts["train"] & contexts["validation"] and not contexts["train"] & contexts["test"] and not contexts["validation"] & contexts["test"]
        rebound = recall.rebind_values(datasets["test"], **{key: plan["model"][key] for key in ("symbols", "pairs", "gap", "queries")})
        assert recall.tensor_hash(zip(("tokens", "targets"), rebound)) == runs[0]["data_sha256"]["test_rebound"]
        changed = (rebound[1] != datasets["test"][1]).float().mean().item()
        assert all(r["binding_check"]["changed_target_fraction"] == changed for r in runs)
        recall.torch.manual_seed(seed)
        model = recall.RecallTransformer(**plan["model"])
        assert recall.tensor_hash(model.state_dict().items()) == runs[0]["initial_state_sha256"]
        values = datasets["test"][0][:, 1:2 * plan["model"]["pairs"]:2] - plan["model"]["symbols"]
        baseline = (values.mode(dim=1).values[:, None] == datasets["test"][1]).float().mean().item()
        row = {"seed": seed, "context_mode_accuracy": baseline, "changed_target_fraction": changed,
               "modes": {mode: metrics(records[("cpu", seed, mode)]) for mode in plan["cells"]}}
        # The first update shares gradients and optimizer state. This checks that
        # each requested slice multiplier actually reaches parameter updates.
        base_mode = "tiger-uniform-qkv"
        for mode in plan["cells"]:
            for key in ("q", "k", "v"):
                actual = row["modes"][mode]["applied_update"]["first_step"]["rms_by_slice"][key]
                base = row["modes"][base_mode]["applied_update"]["first_step"]["rms_by_slice"][key]
                assert math.isclose(actual / base, plan["cells"][mode]["scales"][key], rel_tol=1e-3)
        rows.append(row)
    aggregate = {}
    for mode in plan["cells"]:
        scores = [row["modes"][mode] for row in rows]
        aggregate[mode] = {"test_ce_mean": statistics.mean(s["test"]["ce"] for s in scores),
                           "test_accuracy_mean": statistics.mean(s["test"]["accuracy"] for s in scores),
                           "rebound_accuracy_mean": statistics.mean(s["rebound"]["accuracy"] for s in scores),
                           "binding_gate_pass_count": sum(s["binding_gate_passed"] for s in scores),
                           "sampled_validation_ce_mean": statistics.mean(s["sampled_validation_ce_mean"] for s in scores),
                           "applied_combined_rms_mean": statistics.mean(s["applied_update"]["combined_rms_mean"] for s in scores)}
    contrasts = {}
    for label, (control, feature) in plan["contrasts"].items():
        paired = [(row["modes"][control], row["modes"][feature]) for row in rows]
        deltas = [f["test"]["ce"] - c["test"]["ce"] for c, f in paired]
        contrasts[label] = {"control": control, "feature": feature, "test_ce_delta_by_seed": deltas,
                            "mean_test_ce_delta": statistics.mean(deltas), "seeds_with_lower_test_ce": sum(d < 0 for d in deltas),
                            "sampled_validation_ce_delta_by_seed": [f["sampled_validation_ce_mean"] - c["sampled_validation_ce_mean"] for c, f in paired],
                            "first_step_combined_rms_ratio_by_seed": [f["applied_update"]["first_step"]["combined_rms"] / c["applied_update"]["first_step"]["combined_rms"] for c, f in paired],
                            "mean_applied_combined_rms_ratio_by_seed": [f["applied_update"]["combined_rms_mean"] / c["applied_update"]["combined_rms_mean"] for c, f in paired]}
    interaction = [unit - low for unit, low in zip(contrasts["allocation_at_unit_mean"]["test_ce_delta_by_seed"], contrasts["allocation_at_low_mean"]["test_ce_delta_by_seed"])]
    replay = {"seed": plan["mps_replay"]["seed"], "modes": {}}
    mps_runs = []
    for mode in plan["mps_replay"]["modes"]:
        r = records[("mps", replay["seed"], mode)]
        matches([records[("cpu", replay["seed"], mode)], r])
        replay["modes"][mode] = metrics(r)
        mps_runs.append(r)
    matches(mps_runs)
    summary = {"source_commit": manifest["source_commit"], "environment": runs[0]["environment"],
               "parameter_count": 102912, "confirmation": rows, "aggregate": aggregate, "contrasts": contrasts,
               "test_ce_interaction_by_seed": interaction, "mean_test_ce_interaction": statistics.mean(interaction),
               "input_contexts_disjoint": True, "mps_replay": replay,
               "scope": "five new paired seeds on a short synthetic task; arithmetic mean LR scales matched, actual update RMS observed rather than forced equal; inherited LR; no general optimizer claim"}
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
