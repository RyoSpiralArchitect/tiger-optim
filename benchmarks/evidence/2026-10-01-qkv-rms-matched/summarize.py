"""Verify matched QKV magnitude budgets and report every new paired seed."""
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
    index = json.loads((HERE / "index.json").read_text())
    assert index["plan_sha256"] == hashlib.sha256(plan_bytes).hexdigest()
    assert len(index["results"]) == manifest["raw_count"] == len(plan["seeds"]) * len(plan["recipes"]) == 15
    expected_files = {f"raw/{label}-{seed}.json.gz" for seed in plan["seeds"] for label in plan["recipes"]}
    indexed = {"raw/" + r["file"] + ".gz": r["sha256"] for r in index["results"]}
    assert set(indexed) == expected_files and indexed == manifest["decompressed_raw_sha256"]
    parent_plan = json.loads((HERE.parent / plan["parent_evidence"] / "plan.json").read_text())
    older_plan = json.loads((HERE.parent / parent_plan["parent_evidence"] / "plan.json").read_text())
    prior_seeds = parent_plan["seeds"] + older_plan["confirmation"]["seeds"] + [older_plan["development"]["seed"]]
    assert not set(plan["seeds"]) & set(prior_seeds) and plan["lr"] == parent_plan["lr"]
    spec = importlib.util.spec_from_file_location("recall", ROOT / "benchmarks/bench_associative_recall.py")
    recall = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recall)
    recall.torch.set_default_device("cpu")
    recall.torch.set_num_threads(plan["cpu_threads"])
    historical = {}
    records = {}
    raw_hashes = {}
    for seed in plan["seeds"]:
        for label, recipe in plan["recipes"].items():
            relative = f"raw/{label}-{seed}.json.gz"
            raw = gzip.decompress((HERE / relative).read_bytes())
            assert hashlib.sha256(raw).hexdigest() == indexed[relative]
            r = json.loads(raw)
            assert r["status"] == "complete" and r["schema"] == 4
            assert r["git"] == {"head": manifest["source_commit"], "status": ""}
            for path, digest in r["source_sha256"].items():
                if path not in historical:
                    historical[path] = subprocess.check_output(["git", "show", manifest["source_commit"] + ":" + path], cwd=ROOT)
                assert hashlib.sha256(historical[path]).hexdigest() == digest
            config = r["config"]
            assert config["seed"] == seed and config["mode"] == recipe["mode"]
            assert config["output"] == f"{label}-{seed}.json"
            assert config["measure_qkv_updates"] and not config["development"]
            assert bool(config["qkv_rms_reference"]) == recipe["rms_match"]
            for key in ("lr", "steps", "batch_size", "device", "precision", "cpu_threads", "eval_interval", "eval_size", "test_size"):
                assert config[key] == plan[key]
            assert r["model"] == plan["model"] and r["parameter_count"] == 102912
            controls = r["optimizer_controls"]
            assert controls["initial_qkv_scales"] == recipe["scales"]
            assert not controls["qkv_lr_autoadapt"] and not controls["qkv_spectral_adapt"] and controls["qkv_trust_split"]
            assert controls["trust_space"] == "precond" and controls["trust_clip"] == 5.0
            assert [p["step"] for p in r["steps"]] == list(range(1, plan["steps"] + 1))
            assert [p["step"] for p in r["evaluations"]] == list(range(0, plan["steps"] + 1, plan["eval_interval"]))
            for p in r["steps"]:
                assert math.isfinite(p["loss"]) and math.isfinite(p["grad_norm"])
                assert math.isclose(p["lr"], plan["lr"] * recall.lr_factor(p["step"] - 1, plan["steps"]), rel_tol=1e-12)
                u = p["qkv_update"]
                values = list(u["rms_by_slice"].values())
                assert all(math.isfinite(x) and x >= 0 for x in values + [u["combined_rms"]])
                assert math.isclose(u["combined_rms"] ** 2, statistics.mean(x * x for x in values), rel_tol=1e-5)
            for p in r["evaluations"]:
                assert p["qkv_scales"] == controls["initial_qkv_scales"] and math.isfinite(p["validation"]["ce"])
            assert r["binding_check"]["split"] == "test"
            for score in (r["test"], r["binding_check"]["rebound"], r["binding_check"]["old_targets"]):
                assert math.isfinite(score["ce"]) and 0 <= score["accuracy"] <= 1
                assert score["queries"] == plan["test_size"] * plan["model"]["queries"]
            records[(seed, label)] = r
            raw_hashes[(seed, label)] = indexed[relative]
    for path, content in historical.items():
        if path.startswith("src/"):
            assert subprocess.check_output(["git", "show", manifest["parent_commit"] + ":" + path], cwd=ROOT) == content

    def metrics(r):
        matches = [p["qkv_rms_match"] for p in r["steps"] if "qkv_rms_match" in p]
        return {"test": r["test"], "rebound": r["binding_check"]["rebound"],
                "old_target_accuracy": r["binding_check"]["old_targets"]["accuracy"],
                "binding_gate_passed": r["test"]["accuracy"] >= 0.9 and r["binding_check"]["rebound"]["accuracy"] >= 0.9 and r["binding_check"]["old_targets"]["accuracy"] <= 0.2,
                "sampled_validation_ce_mean": statistics.mean(p["validation"]["ce"] for p in r["evaluations"][1:]),
                "first_sampled_step_at_90_percent": next((p["step"] for p in r["evaluations"] if p["validation"]["accuracy"] >= 0.9), None),
                "applied_combined_rms_mean": statistics.mean(p["qkv_update"]["combined_rms"] for p in r["steps"]),
                "max_rms_relative_error": max((m["relative_error"] for m in matches), default=0),
                "rescale_factor_range": [min(m["rescale_factor"] for m in matches), max(m["rescale_factor"] for m in matches)] if matches else None,
                "final_state_sha256": r["final_state_sha256"]}

    rows = []
    for seed in plan["seeds"]:
        reference = records[(seed, "reference-uniform")]
        for label, recipe in plan["recipes"].items():
            r = records[(seed, label)]
            for key in ("initial_state_sha256", "data_sha256", "source_sha256"):
                assert r[key] == reference[key]
            assert [p["lr"] for p in r["steps"]] == [p["lr"] for p in reference["steps"]]
            if recipe["rms_match"]:
                assert r["config"]["qkv_rms_reference"] == f"reference-uniform-{seed}.json"
                assert r["qkv_rms_reference"] == {"sha256": raw_hashes[(seed, "reference-uniform")],
                                                  "source_commit": manifest["source_commit"], "mode": "tiger-uniform-qkv", "max_relative_tolerance": plan["match_tolerance"]}
                for p, target in zip(r["steps"], reference["steps"]):
                    m = p["qkv_rms_match"]
                    expected = target["qkv_update"]["combined_rms"]
                    assert m["target_rms"] == expected and math.isfinite(m["candidate_rms"]) and m["candidate_rms"] > 0
                    actual = p["qkv_update"]["combined_rms"]
                    assert math.isclose(m["applied_rms"], actual, rel_tol=1e-7)
                    error = abs(actual - expected) / expected
                    assert error <= plan["match_tolerance"] and math.isclose(m["relative_error"], error, rel_tol=1e-12, abs_tol=1e-15)
                    assert math.isclose(m["rescale_factor"], expected / m["candidate_rms"], rel_tol=1e-12)
            else:
                assert r["qkv_rms_reference"] is None and all("qkv_rms_match" not in p for p in r["steps"])
        datasets = {}
        for name, offset, count in (("train", 10000, plan["steps"] * plan["batch_size"]), ("validation", 20000, plan["eval_size"]), ("test", 30000, plan["test_size"])):
            data = recall.corpus(seed + offset, count, **{k: plan["model"][k] for k in ("symbols", "pairs", "gap", "queries")})
            assert recall.tensor_hash(zip(("tokens", "targets"), data)) == reference["data_sha256"][name]
            datasets[name] = data
        contexts = {name: set(map(tuple, data[0][:, :2 * plan["model"]["pairs"]].tolist())) for name, data in datasets.items()}
        assert not contexts["train"] & contexts["validation"] and not contexts["train"] & contexts["test"] and not contexts["validation"] & contexts["test"]
        rebound = recall.rebind_values(datasets["test"], **{k: plan["model"][k] for k in ("symbols", "pairs", "gap", "queries")})
        assert recall.tensor_hash(zip(("tokens", "targets"), rebound)) == reference["data_sha256"]["test_rebound"]
        changed = (rebound[1] != datasets["test"][1]).float().mean().item()
        assert all(records[(seed, label)]["binding_check"]["changed_target_fraction"] == changed for label in plan["recipes"])
        recall.torch.manual_seed(seed)
        assert recall.tensor_hash(recall.RecallTransformer(**plan["model"]).state_dict().items()) == reference["initial_state_sha256"]
        row = {"seed": seed, "changed_target_fraction": changed, "recipes": {label: metrics(records[(seed, label)]) for label in plan["recipes"]}}
        row["reference_replay_bitwise_equal"] = row["recipes"]["reference-uniform"]["final_state_sha256"] == row["recipes"]["matched-uniform"]["final_state_sha256"]
        rows.append(row)
    aggregate = {}
    for label in plan["recipes"]:
        scores = [row["recipes"][label] for row in rows]
        aggregate[label] = {"test_ce_mean": statistics.mean(s["test"]["ce"] for s in scores),
                            "test_accuracy_mean": statistics.mean(s["test"]["accuracy"] for s in scores),
                            "rebound_accuracy_mean": statistics.mean(s["rebound"]["accuracy"] for s in scores),
                            "binding_gate_pass_count": sum(s["binding_gate_passed"] for s in scores),
                            "sampled_validation_ce_mean": statistics.mean(s["sampled_validation_ce_mean"] for s in scores),
                            "max_rms_relative_error": max(s["max_rms_relative_error"] for s in scores)}
    deltas = [row["recipes"]["matched-asymmetric"]["test"]["ce"] - row["recipes"]["matched-uniform"]["test"]["ce"] for row in rows]
    learning_deltas = [row["recipes"]["matched-asymmetric"]["sampled_validation_ce_mean"] - row["recipes"]["matched-uniform"]["sampled_validation_ce_mean"] for row in rows]
    summary = {"source_commit": manifest["source_commit"], "environment": reference["environment"], "confirmation": rows,
               "aggregate": aggregate, "test_ce_delta_by_seed": deltas, "mean_test_ce_delta": statistics.mean(deltas),
               "seeds_with_lower_test_ce": sum(d < 0 for d in deltas), "sampled_validation_ce_delta_by_seed": learning_deltas,
               "reference_replay_bitwise_equal_count": sum(row["reference_replay_bitwise_equal"] for row in rows),
               "input_contexts_disjoint": True,
               "scope": "new paired seeds, equal per-step physical QKV RMS within 0.1 percent, common rescaling wrapper; reference-dependent intervention, non-QKV magnitude and moment state unconstrained; no general optimizer claim"}
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
