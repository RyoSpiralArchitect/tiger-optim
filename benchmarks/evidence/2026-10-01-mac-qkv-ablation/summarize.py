"""Validate frozen Mac ablations and derive per-component learning effects."""
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


def read(relative):
    return json.loads(gzip.decompress((HERE / relative).read_bytes()))


def main():
    plan_bytes = (HERE / "plan.json").read_bytes()
    plan = json.loads(plan_bytes)
    selection = json.loads((HERE / "selection.json").read_text())
    manifest = json.loads((HERE / "manifest.json").read_text())
    assert selection["plan_sha256"] == hashlib.sha256(plan_bytes).hexdigest()
    counts = manifest["raw_counts"]
    assert counts == {"development": 6, "confirmation": 18, "exploratory_preflight": 1, "mps_replay": 1}
    assert len(manifest["decompressed_raw_sha256"]) == sum(counts.values())
    spec = importlib.util.spec_from_file_location("recall", ROOT / "benchmarks/bench_associative_recall.py")
    recall = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recall)
    recall.torch.set_default_device("cpu")
    recall.torch.set_num_threads(2)
    source = None
    records = {}
    for relative, digest in manifest["decompressed_raw_sha256"].items():
        raw = gzip.decompress((HERE / relative).read_bytes())
        assert hashlib.sha256(raw).hexdigest() == digest
        record = json.loads(raw)
        assert record["status"] == "complete" and len(record["steps"]) == plan["steps"]
        for path, expected in record["source_sha256"].items():
            historical = subprocess.check_output(["git", "show", record["git"]["head"] + ":" + path], cwd=ROOT)
            assert hashlib.sha256(historical).hexdigest() == expected
        assert record["git"]["status"] == ""
        if relative.startswith("preflight/"):
            assert record["git"]["head"] == manifest["preflight_commit"]
            continue
        this_source = (record["git"]["head"], record["source_sha256"])
        if source is None:
            source = this_source
        assert source == this_source
        assert record["schema"] == 2
        config = record["config"]
        assert record["model"] == plan["model"]
        for key in ("batch_size", "steps", "eval_interval", "eval_size", "test_size", "precision", "cpu_threads"):
            assert config[key] == plan[key]
        assert config["device"] == ("mps" if relative.startswith("mps/") else plan["device"])
        assert [p["step"] for p in record["steps"]] == list(range(1, plan["steps"] + 1))
        assert [p["step"] for p in record["evaluations"]] == list(range(0, plan["steps"] + 1, plan["eval_interval"]))
        assert all(math.isfinite(p["loss"]) and math.isfinite(p["grad_norm"]) for p in record["steps"])
        assert all(math.isfinite(p["validation"]["ce"]) for p in record["evaluations"])
        assert config["development"] == ("test" not in record)
        assert record["binding_check"]["split"] == ("validation" if config["development"] else "test")
        scores = [record["binding_check"][key] for key in ("rebound", "old_targets")]
        if "test" in record:
            scores.append(record["test"])
        assert all(math.isfinite(score["ce"]) and 0 <= score["accuracy"] <= 1 for score in scores)
        if config["mode"] != "adamw":
            controls = record["optimizer_controls"]
            assert controls["trust_space"] == "precond" and controls["trust_clip"] == 5.0
            adaptive = config["mode"] in ("tiger-full", "tiger-no-spectral")
            assert controls["qkv_lr_autoadapt"] == adaptive
            assert controls["qkv_spectral_adapt"] == (config["mode"] == "tiger-full")
            assert controls["qkv_trust_split"] == (config["mode"] != "tiger-global-trust")
            initial = {"q": 1.0, "k": 1.0, "v": 1.0} if config["mode"] in ("tiger-uniform-qkv", "tiger-global-trust") else {"q": 0.9, "k": 0.8, "v": 1.1}
            assert controls["initial_qkv_scales"] == initial
            assert controls["qkv_lr_interval"] == 25 and controls["qkv_lr_gain"] == 0.02
            if adaptive:
                assert record["evaluations"][-1]["qkv_scales"] != initial
                for point in record["evaluations"][1:]:
                    feedback = point["qkv_feedback"]
                    assert feedback["qkv_gamma_eff"] is not None
                    if config["mode"] == "tiger-no-spectral":
                        assert feedback["qkv_freq_factor"] == feedback["qkv_phase_boost"] == 1.0
                if config["mode"] == "tiger-full":
                    for key in ("qkv_freq_factor", "qkv_phase_boost"):
                        assert any(p["qkv_feedback"][key] != 1.0 for p in record["evaluations"][1:])
            else:
                assert all(point["qkv_scales"] == initial for point in record["evaluations"])
        records[relative] = record
    assert source[0] == manifest["source_commit"]

    def record(name):
        return records["raw/" + name + ".json.gz"]

    def matches(runs):
        for other in runs[1:]:
            assert other["initial_state_sha256"] == runs[0]["initial_state_sha256"]
            assert other["data_sha256"] == runs[0]["data_sha256"]
            assert all(math.isclose(x["lr"] / other["config"]["lr"], y["lr"] / runs[0]["config"]["lr"], rel_tol=1e-12)
                       for x, y in zip(other["steps"], runs[0]["steps"]))

    dev_runs = []
    for mode, rates in plan["development"]["rates"].items():
        candidates = []
        for rate in rates:
            r = record("dev-" + mode + "-" + str(rate))
            assert r["config"]["seed"] == plan["development"]["seed"] and r["config"]["mode"] == mode
            assert r["config"]["lr"] == rate
            check = r["binding_check"]
            qualified = (r["evaluations"][-1]["validation"]["accuracy"] >= 0.9
                         and check["rebound"]["accuracy"] >= 0.9 and check["old_targets"]["accuracy"] <= 0.2)
            candidates.append({"lr": rate, "ce": r["evaluations"][-1]["validation"]["ce"], "qualified": qualified})
            dev_runs.append(r)
        assert candidates == selection["selected"][mode]["candidates"]
        chosen = min(candidates, key=lambda r: r["ce"])
        assert selection["selected"][mode]["lr"] == chosen["lr"] and chosen["qualified"]
    matches(dev_runs)

    rows = []
    for seed in plan["confirmation"]["seeds"]:
        modes = plan["confirmation"]["modes"]
        runs = [record("confirm-" + mode + "-" + str(seed)) for mode in modes]
        matches(runs)
        assert len({r["final_state_sha256"] for r in runs}) == len(modes)
        datasets = {}
        for name, offset, count in (("train", 10000, plan["steps"] * plan["batch_size"]), ("validation", 20000, plan["eval_size"]), ("test", 30000, plan["test_size"])):
            data = recall.corpus(seed + offset, count, **{k: plan["model"][k] for k in ("symbols", "pairs", "gap", "queries")})
            assert recall.tensor_hash(zip(("tokens", "targets"), data)) == runs[0]["data_sha256"][name]
            datasets[name] = data
        contexts = {name: set(map(tuple, data[0][:, :2 * plan["model"]["pairs"]].tolist())) for name, data in datasets.items()}
        assert not contexts["train"] & contexts["validation"] and not contexts["train"] & contexts["test"]
        assert not contexts["validation"] & contexts["test"]
        rebound = recall.rebind_values(datasets["test"], **{k: plan["model"][k] for k in ("symbols", "pairs", "gap", "queries")})
        assert recall.tensor_hash(zip(("tokens", "targets"), rebound)) == runs[0]["data_sha256"]["test_rebound"]
        changed = (rebound[1] != datasets["test"][1]).float().mean().item()
        values = datasets["test"][0][:, 1:2 * plan["model"]["pairs"]:2] - plan["model"]["symbols"]
        baseline_accuracy = (values.mode(dim=1).values[:, None] == datasets["test"][1]).float().mean().item()
        row = {"seed": seed, "context_mode_accuracy": baseline_accuracy, "changed_target_fraction": changed, "modes": {}}
        for mode, r in zip(modes, runs):
            assert r["config"]["seed"] == seed and r["config"]["mode"] == mode
            family = "adamw" if mode == "adamw" else "tiger-full"
            assert r["config"]["lr"] == selection["selected"][family]["lr"]
            assert r["binding_check"]["changed_target_fraction"] == changed
            assert r["test"]["queries"] == plan["test_size"] * plan["model"]["queries"]
            row["modes"][mode] = {
                "test": r["test"], "rebound": r["binding_check"]["rebound"],
                "old_target_accuracy": r["binding_check"]["old_targets"]["accuracy"],
                "sampled_validation_ce_mean": statistics.mean(p["validation"]["ce"] for p in r["evaluations"][1:]),
                "first_sampled_step_at_90_percent": next((p["step"] for p in r["evaluations"] if p["validation"]["accuracy"] >= 0.9), None),
                "final_qkv_scales": r["evaluations"][-1]["qkv_scales"],
            }
        rows.append(row)
    contrasts = {}
    for label, (control, feature) in plan["contrasts"].items():
        deltas = [row["modes"][feature]["test"]["ce"] - row["modes"][control]["test"]["ce"] for row in rows]
        learning_deltas = [row["modes"][feature]["sampled_validation_ce_mean"] - row["modes"][control]["sampled_validation_ce_mean"] for row in rows]
        contrasts[label] = {"control": control, "feature": feature, "test_ce_delta_by_seed": deltas,
                            "mean_test_ce_delta": statistics.mean(deltas), "seeds_with_lower_test_ce": sum(v < 0 for v in deltas),
                            "sampled_validation_ce_mean_delta_by_seed": learning_deltas}
    aggregate = {}
    for mode in plan["confirmation"]["modes"]:
        aggregate[mode] = {"test_ce_mean": statistics.mean(row["modes"][mode]["test"]["ce"] for row in rows),
                           "test_accuracy_mean": statistics.mean(row["modes"][mode]["test"]["accuracy"] for row in rows),
                           "rebound_accuracy_mean": statistics.mean(row["modes"][mode]["rebound"]["accuracy"] for row in rows)}
    summary = {"source_commit": source[0], "parameter_count": dev_runs[0]["parameter_count"],
               "environment": dev_runs[0]["environment"], "development": selection["selected"],
               "confirmation": rows, "aggregate": aggregate, "contrasts": contrasts,
               "context_mode_accuracy_mean": statistics.mean(row["context_mode_accuracy"] for row in rows),
               "input_contexts_disjoint": True,
               "scope": "short synthetic retrieval, frozen rates, three paired confirmation seeds; no general optimizer or wall-clock speed claim"}
    for relative, r in records.items():
        if relative.startswith("mps/"):
            assert r["config"]["mode"] == "tiger-full" and r["config"]["seed"] == plan["confirmation"]["seeds"][0]
            assert r["config"]["lr"] == selection["selected"]["tiger-full"]["lr"]
            cpu = record("confirm-tiger-full-" + str(r["config"]["seed"]))
            matches([cpu, r])
            summary["mps_replay"] = {"seed": r["config"]["seed"], "test": r["test"],
                                      "binding_check": r["binding_check"], "cpu_test": cpu["test"]}
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
