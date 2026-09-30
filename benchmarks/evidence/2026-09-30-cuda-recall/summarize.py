"""Verify every archived run and summarize the frozen recall comparison."""
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


def load(name):
    return json.loads(gzip.decompress((HERE / "raw" / (name + ".json.gz")).read_bytes()))


def main():
    plan_bytes = (HERE / "plan.json").read_bytes()
    plan = json.loads(plan_bytes)
    selection = json.loads((HERE / "selection.json").read_text())
    assert hashlib.sha256(plan_bytes).hexdigest() == selection["plan_sha256"]
    manifest = json.loads((HERE / "manifest.json").read_text())
    for relative, digest in manifest["decompressed_raw_sha256"].items():
        assert hashlib.sha256(gzip.decompress((HERE / relative).read_bytes())).hexdigest() == digest
    records = {}
    source = None
    for path in sorted((HERE / "raw").glob("*.json.gz")):
        record = json.loads(gzip.decompress(path.read_bytes()))
        assert record["status"] == "complete"
        config = record["config"]
        assert config["device"] == "cuda" and config["precision"] == "bf16"
        assert record["model"] == plan["model"]
        assert config["batch_size"] == plan["batch_size"]
        assert record["git"]["status"] == ""
        this_source = (record["git"]["head"], record["source_sha256"])
        if source is None:
            source = this_source
        assert source == this_source
        assert [r["step"] for r in record["steps"]] == list(range(1, config["steps"] + 1))
        assert all(math.isfinite(r["loss"]) and math.isfinite(r["grad_norm"]) for r in record["steps"])
        assert [r["step"] for r in record["evaluations"]] == list(range(0, config["steps"] + 1, 100))
        assert all(math.isfinite(r["validation"]["ce"]) for r in record["evaluations"])
        assert config["development"] == ("test" not in record)
        records[path.name.removesuffix(".json.gz")] = record
    assert len(records) == 18
    for relative, digest in source[1].items():
        historical = subprocess.check_output(["git", "show", source[0] + ":" + relative], cwd=ROOT)
        assert hashlib.sha256(historical).hexdigest() == digest, relative

    def match_inputs(runs):
        for other in runs[1:]:
            assert other["initial_state_sha256"] == runs[0]["initial_state_sha256"]
            assert other["data_sha256"] == runs[0]["data_sha256"]
            for left, right in zip(other["steps"], runs[0]["steps"]):
                assert math.isclose(left["lr"] / other["config"]["lr"],
                                    right["lr"] / runs[0]["config"]["lr"], rel_tol=1e-12)

    dev = plan["development"]
    development = []
    for mode, rates in dev["learning_rates"].items():
        for rate in rates:
            r = records["dev-" + mode + "-" + str(rate)]
            assert r["config"]["seed"] == dev["seed"] and r["config"]["steps"] == dev["steps"]
            assert r["config"]["lr"] == rate and r["config"]["mode"] == mode
            assert "test" not in r["data_sha256"]
            development.append(r)
        candidates = selection["selected"][mode]["candidates"]
        assert candidates == [{"lr": rate, "validation_ce": records["dev-" + mode + "-" + str(rate)]["evaluations"][-1]["validation"]["ce"]} for rate in rates]
        assert selection["selected"][mode]["lr"] == min(candidates, key=lambda p: p["validation_ce"])["lr"]
    match_inputs(development)

    # Reconstruct test inputs and check their hash before computing this
    # descriptive, query-independent baseline. It did not select training settings.
    spec = importlib.util.spec_from_file_location("recall", ROOT / "benchmarks/bench_associative_recall.py")
    recall = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recall)
    recall.torch.set_num_threads(2)
    rows = []
    confirm = plan["confirmation"]
    for seed in confirm["seeds"]:
        runs = [records["confirm-" + mode + "-" + str(seed)] for mode in confirm["modes"]]
        match_inputs(runs)
        config = runs[0]["config"]
        tokens, targets = recall.corpus(seed + 30000, config["test_size"], **{key: config[key] for key in ("symbols", "pairs", "gap", "queries")})
        assert recall.tensor_hash(zip(("tokens", "targets"), (tokens, targets))) == runs[0]["data_sha256"]["test"]
        values = tokens[:, 1:2 * config["pairs"]:2] - config["symbols"]
        common_value = values.mode(dim=1).values
        context_mode_accuracy = (common_value[:, None] == targets).float().mean().item()
        counts = recall.F.one_hot(values, num_classes=config["symbols"]).sum(dim=1)
        probabilities = counts.gather(1, targets).float() / config["pairs"]
        context_frequency_ce = -probabilities.log().mean().item()
        row = {"seed": seed, "context_mode_accuracy": context_mode_accuracy,
               "context_frequency_ce": context_frequency_ce, "modes": {}}
        for mode, r in zip(confirm["modes"], runs):
            assert r["config"]["seed"] == seed and r["config"]["steps"] == confirm["steps"]
            assert r["config"]["mode"] == mode
            family = "adamw" if mode == "adamw" else "tiger-full"
            assert r["config"]["lr"] == selection["selected"][family]["lr"]
            assert len(r["data_sha256"]) == 3
            assert r["test"]["queries"] == config["test_size"] * config["queries"]
            assert math.isfinite(r["test"]["ce"]) and 0 <= r["test"]["accuracy"] <= 1
            if mode == "tiger-fixed-qkv":
                assert all(p["qkv_scales"] == {"q": 0.9, "k": 0.8, "v": 1.1} for p in r["evaluations"])
            elif mode.startswith("tiger"):
                assert r["evaluations"][-1]["qkv_scales"] != r["evaluations"][0]["qkv_scales"]
            row["modes"][mode] = {**r["test"], "qkv_scales": r["evaluations"][-1]["qkv_scales"]}
        rows.append(row)
    aggregates = {}
    for mode in confirm["modes"]:
        aggregates[mode] = {}
        for metric in ("ce", "accuracy"):
            values = [row["modes"][mode][metric] for row in rows]
            aggregates[mode][metric] = {"mean": statistics.mean(values), "min": min(values), "max": max(values)}
    diagnostic = json.loads(gzip.decompress((HERE / "diagnostics/adamw-5000.json.gz").read_bytes()))
    diagnostic_plan = json.loads((HERE / "diagnostic_plan.json").read_text())
    assert diagnostic["status"] == "complete" and "test" not in diagnostic
    assert (diagnostic["git"]["head"], diagnostic["source_sha256"]) == source
    for key in ("mode", "lr", "seed", "steps", "development", "eval_interval"):
        assert diagnostic["config"][key] == diagnostic_plan[key]
    assert len(diagnostic["steps"]) == 5000
    assert all(math.isfinite(p["loss"]) for p in diagnostic["steps"])
    dc = diagnostic["config"]
    tokens, targets = recall.corpus(dc["seed"] + 20000, dc["eval_size"], **{key: dc[key] for key in ("symbols", "pairs", "gap", "queries")})
    assert recall.tensor_hash(zip(("tokens", "targets"), (tokens, targets))) == diagnostic["data_sha256"]["validation"]
    values = tokens[:, 1:2 * dc["pairs"]:2] - dc["symbols"]
    counts = recall.F.one_hot(values, num_classes=dc["symbols"]).sum(dim=1)
    diagnostic_baseline = {
        "ce": -(counts.gather(1, targets).float() / dc["pairs"]).log().mean().item(),
        "accuracy": (values.mode(dim=1).values[:, None] == targets).float().mean().item(),
    }
    summary = {
        "source_commit": source[0], "parameter_count": development[0]["parameter_count"],
        "environment": development[0]["environment"], "development": selection["selected"],
        "confirmation": rows, "aggregate": aggregates,
        "uniform_value_guess_accuracy": 1 / plan["model"]["symbols"],
        "context_mode_accuracy_mean": statistics.mean(row["context_mode_accuracy"] for row in rows),
        "context_frequency_ce_mean": statistics.mean(row["context_frequency_ce"] for row in rows),
        "full_beats_no_spectral_ce_seeds": sum(row["modes"]["tiger-full"]["ce"] < row["modes"]["tiger-no-spectral"]["ce"] for row in rows),
        "full_beats_fixed_qkv_ce_seeds": sum(row["modes"]["tiger-full"]["ce"] < row["modes"]["tiger-fixed-qkv"]["ce"] for row in rows),
        "post_confirmation_diagnostic": {"seed": dc["seed"], "steps": dc["steps"],
            "validation": diagnostic["evaluations"][-1]["validation"],
            "context_baseline": diagnostic_baseline, "test_evaluated": False},
        "scope": "finite update budget, synthetic retrieval, three confirmation seeds, limited LR search; no speed or general optimizer ranking claim",
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
