"""Validate stored QKV runs and regenerate a compact comparison."""
import gzip
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INPUTS = ("initial_state_sha256", "corpus_sha256", "minibatches_sha256")

def load(name):
    return json.loads(gzip.decompress((ROOT / "raw" / (name + ".json.gz")).read_bytes()))

def inputs_equal(left, right):
    assert left["seed"] == right["seed"]
    for key in INPUTS:
        assert left[key] == right[key], (key, left["seed"])

def final(run):
    return run["evaluations"][-1]["heldout"]["ce"]

def main():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    for name, digest in manifest["decompressed_raw_sha256"].items():
        raw = gzip.decompress((ROOT / name).read_bytes())
        assert hashlib.sha256(raw).hexdigest() == digest
        record = json.loads(raw)
        for run in record["runs"]:
            assert run["status"] == "complete" and len(run["steps"]) == 100
            assert run["evaluations"][0]["step"] == 0
            assert run["evaluations"][-1]["step"] == 100
            assert all(math.isfinite(p["heldout"]["ce"]) for p in run["evaluations"])
        if "source_commit" in record:
            version = "baseline" if record["source_commit"] == manifest["baseline_commit"] else "candidate"
            assert record["source_commit"] == manifest[version + "_commit"]
            assert record["source_sha256"] == manifest[version + "_source_sha256"]
            assert record["driver_sha256"] == hashlib.sha256((ROOT / "scale_probe.py").read_bytes()).hexdigest()
            assert record["toy_sha256"] == manifest["toy_sha256"]
        else:
            assert record["git"]["head"] == manifest["baseline_commit"]
            assert record["source_sha256"]["src/tiger_optim/tiger.py"] == manifest["baseline_source_sha256"]
            assert record["source_sha256"]["benchmarks/bench_toy_transformer_learning.py"] == manifest["toy_sha256"]
            assert record["source_sha256"][str((ROOT / "probe.py").relative_to(ROOT.parents[2]))] == hashlib.sha256((ROOT / "probe.py").read_bytes()).hexdigest()

    names = ("gain-zero", "baseline", "fast", "fast-spectral")
    exploration = {name: load(name + "-development") for name in names}
    development = []
    for i, seed in enumerate((23, 37, 41)):
        runs = {name: record["runs"][i] for name, record in exploration.items()}
        for run in runs.values():
            inputs_equal(runs["gain-zero"], run)
        assert runs["gain-zero"]["seed"] == seed
        assert all(step["qkv_scales"] == {"q": 0.9, "k": 0.8, "v": 1.1}
                   for step in runs["gain-zero"]["steps"])
        development.append({"seed": seed, "final_heldout_ce": {
            name: final(run) for name, run in runs.items()},
            "final_qkv_scales": {name: run["steps"][-1]["qkv_scales"] for name, run in runs.items()}})

    confirmation = []
    confirm_names = ("gain-zero", "fast", "fast-spectral")
    confirm = {name: load(name + "-confirmation") for name in confirm_names}
    expected_configs = {
        "gain-zero": {"qkv_lr_gain": 0.0, "qkv_lr_interval": 25},
        "baseline": {"qkv_lr_gain": 0.02, "qkv_lr_interval": 25},
        "fast": {"qkv_lr_gain": 0.2, "qkv_lr_interval": 5},
        "fast-spectral": {"qkv_lr_gain": 0.2, "qkv_lr_interval": 5},
    }
    for mapping in (exploration, confirm):
        for name, record in mapping.items():
            assert record["config"] == name and record["overrides"] == expected_configs[name]
            assert record["mode"] == ("tiger-full" if name == "fast-spectral" else "tiger-no-spectral")
    for i, seed in enumerate((61, 67, 71)):
        runs = {name: record["runs"][i] for name, record in confirm.items()}
        for run in runs.values():
            inputs_equal(runs["gain-zero"], run)
        assert runs["gain-zero"]["seed"] == seed
        confirmation.append({"seed": seed, "final_heldout_ce": {
            name: final(run) for name, run in runs.items()}})

    before = load("scale-before-group-cpu")
    reference = load("scale-reference-lr-cpu")
    after = load("scale-after-group-cpu")
    old_path = ROOT.parent / "2026-09-27-toy-transformer-learning/raw/locked/v2-full-tail-confirm.json.gz"
    old = json.loads(gzip.decompress(old_path.read_bytes()))
    assert before["control"] == after["control"] == "group"
    assert reference["control"] == "lr"
    assert before["factor"] == reference["factor"] == after["factor"] == 0.5
    correctness = []
    for i, seed in enumerate((43, 47, 53)):
        b, r, a, o = [record["runs"][i] for record in (before, reference, after, old)]
        for run in (r, a, o):
            inputs_equal(b, run)
        assert b["seed"] == seed
        assert b["evaluations"] == o["evaluations"]
        assert a["evaluations"] == r["evaluations"]
        assert a["final_state_sha256"] == r["final_state_sha256"]
        correctness.append({"seed": seed, "before_group_half_ce": final(b),
            "reference_lr_half_ce": final(r), "after_group_half_ce": final(a),
            "after_matches_reference_model_and_evaluations": True})
    default = load("scale-after-default-cpu")
    assert default["factor"] == 1.0
    inputs_equal(default["runs"][0], old["runs"][0])
    assert default["runs"][0]["evaluations"] == old["runs"][0]["evaluations"]

    mps_r, mps_a = load("scale-reference-lr-mps"), load("scale-after-group-mps")
    assert mps_r["device"] == mps_a["device"] == "mps"
    r, a = mps_r["runs"][0], mps_a["runs"][0]
    inputs_equal(r, a)
    inputs_equal(r, reference["runs"][0])
    assert [p["step"] for p in r["evaluations"]] == [p["step"] for p in a["evaluations"]]
    assert [p["lr"] for p in r["steps"]] == [p["lr"] for p in a["steps"]]
    summary = {
        "development": development, "confirmation": confirmation,
        "group_scale_cpu": correctness,
        "default_scale_cpu_seed43_evaluations_unchanged": True,
        "group_scale_mps": {"seed": r["seed"], "reference_ce": final(r), "after_ce": final(a),
            "max_abs_heldout_ce_difference": max(abs(x["heldout"]["ce"] - y["heldout"]["ce"])
                for x, y in zip(r["evaluations"], a["evaluations"])),
            "max_abs_heldout_accuracy_difference": max(abs(x["heldout"]["accuracy"] - y["heldout"]["accuracy"])
                for x, y in zip(r["evaluations"], a["evaluations"]))},
        "interpretation": [
            "More frequent/higher-gain QKV adaptation changed learning but did not improve every development seed.",
            "Adding spectral adaptation to the fast setting did not consistently improve held-out CE.",
            "The fixed fast candidate beat gain-zero on only one of three new confirmation seeds; no default change is supported.",
            "The corrected group scale exactly matches direct QKV base-LR scaling on three CPU seeds.",
            "A smaller QKV LR was worse here; the correction makes the requested control effective, not generally superior.",
            "No wall-clock performance claim; timings were collected with uncontrolled host load.",
        ],
    }
    (ROOT / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
