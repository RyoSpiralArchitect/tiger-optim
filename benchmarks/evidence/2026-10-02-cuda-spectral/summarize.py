"""Validate frozen source/input receipts before deriving paired CUDA results."""
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import subprocess

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def read_json(name):
    return json.loads((HERE / name).read_text())


def main():
    plan = read_json("plan.json")
    preflight = read_json("preflight_plan.json")
    manifest = read_json("manifest.json")
    selection = read_json("preflight_selection.json")
    index = read_json("confirmation_index.json")
    final_compatibility = read_json("final_compatibility.json")
    initializations = read_json("initialization/receipt.json")
    input_replay = read_json("input_replay/receipt.json")
    assert input_replay["source_commit"] == manifest["implementation_source_commit"]
    assert input_replay["replayer_sha256"] == hashlib.sha256((HERE / "replay_inputs.py").read_bytes()).hexdigest()
    assert input_replay["environment"]["torch"] == "2.13.0+cu132" and input_replay["environment"]["architecture"] == "x86_64"
    assert set(input_replay["seeds"]) == set(map(str, plan["seeds"]))
    assert initializations["source_commit"] == manifest["implementation_source_commit"]
    assert initializations["replayer_sha256"] == hashlib.sha256((HERE / "replay_initialization.py").read_bytes()).hexdigest()
    assert initializations["environment"]["torch"] == "2.13.0+cu132"
    assert initializations["environment"]["architecture"] == "x86_64" and initializations["environment"]["device"] == "cpu"
    assert set(initializations["seeds"]) == set(map(str, plan["seeds"]))
    assert final_compatibility["plan_sha256"] == hashlib.sha256((HERE / "compatibility_plan.json").read_bytes()).hexdigest()
    assert selection["plan_sha256"] == hashlib.sha256((HERE / "preflight_plan.json").read_bytes()).hexdigest()
    assert index["plan_sha256"] == hashlib.sha256((HERE / "plan.json").read_bytes()).hexdigest()
    assert index["complete"] and len(index["results"]) == 36
    expected = {f"preflight-{task}-{mode}-{lr}.json" for task in preflight["tasks"]
                for mode, rates in preflight["learning_rates"].items() for lr in rates}
    expected |= {f"confirm-{precision}-{mode}-{seed}.json" for precision, seeds in plan["precisions"].items()
                 for mode in plan["modes"] for seed in seeds}
    expected.add("compatibility-default-tiger.json")
    expected.add("compatibility-final-tiger.json")
    assert manifest["raw_count"] == len(expected) == 46
    assert set(manifest["decompressed_raw_sha256"]) == {"raw/" + label + ".gz" for label in expected}
    assert {p.name for p in (HERE / "raw").glob("*.gz")} == {label + ".gz" for label in expected}
    spec = importlib.util.spec_from_file_location("recall", ROOT / "benchmarks/bench_associative_recall.py")
    recall = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recall)
    recall.torch.set_default_device("cpu")
    recall.torch.set_num_threads(2)
    historical = {}
    records = {}

    def binding_passed(r, gate, development=False):
        original = r["evaluations"][-1]["validation"] if development else r["test"]
        return (original["accuracy"] >= gate["original_min"] and
                r["binding_check"]["rebound"]["accuracy"] >= gate["rebound_min"] and
                r["binding_check"]["old_targets"]["accuracy"] <= gate["obsolete_max"])

    def validate(label, options, model, source, schema, development):
        relative = "raw/" + label + ".json.gz"
        data = gzip.decompress((HERE / relative).read_bytes())
        assert hashlib.sha256(data).hexdigest() == manifest["decompressed_raw_sha256"][relative]
        r = json.loads(data)
        assert r["status"] == "complete" and r["schema"] == schema
        assert r["git"] == {"head": source, "status": ""}
        assert r["model"] == model and r["config"]["output"] == label + ".json"
        assert r["config"]["development"] == development
        assert r["config"]["measure_qkv_updates"] and r["qkv_rms_reference"] is None
        assert not r["config"]["qkv_rms_reference"] and r["config"]["checkpoint"] is None
        for key, value in options.items():
            assert r["config"][key] == value, (label, key)
        source_files = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", source, "src/tiger_optim"], cwd=ROOT, text=True).splitlines()
        source_files = ["benchmarks/bench_associative_recall.py"] + [p for p in source_files if p.endswith(".py")]
        assert set(r["source_sha256"]) == set(source_files)
        for path, digest in r["source_sha256"].items():
            key = (source, path)
            if key not in historical:
                historical[key] = subprocess.check_output(["git", "show", source + ":" + path], cwd=ROOT)
            assert hashlib.sha256(historical[key]).hexdigest() == digest
        assert r["environment"]["device"] == "NVIDIA GeForce RTX 5090"
        assert r["environment"]["torch"] == "2.13.0+cu132" and r["environment"]["cuda"] == "13.2"
        assert r["environment"]["precision"] == options["precision"] and not r["environment"]["tf32"]
        assert r["environment"]["cpu_threads"] == 2 and r["peak_cuda_bytes"] > 0
        assert [p["step"] for p in r["steps"]] == list(range(1, options["steps"] + 1))
        assert [p["step"] for p in r["evaluations"]] == list(range(0, options["steps"] + 1, options["eval_interval"]))
        controls = r["optimizer_controls"]
        mode = options["mode"]
        if mode == "adamw":
            assert controls is None
        else:
            assert controls["qkv_lr_autoadapt"] and controls["qkv_trust_split"]
            assert controls["qkv_spectral_adapt"] == (mode != "tiger-no-spectral")
            assert controls["initial_qkv_scales"] == {"q": 0.9, "k": 0.8, "v": 1.1}
            assert controls["qkv_lr_interval"] == 25 and controls["qkv_lr_gain"] == 0.02
            assert controls["trust_space"] == "precond" and controls["trust_clip"] == 5.0
            if schema == 5:
                assert controls["initial_qkv_spectral_strength"] == 1.0
        for p in r["steps"]:
            assert all(math.isfinite(p[k]) for k in ("loss", "lr", "grad_norm", "seconds"))
            assert math.isclose(p["lr"], options["lr"] * recall.lr_factor(p["step"] - 1, options["steps"]), rel_tol=1e-12)
            u = p["qkv_update"]
            values = list(u["rms_by_slice"].values())
            assert all(math.isfinite(v) and v >= 0 for v in values + [u["combined_rms"]])
            assert math.isclose(u["combined_rms"] ** 2, statistics.mean(v * v for v in values), rel_tol=1e-5)
            if schema == 5:
                strength = None if mode == "adamw" else (recall.spectral_strength_factor(p["step"] - 1, options["steps"])
                                                        if mode == "tiger-spectral-fade" else 1.0)
                actual_strength = p["qkv_spectral_strength"]
                if strength is None or strength in (0.0, 1.0):
                    assert actual_strength == strength
                else:
                    # Host libm implementations can differ by one cosine ULP.
                    assert math.isclose(actual_strength, strength, rel_tol=1e-12, abs_tol=1e-15)
        scores = [p["validation"] for p in r["evaluations"]] + [r["binding_check"]["rebound"], r["binding_check"]["old_targets"]]
        if development:
            assert "test" not in r and set(r["data_sha256"]) == {"train", "validation", "validation_rebound"}
        else:
            scores.append(r["test"])
            assert set(r["data_sha256"]) == {"train", "validation", "test", "test_rebound"}
        if mode in ("tiger-no-spectral", "tiger-spectral-fade"):
            feedback = r["evaluations"][-1]["qkv_feedback"]
            assert feedback["qkv_freq_factor"] == feedback["qkv_phase_boost"] == 1.0
        for e in r["evaluations"]:
            if controls is not None:
                assert all(0.7 <= s <= 1.3 for s in e["qkv_scales"].values())
                assert all(v is None or math.isfinite(v) for v in e["qkv_feedback"].values())
        assert r["binding_check"]["split"] == ("validation" if development else "test")
        assert all(math.isfinite(s["ce"]) and 0 <= s["accuracy"] <= 1 for s in scores)
        assert r["final_state_sha256"] and r["initial_state_sha256"]
        records[label] = r
        return r

    preflight_results = {}
    for task, config in preflight["tasks"].items():
        preflight_results[task] = {}
        for mode, rates in preflight["learning_rates"].items():
            preflight_results[task][mode] = []
            for lr in rates:
                label = f"preflight-{task}-{mode}-{lr}"
                options = {key: preflight[key] for key in ("device", "precision", "cpu_threads", "eval_interval", "eval_size", "test_size")}
                options.update({key: config[key] for key in ("seed", "steps", "batch_size")})
                options.update(mode=mode, lr=lr)
                r = validate(label, options, config["model"], manifest["preflight_source_commit"], 4, True)
                passed = binding_passed(r, preflight["binding_gate"], True)
                candidate = next(c for c in selection["candidates"][task][mode] if c["lr"] == lr)
                assert candidate["binding_gate_passed"] == passed
                assert candidate["sha256"] == manifest["decompressed_raw_sha256"]["raw/" + label + ".json.gz"]
                assert candidate["validation_ce"] == r["evaluations"][-1]["validation"]["ce"]
                preflight_results[task][mode].append(candidate)
    chosen = None
    for task in ("medium", "short"):
        eligible = {mode: [c for c in candidates if c["binding_gate_passed"]]
                    for mode, candidates in preflight_results[task].items()}
        if all(eligible.values()):
            chosen = {"task": task, "learning_rates": {mode: min(candidates, key=lambda c: c["validation_ce"])["lr"]
                                                      for mode, candidates in eligible.items()}}
            break
    assert chosen == selection["selected"] == {"task": "medium", "learning_rates": {"adamw": 0.001, "tiger-full": 0.001}}
    assert plan["lr"] == 0.001 and plan["model"] == preflight["tasks"]["medium"]["model"]
    assert not set(plan["seeds"]) & {c["seed"] for c in preflight["tasks"].values()}

    old = records["preflight-medium-tiger-full-0.001"]
    compatibility_options = {k: old["config"][k] for k in ("mode", "device", "precision", "cpu_threads", "eval_interval", "eval_size", "test_size", "seed", "steps", "batch_size", "lr")}
    for label, source in (("compatibility-default-tiger", manifest["source_commit"]),
                          ("compatibility-final-tiger", manifest["implementation_source_commit"])):
        compatibility = validate(label, compatibility_options, old["model"], source, 5, True)
        for key in ("initial_state_sha256", "final_state_sha256", "data_sha256", "binding_check"):
            assert compatibility[key] == old[key]
        assert [{k: p[k] for k in ("step", "loss", "lr", "grad_norm", "qkv_update")} for p in old["steps"]] == [
            {k: p[k] for k in ("step", "loss", "lr", "grad_norm", "qkv_update")} for p in compatibility["steps"]]
        assert [{k: p[k] for k in ("step", "validation", "qkv_scales", "qkv_feedback")} for p in old["evaluations"]] == [
            {k: p[k] for k in ("step", "validation", "qkv_scales", "qkv_feedback")} for p in compatibility["evaluations"]]
    assert final_compatibility["source_commit"] == manifest["implementation_source_commit"]
    assert final_compatibility["reference_sha256"] == manifest["decompressed_raw_sha256"]["raw/preflight-medium-tiger-full-0.001.json.gz"]
    assert final_compatibility["result_sha256"] == manifest["decompressed_raw_sha256"]["raw/compatibility-final-tiger.json.gz"]
    assert final_compatibility["all_numeric_traces_equal"] and final_compatibility["final_weights_bitwise_equal"]
    indexed = {r["file"]: r["sha256"] for r in index["results"]}
    assert set(indexed) == {f"confirm-{precision}-{mode}-{seed}.json" for precision, seeds in plan["precisions"].items()
                            for mode in plan["modes"] for seed in seeds}
    all_rows = {}
    aggregates = {}
    deltas = {}
    native_initialization_diagnostics = {}
    native_input_diagnostics = {}
    for precision, seeds in plan["precisions"].items():
        rows = []
        for seed in seeds:
            runs = {}
            for mode in plan["modes"]:
                label = f"confirm-{precision}-{mode}-{seed}"
                options = {key: plan[key] for key in ("device", "cpu_threads", "lr", "steps", "batch_size", "eval_interval", "eval_size", "test_size")}
                options.update(mode=mode, seed=seed, precision=precision)
                r = validate(label, options, plan["model"], manifest["source_commit"], 5, False)
                assert indexed[label + ".json"] == manifest["decompressed_raw_sha256"]["raw/" + label + ".json.gz"]
                runs[mode] = r
            reference = runs["adamw"]
            for r in runs.values():
                for key in ("initial_state_sha256", "data_sha256", "source_sha256"):
                    assert r[key] == reference[key]
                assert [p["lr"] for p in r["steps"]] == [p["lr"] for p in reference["steps"]]
            data_config = {k: plan["model"][k] for k in ("symbols", "pairs", "gap", "queries")}
            contexts = {}
            replay = input_replay["seeds"][str(seed)]
            assert replay["file"] == f"key-ties-{seed}.npz"
            ties_file = HERE / "input_replay" / replay["file"]
            assert hashlib.sha256(ties_file.read_bytes()).hexdigest() == replay["file_sha256"]
            assert input_replay["benchmark_sha256"] == reference["source_sha256"]["benchmarks/bench_associative_recall.py"]
            native_input_diagnostics[str(seed)] = {}
            for split, offset, count in (("train", 10000, plan["steps"] * plan["batch_size"]),
                                         ("validation", 20000, plan["eval_size"]), ("test", 30000, plan["test_size"])):
                data = recall.corpus(seed + offset, count, **data_config)
                native_data_hash = recall.tensor_hash(zip(("tokens", "targets"), data))
                receipt = replay["splits"][split]
                assert receipt["native_replay_bitwise_equal"] and receipt["sha256"] == reference["data_sha256"][split]
                with np.load(ties_file, allow_pickle=False) as saved:
                    assert set(saved.files) == {name + suffix for name in ("train", "validation", "test") for suffix in ("_rows", "_keys")}
                    row_array, key_array = saved[split + "_rows"], saved[split + "_keys"]
                    assert row_array.dtype == key_array.dtype == np.int64
                    assert row_array.shape == (receipt["tie_row_count"],) and key_array.shape == (len(row_array), data_config["pairs"])
                    assert np.all((row_array >= 0) & (row_array < count)) and np.all(np.diff(row_array) > 0)
                    assert np.all((key_array >= 0) & (key_array < data_config["symbols"]))
                    tied_rows, keys = recall.torch.from_numpy(row_array), recall.torch.from_numpy(key_array)
                    if len(tied_rows):
                        rng = recall.torch.Generator(device="cpu").manual_seed(seed + offset)
                        scores = recall.torch.rand(count, data_config["symbols"], generator=rng, device="cpu")[tied_rows]
                        old_keys = data[0][tied_rows, :2 * data_config["pairs"]:2]
                        assert recall.torch.equal(scores.gather(1, old_keys), scores.gather(1, keys))
                        assert (keys.sort(dim=1).values[:, 1:] != keys.sort(dim=1).values[:, :-1]).all()
                        start = 2 * data_config["pairs"] + data_config["gap"] + 1
                        selected = (data[0][tied_rows, start::2, None] == old_keys[:, None, :]).long().argmax(-1)
                        data[0][tied_rows, :2 * data_config["pairs"]:2] = keys
                        data[0][tied_rows, start::2] = keys.gather(1, selected)
                native_input_diagnostics[str(seed)][split] = {"native_sha256": native_data_hash,
                                                              "native_bitwise_equal": native_data_hash == reference["data_sha256"][split],
                                                              "equal_score_rows": receipt["tie_row_count"]}
                assert recall.tensor_hash(zip(("tokens", "targets"), data)) == reference["data_sha256"][split]
                contexts[split] = set(map(tuple, data[0][:, :2 * plan["model"]["pairs"]].tolist()))
                if split == "test":
                    rebound = recall.rebind_values(data, **data_config)
                    assert recall.tensor_hash(zip(("tokens", "targets"), rebound)) == reference["data_sha256"]["test_rebound"]
                    changed = (data[1] != rebound[1]).float().mean().item()
                    assert all(r["binding_check"]["changed_target_fraction"] == changed for r in runs.values())
            assert not contexts["train"] & contexts["validation"] and not contexts["train"] & contexts["test"] and not contexts["validation"] & contexts["test"]
            recall.torch.manual_seed(seed)
            model = recall.RecallTransformer(**plan["model"])
            native_hash = recall.tensor_hash(model.state_dict().items())
            initializer = initializations["seeds"][str(seed)]
            assert initializer["native_replay_bitwise_equal"]
            assert initializer["initial_state_sha256"] == reference["initial_state_sha256"]
            assert initializations["benchmark_sha256"] == reference["source_sha256"]["benchmarks/bench_associative_recall.py"]
            assert initializer["embedding_file"] == f"embeddings-{seed}.npz"
            fixture = HERE / "initialization" / initializer["embedding_file"]
            assert hashlib.sha256(fixture.read_bytes()).hexdigest() == initializer["embedding_file_sha256"]
            differences = {}
            with np.load(fixture, allow_pickle=False) as values, recall.torch.no_grad():
                assert set(values.files) == {"token", "position"}
                for key in values.files:
                    weight = getattr(model, key).weight
                    array = values[key]
                    assert array.dtype == np.float32 and tuple(array.shape) == tuple(weight.shape)
                    assert np.isfinite(array).all()
                    stored = recall.torch.from_numpy(array)
                    differences[key] = (weight - stored).abs().max().item()
                    weight.copy_(stored)
            native_initialization_diagnostics[str(seed)] = {"native_state_sha256": native_hash,
                                                            "native_bitwise_equal": native_hash == reference["initial_state_sha256"],
                                                            "embedding_max_absolute_difference": differences}
            assert recall.tensor_hash(model.state_dict().items()) == reference["initial_state_sha256"]
            assert all(r["parameter_count"] == sum(p.numel() for p in model.parameters()) == 608256 for r in runs.values())
            prefix = all(a["model_state_sha256"] == b["model_state_sha256"] for a, b in
                         zip(runs["tiger-full"]["evaluations"][:9], runs["tiger-spectral-fade"]["evaluations"][:9]))
            metrics = {}
            for mode, r in runs.items():
                metrics[mode] = {"test": r["test"], "rebound": r["binding_check"]["rebound"],
                                 "obsolete_target_accuracy": r["binding_check"]["old_targets"]["accuracy"],
                                 "binding_gate_passed": binding_passed(r, plan["binding_gate"]),
                                 "sampled_validation_ce_mean": statistics.mean(p["validation"]["ce"] for p in r["evaluations"][1:]),
                                 "first_sampled_90_percent_step": next((p["step"] for p in r["evaluations"] if p["validation"]["accuracy"] >= 0.9), None),
                                 "applied_qkv_rms_mean": statistics.mean(p["qkv_update"]["combined_rms"] for p in r["steps"]),
                                 "train_seconds": r["train_seconds"], "peak_cuda_bytes": r["peak_cuda_bytes"]}
            rows.append({"seed": seed, "recipes": metrics, "full_fade_first_half_bitwise_equal": prefix})
        all_rows[precision] = rows
        aggregates[precision] = {}
        for mode in plan["modes"]:
            ms = [r["recipes"][mode] for r in rows]
            aggregates[precision][mode] = {"test_ce_mean": statistics.mean(m["test"]["ce"] for m in ms),
                                           "test_ce_median": statistics.median(m["test"]["ce"] for m in ms),
                                           "test_accuracy_mean": statistics.mean(m["test"]["accuracy"] for m in ms),
                                           "rebound_accuracy_mean": statistics.mean(m["rebound"]["accuracy"] for m in ms),
                                           "binding_gate_pass_count": sum(m["binding_gate_passed"] for m in ms),
                                           "sampled_validation_ce_mean": statistics.mean(m["sampled_validation_ce_mean"] for m in ms)}
        deltas[precision] = {}
        for left, right in (("tiger-full", "tiger-no-spectral"), ("tiger-spectral-fade", "tiger-full")):
            values = [r["recipes"][left]["test"]["ce"] - r["recipes"][right]["test"]["ce"] for r in rows]
            deltas[precision][left + "_minus_" + right] = {"by_seed": dict(zip(map(str, seeds), values)),
                                                        "mean": statistics.mean(values), "lower_ce_count": sum(v < 0 for v in values)}
    summary = {"source_commit": manifest["source_commit"], "implementation_source_commit": manifest["implementation_source_commit"], "environment": compatibility["environment"],
               "parameter_count": 608256, "training_updates": sum(len(r["steps"]) for r in records.values()),
               "preflight_selection": chosen, "default_strength_bitwise_compatible": True,
               "initialization_replay": {"original_runtime": initializations["environment"], "original_bitwise_match_count": 7,
                                          "portable_embedding_match_count": 7, "local_torch": recall.torch.__version__,
                                          "native_diagnostics": native_initialization_diagnostics},
               "input_replay": {"original_bitwise_match_count": 21, "portable_equal_score_replay_count": 21,
                                  "native_diagnostics": native_input_diagnostics},
               "all_input_contexts_disjoint": True, "confirmation": all_rows, "aggregate": aggregates, "paired_deltas": deltas,
               "scope": plan["scope"]}
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("training_updates", "preflight_selection", "aggregate", "paired_deltas")}, indent=2))


if __name__ == "__main__":
    main()
