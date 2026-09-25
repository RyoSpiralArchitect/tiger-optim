"""Single controlled ablation of quality_probe.py: Tiger agc_clip=0.0."""

from __future__ import annotations

import copy
import hashlib
import json
import runpy
import sys
import time
from pathlib import Path

BASE = Path("/tmp/tiger-optim-20260925-quality")
probe = runpy.run_path(str(BASE / "quality_probe.py"))
torch = probe["torch"]
Tiger = probe["Tiger"]
MLP = probe["MLP"]
data_for_seed = probe["data_for_seed"]
check_cpu = probe["check_cpu"]


def run(seed: int):
    train_x, train_y, heldout_x, heldout_y, batches = data_for_seed(seed)
    torch.manual_seed(seed + 1000)
    initial = copy.deepcopy(MLP().to("cpu").state_dict())
    model = MLP().to("cpu")
    model.load_state_dict(initial)
    check_cpu(model, train_x, train_y, heldout_x, heldout_y, batches)
    group = {"params": list(model.parameters()), "rms_clip_threshold": 1.0,
             "rms_clip_granularity": "param"}
    optimizer = Tiger(
        [group], lr=2e-4, weight_decay=0.01,
        factored=True, precond_alpha=1.0, trust_space="precond",
        update_buffer_dtype="fp32", agc_clip=0.0, trust_clip=5.0,
        auto_lr=False, auto_blend=False,
    )
    trace = []
    start = time.perf_counter()
    for step in range(probe["STEPS"] + 1):
        if step in probe["EVAL_STEPS"]:
            model.eval()
            with torch.no_grad():
                trace.append({
                    "step": step,
                    "train_mse": torch.nn.functional.mse_loss(model(train_x), train_y).item(),
                    "heldout_mse": torch.nn.functional.mse_loss(model(heldout_x), heldout_y).item(),
                })
        if step == probe["STEPS"]:
            break
        model.train()
        batch = batches[step]
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(train_x[batch]), train_y[batch])
        loss.backward()
        optimizer.step()
        assert all(torch.isfinite(p).all().item() for p in model.parameters())
    return {"seed": seed, "trace": trace,
            "elapsed_seconds": time.perf_counter() - start}


def main():
    assert torch.get_default_device() == torch.device("cpu")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    baseline = json.loads((BASE / "results.json").read_text())
    result = {
        "label": "single AGC ablation; preliminary CPU-only evidence",
        "baseline_script_sha256": hashlib.sha256((BASE / "quality_probe.py").read_bytes()).hexdigest(),
        "baseline_results_sha256": hashlib.sha256((BASE / "results.json").read_bytes()).hexdigest(),
        "tiger_py_sha256": hashlib.sha256((probe["REPO"] / "src/tiger_optim/tiger.py").read_bytes()).hexdigest(),
        "sole_config_change": {"agc_clip": {"baseline": 0.02, "ablation": 0.0}},
        "torch_default_device": str(torch.get_default_device()),
        "seeds": [],
    }
    for seed in probe["SEEDS"]:
        entry = run(seed)
        base_entry = next(item for item in baseline["seeds"] if item["seed"] == seed)
        base_tiger = next(item for item in base_entry["runs"] if item["optimizer"] == "Tiger")
        adamw = next(item for item in base_entry["runs"] if item["optimizer"] == "AdamW")
        entry["baseline_tiger_final_heldout_mse"] = base_tiger["trace"][-1]["heldout_mse"]
        entry["adamw_final_heldout_mse"] = adamw["trace"][-1]["heldout_mse"]
        result["seeds"].append(entry)
        print(json.dumps({
            "seed": seed,
            "tiger_no_agc_final_heldout_mse": entry["trace"][-1]["heldout_mse"],
            "tiger_baseline_final_heldout_mse": entry["baseline_tiger_final_heldout_mse"],
            "adamw_final_heldout_mse": entry["adamw_final_heldout_mse"],
        }, sort_keys=True), flush=True)
    (BASE / "ablation_results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("results=" + str(BASE / "ablation_results.json"), flush=True)


if __name__ == "__main__":
    main()
