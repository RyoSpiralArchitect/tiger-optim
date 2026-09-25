import json
import platform

import torch


def current(tensors):
    checks = [torch.isfinite(tensor).all() for tensor in tensors]
    return bool(torch.stack(checks).all().item())


def reduced(tensors):
    extrema = [tensor.abs().amax() for tensor in tensors]
    return bool(torch.isfinite(torch.stack(extrema).amax()).item())


def count_ops(fn, tensors, repeats=3):
    for _ in range(2):
        fn(tensors)
    torch.mps.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
        for _ in range(repeats):
            fn(tensors)
    torch.mps.synchronize()
    return {
        key: next((event.count for event in profiler.key_averages() if event.key == key), 0)
        for key in ("aten::_local_scalar_dense", "aten::isfinite", "aten::amax", "aten::all")
    }


torch.manual_seed(0)
device = torch.device("mps")
payload = {
    "torch": torch.__version__,
    "os": platform.platform(),
    "device": "mps",
    "cases": {},
    "profile": {},
}
for dtype in (torch.float32, torch.float16, torch.bfloat16):
    base = [torch.randn(64, 64, device=device, dtype=dtype) for _ in range(4)]
    cases = {}
    for name, value in (("finite", 1.0), ("nan", float("nan")), ("pos_inf", float("inf")), ("neg_inf", float("-inf"))):
        tensors = [t.clone() for t in base]
        tensors[2][0, 0] = value
        cases[name] = {"current": current(tensors), "reduced": reduced(tensors)}
    payload["cases"][str(dtype)] = cases
    payload["profile"][str(dtype)] = {
        "current": count_ops(current, base),
        "reduced": count_ops(reduced, base),
    }
print(json.dumps(payload, indent=2))
