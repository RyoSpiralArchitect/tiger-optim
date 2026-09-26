#!/usr/bin/env python3
"""Attribute MPS scalar host reads during three TinyMix Tiger steps."""

from __future__ import annotations

import collections
import hashlib
import inspect
import json
import linecache
import sys
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks"))
from bench_compare_optim import TinyMix, build_optimizer, sync  # noqa: E402


class ScalarReads(TorchDispatchMode):
    def __init__(self):
        self.lines = collections.Counter()
        self.total = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func is torch.ops.aten._local_scalar_dense.default:
            self.total += 1
            frame = inspect.currentframe()
            try:
                while frame is not None:
                    if frame.f_code.co_filename == str(ROOT / "src/tiger_optim/tiger.py"):
                        line = frame.f_lineno
                        self.lines[f"{line}: {linecache.getline(frame.f_code.co_filename, line).strip()}"] += 1
                        break
                    frame = frame.f_back
                else:
                    self.lines["<outside tiger.py>"] += 1
            finally:
                del frame
        return func(*args, **(kwargs or {}))


def main():
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")
    torch.manual_seed(0)
    dev = torch.device("mps")
    model = TinyMix(d=256, ff=512, heads=4).to(dev)
    x = torch.randn(16, 32, 256, device=dev)
    y = torch.randn(16, 32, 256, device=dev)
    loss_fn = torch.nn.MSELoss()
    opt = build_optimizer(model, "tiger_v21_full", dev)
    rows = []
    for step in range(1, 27):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        sync(dev)
        opt.report_metrics(float(loss.detach().item()))
        if step in (24, 25, 26):
            with ScalarReads() as reads:
                opt.step()
            sync(dev)
            rows.append({"step": step, "total": reads.total, "lines": dict(reads.lines.most_common())})
        else:
            opt.step()
            sync(dev)
    source = ROOT / "src/tiger_optim/tiger.py"
    print(json.dumps({"torch": torch.__version__, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                      "rows": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
