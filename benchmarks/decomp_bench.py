#!/usr/bin/env python3
"""Feature-decomposition micro-benchmark for Tiger's opt.step.

Measures the marginal cost of each adaptive feature by toggling one knob at
a time. Useful for localizing dispatch-bound hotspots before optimizing.

Diagnostic only — this harness is NOT claim-gate evidence (see README
"Benchmark and Claim Gate"): it runs in-process, persists no raw JSON, and
prints only stdout medians.
"""
import argparse
import statistics
import sys
import time

sys.path.insert(0, "src")
import torch

from tiger_optim import Tiger, build_tagged_param_groups


class TinyMix(torch.nn.Module):
    def __init__(self, d=256, ff=512, heads=4):
        super().__init__()
        self.mha = torch.nn.MultiheadAttention(d, heads, batch_first=True)
        self.up = torch.nn.Linear(d, ff)
        self.down = torch.nn.Linear(ff, d)
        self.ln = torch.nn.LayerNorm(d)

    def forward(self, x):
        h, _ = self.mha(x, x, x)
        h = torch.nn.functional.gelu(self.up(h))
        return self.ln(self.down(h))


def sync(dev):
    if dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif dev.type == "mps":
        try:
            torch.mps.synchronize()
        except Exception:
            pass


def run(label, dev, steps=100, **kw):
    torch.manual_seed(0)
    model = TinyMix().to(dev)
    groups = build_tagged_param_groups(model, base_lr=3e-4, base_wd=0.01, enable_qkv_slicing=True)
    base = dict(factored=True, precond_alpha=1.0, trust_space="precond", trust_ema_beta=0.9,
                use_foreach_update=True, bucket_standardize=True, bucket_scalarless=True,
                qkv_lr_autoadapt=True, lora_density_adapt=True)
    base.update(kw)
    opt = Tiger(groups, **base)
    x = torch.randn(16, 32, 256, device=dev)
    y = torch.randn(16, 32, 256, device=dev)
    lf = torch.nn.MSELoss()

    def it():
        opt.zero_grad(set_to_none=True)
        loss = lf(model(x), y)
        loss.backward()
        sync(dev)
        opt.step()
        sync(dev)

    for _ in range(30):
        it()
    ts = []
    for _ in range(steps):
        t = time.perf_counter()
        opt.step()
        sync(dev)
        ts.append((time.perf_counter() - t) * 1000)
        opt.zero_grad(set_to_none=True)
        loss = lf(model(x), y)
        loss.backward()
        sync(dev)
    print(f"{label:38s} opt.step {statistics.median(ts):7.3f} ms")


def main():
    ap = argparse.ArgumentParser("Tiger feature-decomposition bench")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=100)
    args = ap.parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    dev = torch.device(args.device)

    # NOTE: "v21 bench cfg" matches bench_compare_optim.py's `common` dict,
    # not Tiger's constructor defaults.
    run("v21 bench cfg (full)", dev, args.steps)
    run("skip_if_nonfinite=False", dev, args.steps, skip_if_nonfinite=False)
    run("use_trust_ratio=False", dev, args.steps, use_trust_ratio=False)
    run("factored=False", dev, args.steps, factored=False)
    # NOTE: qkv_lr_autoadapt also gates the spectral work, so this row is not
    # an independent single-knob delta.
    run("no qkv autoadapt", dev, args.steps, qkv_lr_autoadapt=False)
    # NOTE: TinyMix has no lora_a/lora_b params; this toggles near-zero work.
    run("no lora density", dev, args.steps, lora_density_adapt=False)
    # NOTE: spectral work fires only every qkv_lr_interval=25 steps, so a
    # median mostly hides its cost.
    run("no spectral adapt", dev, args.steps, qkv_spectral_adapt=False)
    # NOTE: "lean" still keeps bucket_standardize, precond trust space, and
    # trust_ema_beta from the bench cfg — it is not Tiger's bare defaults.
    run("lean (guards+trust+adapt off)", dev, args.steps, skip_if_nonfinite=False,
        use_trust_ratio=False, factored=False, qkv_lr_autoadapt=False,
        lora_density_adapt=False, qkv_spectral_adapt=False, agc_clip=0.0,
        auto_ffn_asym=False, lora_cross_adapt=False, lora_bridge=False)


if __name__ == "__main__":
    main()
