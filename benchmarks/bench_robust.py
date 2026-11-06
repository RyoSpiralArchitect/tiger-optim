#!/usr/bin/env python3
"""Robust Tiger opt.step micro-benchmark.

Design goals:
- diagnose variance: per-step series, interquartile stats, GPU clock check
- repeated in-process measurement with median-of-medians
"""
import sys, time, statistics, json, subprocess
sys.path.insert(0, "src")
import torch
from tiger_optim import Tiger, build_tagged_param_groups


class TinyMix(torch.nn.Module):
    def __init__(self, d=256, ff=512, heads=4):
        super().__init__()
        self.mha = torch.nn.MultiheadAttention(d, heads, batch_first=True)
        self.up = torch.nn.Linear(d, ff); self.down = torch.nn.Linear(ff, d)
        self.ln = torch.nn.LayerNorm(d)
    def forward(self, x):
        h, _ = self.mha(x, x, x)
        h = torch.nn.functional.gelu(self.up(h))
        return self.ln(self.down(h))


def gpu_info():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,clocks.max.sm,temperature.gpu,power.draw,utilization.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip()
    except Exception as e:
        return f"nvidia-smi failed: {e}"


def sync(dev):
    if dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif dev.type == "mps":
        try:
            torch.mps.synchronize()
        except Exception:
            pass


def measure(opt, model, lf, x, y, steps, dev, tag=""):
    def step_once():
        opt.zero_grad(set_to_none=True)
        loss = lf(model(x), y)
        loss.backward()
        sync(dev)
        t = time.perf_counter()
        opt.step()
        sync(dev)
        return (time.perf_counter() - t) * 1000.0

    for _ in range(30):
        step_once()
    ts = [step_once() for _ in range(steps)]
    ts_s = sorted(ts)
    n = len(ts_s)
    q = lambda f: ts_s[int(f * (n - 1))]
    return {
        "tag": tag, "steps": steps,
        "median": statistics.median(ts), "mean": statistics.mean(ts),
        "p10": q(0.10), "p25": q(0.25), "p75": q(0.75), "p90": q(0.90),
        "min": ts_s[0], "max": ts_s[-1],
        "stdev": statistics.pstdev(ts),
    }


def build(mode, model):
    groups = build_tagged_param_groups(model, base_lr=3e-4, base_wd=0.01, enable_qkv_slicing=True)
    common = dict(factored=True, precond_alpha=1.0, trust_space="precond", trust_ema_beta=0.9,
                  use_foreach_update=True, bucket_standardize=True, bucket_scalarless=True,
                  qkv_lr_autoadapt=True, lora_density_adapt=True)
    if mode == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.999), weight_decay=0.01)
    return Tiger(groups, **common)


def main():
    import argparse
    ap = argparse.ArgumentParser("Tiger opt.step robust bench")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    dev = torch.device(args.device)
    torch.manual_seed(0)
    if dev.type == "cuda":
        print("gpu:", gpu_info())
    results = []
    for rep in range(args.reps):
        torch.manual_seed(0)
        model = TinyMix().to(dev)
        x = torch.randn(16, 32, 256, device=dev)
        y = torch.randn(16, 32, 256, device=dev)
        lf = torch.nn.MSELoss()
        opt = build("tiger_v21_full", model)
        r = measure(opt, model, lf, x, y, steps=args.steps, dev=dev,
                    tag=f"tiger_v21_full rep{rep}")
        results.append(r)
        print(json.dumps(r))
        del opt, model
        torch.cuda.empty_cache()
    print("gpu:", gpu_info())
    meds = [r["median"] for r in results]
    print("median-of-medians:", statistics.median(meds), "spread:", min(meds), "-", max(meds))


if __name__ == "__main__":
    main()
