# Tiger Optimizer
Tiger is a PyTorch optimizer exploring sign-aware updates, trust ratios, and
LoRA/QKV adaptation. 🐅

<p align="center">
  <img src="https://img.shields.io/badge/License-AGPL--3.0-blue.svg" alt="AGPL-3.0">
  <a href="#pricing--licensing"><img src="https://img.shields.io/badge/Commercial%20License-Available-orange.svg" alt="Commercial License Available"></a>
  <img src="https://img.shields.io/badge/PyTorch-2.x-lightgrey.svg" alt="PyTorch 2.x">
  <a href="issues?q=label%3Abenchmark"><img src="https://img.shields.io/badge/Benchmarks-help%20wanted-brightgreen.svg" alt="Benchmarks: help wanted"></a>
</p>
<p align="center"><i>Sign‑aware, trust‑ratio, LoRA‑PID with inertia — precise like a tiger.</i></p>

The timing figures below are historical local notes for Tiger v2.1. They are
not a performance or convergence claim for this checkout; the corresponding raw
results and environment records are not tracked in this repository.

---

## Contents
- [Install](#install)
- [Quickstart](#quickstart)
- [Historical Local Timing Notes](#historical-local-timing-notes)
- [Benchmark and Claim Gate](#benchmark-and-claim-gate)
- [System Info for This CUDA Run (Legacy Reference)](#system-info-for-this-cuda-run-legacy-reference)
- [Call for Community CUDA Runs](#call-for-community-cuda-runs)
- [Pricing & Licensing](#pricing--licensing)
- [Experimental Starting Settings](#experimental-starting-settings)
- [Legacy CUDA: Quick Preset](#legacy-cuda-quick-preset)
- [Roadmap & Lessons from Legacy GPUs](#roadmap--lessons-from-legacy-gpus)

---

## Install

```bash
# dev install from this repo
pip install -e .

# editable install with tests/benchmark tooling
pip install -e ".[dev]"

# add the optional Julia acceleration bridge
pip install -e ".[julia]"

# or grab everything most contributors want
pip install -e ".[dev,julia,bench]"

# published package
# pip install tiger-optim
# pip install "tiger-optim[julia,bench]"
```

> Tiger Optimizer is released under **GNU AGPL‑3.0**.  
> **Commercial licenses** (OEM/Enterprise) are available for proprietary integration.

---

## Quickstart

```python
import torch, torch.nn as nn
from tiger_optim import Tiger, build_tagged_param_groups

# tiny demo model
class TinyMix(nn.Module):
    def __init__(self, d=256, ff=512, heads=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(d, heads, batch_first=True)
        self.up  = nn.Linear(d, ff)
        self.down= nn.Linear(ff, d)
        self.ln  = nn.LayerNorm(d)
    def forward(self, x):
        h,_ = self.mha(x,x,x)
        h = torch.nn.functional.gelu(self.up(h))
        h = self.down(h)
        return self.ln(h)

model = TinyMix()
groups = build_tagged_param_groups(model, base_lr=3e-4, base_wd=0.01, enable_qkv_slicing=True)

opt = Tiger(
    groups,
    # modern default
    factored=True, precond_alpha=1.0, trust_space="precond",
    use_foreach_update=True, bucket_standardize=True, bucket_scalarless=True,
)

x = torch.randn(16, 32, 256)
y = torch.randn(16, 32, 256)
loss_fn = nn.MSELoss()

opt.zero_grad(set_to_none=True)
loss = loss_fn(model(x), y)
loss.backward()
opt.step()
```

LoRA/QKV adaptation and scalarless foreach options in this repository execute
when enabled; this public build has no license-based no-op path for them.

### Reporting loss and resuming training

The plateau-based `auto_lr` and `auto_blend` controls act only after the training
loop calls `opt.report_metrics(loss=...)`. With `auto_lr=True`, a plateau of
`plateau_patience` reported finite losses multiplies each group LR by
`lr_decay`, down to `lr_min`. These controls run in eager mode.

Save both the model and optimizer to resume the blend schedule and adaptation
state. Recreate the optimizer with parameter groups in the same order before
loading:

```python
torch.save({"model": model.state_dict(), "optimizer": opt.state_dict()}, "checkpoint.pt")

# In a later process, after recreating the model and its tagged groups:
checkpoint = torch.load("checkpoint.pt", map_location="cpu")
model.load_state_dict(checkpoint["model"])
opt = Tiger(build_tagged_param_groups(model, base_lr=3e-4, base_wd=0.01))
opt.load_state_dict(checkpoint["optimizer"])
```

Checkpoints made before Tiger stored adaptive state cannot resume exactly. The
loader warns about this and needs fresh QKV rules from the new model when the
old checkpoint contains QKV groups.

With FP16 or BF16 parameters and `skip_if_nonfinite=True`, median bucket
standardization is rejected before updating. Use the `global` or `mean` bucket
source for low-precision parameters.

---

## Acceleration toolchain (Pure Python core + optional Julia)

Tiger v2.4.0 keeps the optimizer entirely in Python while still offering two
runtime acceleration paths:

- `torch`: a real backend built on native PyTorch kernels that works on CPU,
  MPS, and CUDA devices without extra dependencies.
- `julia`: an optional backend for CPU-centric primitives (softsign, RMS and
  vector-norm). When Julia ≥1.9 and the `juliacall` bridge are present Tiger
  can dispatch to Julia loops.

The runtime still profiles every backend invocation and automatically reorders
the priority list to favour the fastest healthy implementation. Backends that
raise errors (or return `None`) are temporarily suppressed so later calls can
fall back to another backend or the PyTorch reference path.

Runtime selection is automatic. You can inspect the availability at runtime:

```python
from tiger_optim import available_backends, current_backend_priority
print(available_backends())  # e.g. {"julia": False, "torch": True}
print(current_backend_priority())  # runtime ordering after scoring
```

### Runtime configuration

Tiger now exposes lightweight runtime controls so you can experiment without
restarting your notebook or script:

```python
from tiger_optim import (
    available_backends,
    backend_diagnostics,
    configure_backends,
    current_backend_priority,
    refresh_backend_state,
    reset_backend_configuration,
)

# Prefer Julia for the current process, then the torch backend
configure_backends(preferred=["julia", "torch"])

# Prefer the device-native torch backend on MPS/CUDA
configure_backends(preferred=["torch"])

# Disable all native accelerators (forces the PyTorch eager path)
configure_backends(disabled=["all"])

# Clear runtime overrides and reset performance history, e.g. after rebuilding a backend
reset_backend_configuration()
refresh_backend_state(reload=True, reset_metrics=True)
print(available_backends())
print(current_backend_priority())
print(backend_diagnostics())
```

Prefer environment variables? Set them before import:

```bash
export TIGER_ACCEL_DISABLE=all       # disable all accelerators
export TIGER_ACCEL_PREFER=julia,torch     # prefer Julia when available
```

Both signals are lazily cached, so changes made at runtime can be picked up via
`refresh_backend_state()`.

If no accelerator is available Tiger falls back to the stock PyTorch
implementations, so you can opt-in incrementally.

---

## Historical Local Timing Notes

Earlier local TinyMix notes recorded the following medians during Tiger v2.1
comparisons, using 120 steps and 30–50 warmup steps. The raw JSON, plots,
exact source revision, and full environment records for these rows are not
tracked here.
These figures cannot establish a current speed or convergence comparison.

| Device | Optimizer | Median step time |
|-------:|:---------:|-----------------:|
| CPU (Mac) | Tiger v2.1 (full) | **8.50 ms** |
| MPS (Mac) | Tiger v2.1 (full) | **26.84–35.53 ms** |
| CUDA (Win, GTX 1650 / CUDA 11.1) | AdamW | **5.43–6.69 ms** |
| CUDA (Win, GTX 1650 / CUDA 11.1) | Tiger v2.1 (full) | **14.8–15.0 ms** |

The CUDA entries came from a legacy GPU and driver (see below). Measure on the
target workload and device before drawing a performance conclusion.

## Benchmark and Claim Gate

For a new result, record the exact commit and dirty state, Python/PyTorch and
device/driver versions, seed, command, warmup, and step count. Run AdamW and
Tiger on the same seeded model and data, repeat each mode in at least three
fresh processes, and retain every raw JSON file. Use a clean result set for
each comparison: the plotter reads all matching files, while the summarizer
selects the latest file per device and mode rather than measuring variation.
Publish all raw files (with SHA-256 hashes), the spread across runs, and plots
alongside any timing claim. A convergence claim also needs a defined
quality metric, a fixed compute budget, and multiple seeds; the current bench
CLI is a smoke/timing tool and does not establish that claim.

The [2026-09-25 CPU smoke](benchmarks/evidence/2026-09-25-cpu-smoke/README.md)
preserves three fresh-process pairs with raw JSON and hashes. On that fixed
synthetic workload, Tiger took 8.97 ms per measured compute step versus 4.05 ms
for AdamW, and ended with a higher training loss. The configurations differ and
the runs repeat one seed, so this is a local negative result rather than a
general comparison.

The [2026-09-26 Mac diagnostic](benchmarks/evidence/2026-09-26-mac-perf/README.md)
preserves CPU/MPS raw runs, profiler traces, source hashes, and the host-load
caveat. Consolidating finite checks reduced profiled MPS scalar reads inside
Tiger from 68 to 30 per step at the recorded intermediate source revisions;
the final source measured 36 per step after the FP32 overflow guard.
In the final paired MPS smoke, Tiger's optimizer median was 23.95 ms versus
0.998 ms for AdamW. The Tiger global-step-25 spike remains unresolved, and
these runs do not establish a wall-clock speedup or a convergence comparison.

Run `python benchmarks/bench_quality_smoke.py` for a separate CPU toy task with
held-out data and three seeds. Its Tiger recipe uses a cosine LR schedule while
the AdamW reference uses a fixed LR, so its output checks learning rather than
ranking the optimizers; the archived traces are
[here](benchmarks/evidence/2026-09-25-cpu-quality/README.md).

```bash
git rev-parse HEAD
git status --short
python -c 'import platform, torch; print(platform.platform()); print(torch.__version__)'
for run in 1 2 3; do
  python benchmarks/bench_compare_optim.py --device cpu --steps 200 --warmup 50 --modes adamw tiger_v21_full
done
python benchmarks/summarize_results.py --pattern 'benchmarks/results/compare-*.json' --markdown-out benchmarks/results/summary.md
shasum -a 256 benchmarks/results/compare-*.json
```

Use the wildcard above only when `benchmarks/results` contains the current
comparison. Run the same command on MPS/CUDA only when that device is available.

---

## System Info for This CUDA Run (Legacy Reference)

- OS: Windows  
- GPU: **GeForce GTX 1650** (Turing, **4 GB**, **SM 7.5**)  
- NVIDIA Driver: **457.49**  
- CUDA reported by `nvidia-smi`: **11.1**  
- Notes: legacy hardware (no TF32) with older driver/runtime; some fused/foreach paths may not be effective.

---

## Call for Community CUDA Runs

We’d love **fresh results on modern GPUs** (Ampere/Ada/Hopper; CUDA 11.8+/12.x).

**How to contribute**
1. Run:
   ```bash
   git rev-parse HEAD
   git status --short
   python benchmarks/bench_compare_optim.py --device cuda --steps 200 --warmup 50
   python benchmarks/plot_bench.py --out-dir benchmarks/plots
   python benchmarks/summarize_results.py --pattern "benchmarks/results/compare-*.json" --markdown-out benchmarks/results/summary.md
   ```
2. Collect and attach:
   - `benchmarks/results/compare-*.json` (AdamW + Tiger)
   - `benchmarks/results/summary.md`
   - `benchmarks/plots/median_step_time.png`, `benchmarks/plots/loss_curves.png`
   - SHA-256 hashes for the raw JSON and the exact source commit/dirty state
   - Environment info:
     ```
     nvidia-smi
     python - <<'PY'
     import torch,sys
     print('torch=', torch.__version__)
     print('torch.cuda(build)=', torch.version.cuda)
     print('cuDNN=', torch.backends.cudnn.version())
     print('GPU=', torch.cuda.get_device_name(0))
     print('SM=' + '.'.join(map(str, torch.cuda.get_device_capability(0))))
     PY
     ```
3. Open a GitHub Issue titled  
   **Benchmark: &lt;GPU model&gt; (CUDA &lt;build&gt;, Driver &lt;ver&gt;)**  
   We’ll **credit contributors** in the README.

---

## Experimental Starting Settings

- **Conservative clipping experiment.** This configuration keeps the update
  buffer in FP32, but its AGC setting can make learning very slow. Validate the
  actual update size and held-out loss on your task. RMS clipping is a parameter
  group option:
  ```python
  groups = [{"params": model.parameters(),
             "rms_clip_threshold": 1.0, "rms_clip_granularity": "param"}]
  opt = Tiger(groups, update_buffer_dtype="fp32", lr=2e-4,
              agc_clip=0.02, trust_clip=5.0)
  ```
  In a [three-seed held-out CPU probe](benchmarks/evidence/2026-09-25-cpu-quality/README.md),
  this setting barely learned the test task; treat it as a clipping experiment,
  not a validated quality preset.
- **MPS (Apple Silicon)**: keep Triton flags off; prefer FP32 update buffer.  
- **CUDA (modern)**: try `use_foreach_update=True`, `bucket_standardize=True`, and Triton stats if available.

### Profiling `opt.step()` on MPS

Use the built-in profiler harness when you want to see where Apple Silicon time is
really going:

```bash
python benchmarks/bench_profile_v21.py \
  --device mps \
  --steps 4 \
  --warmup 2 \
  --torch-profiler \
  --profile-steps 3
```

The profiler output can identify operations to investigate. Preserve the trace,
table, source revision, and a paired baseline before reporting an improvement.

---

## Legacy CUDA: Quick Preset

For Turing‑class / older drivers (e.g., GTX 1650, CUDA 11.1), use a leaner path:

```python
Tiger(
  groups,
  factored=False, precond_alpha=0.0,      # lighten preconditioning
  use_trust_ratio=False,                  # cut extra norms
  use_foreach_update=False,               # avoid foreach overhead
  bucket_standardize=False, bucket_scalarless=False,
  update_buffer_dtype="fp32",
  lr=2e-4, agc_clip=0.02, trust_clip=5.0
)
```

---

## Roadmap & Lessons from Legacy GPUs

From our GTX 1650 (Driver 457.49 / CUDA 11.1) measurements:

1. **Auto‑Preset by Capability/Driver**  
   - Detect `SM` & driver/runtime at init and choose **`preset="modern"` / `preset="legacy"`**.  
   - Gate foreach/bucketization/Triton paths and preconditioning strength automatically.

2. **Lean Path for WDDM / Older Drivers**  
   - Provide an **in‑place fused update** without bucketization; minimize tensor re‑reads.  
   - Prefer FP32 update buffers, lighter trust math, optional AGC only.

3. **Convergence Guard on Legacy**  
   - If loss plateaus >N steps and `Δloss≈0`, auto‑toggle `use_sign=False` and reduce WD;  
     re‑enable gradually once descent is detected.

4. **Minimal‑alloc Foreach**  
   - On CC ≤7.5, bypass scalarless stats; avoid small kernel storms; coalesce tiny params.

5. **Debug Hooks**  
   - `opt.debug_check()` to log per‑group norms (p/m/update), non‑finite counts, and effective LR/trust (device‑safe).

6. **Docs**  
   - A dedicated **“Legacy CUDA Playbook”** with presets, known gotchas (WDDM, driver 45x/46x), and validation checklist.

These items will land as: `Tiger(..., preset="auto")` with internal feature gating; a `--legacy` flag in benches; and a **single‑kernel apply** path for legacy devices.
