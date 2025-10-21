# tiger-optim
Tiger Optimizer — beyond AdamW. A precision beast born from pure Python and stubborn curiosity — LoRA-aware, QKV-adaptive, trust-ratio-driven 🐅

<p align="center">
  <img src="https://img.shields.io/badge/License-AGPL--3.0-blue.svg" alt="AGPL-3.0">
  <a href="docs/PRICING.md"><img src="https://img.shields.io/badge/Commercial%20License-Available-orange.svg" alt="Commercial License Available"></a>
  <img src="https://img.shields.io/badge/Apple%20Silicon-MPS%20Verified-success.svg" alt="MPS Verified">
  <img src="https://img.shields.io/badge/PyTorch-2.x-lightgrey.svg" alt="PyTorch 2.x">
  <a href="issues?q=label%3Abenchmark"><img src="https://img.shields.io/badge/Benchmarks-help%20wanted-brightgreen.svg" alt="Benchmarks: help wanted"></a>
</p>

<h1 align="center">Tiger Optimizer</h1>
<p align="center"><i>Sign‑aware, trust‑ratio, LoRA‑PID with inertia — precise like a tiger.</i></p>

> **Bold, evidence‑based.** Tiger is verified on Apple’s MPS backend (real hardware).  
> On our Mac microbench (TinyMix; 120 steps), Tiger v2.1 (full) achieved a median of
> <b>8.50 ms/step (CPU)</b> and <b>26.84–35.53 ms/step (MPS)</b> with convergence comparable to AdamW.  
> On Windows + CUDA (legacy GPU), Tiger v2.1 (full) measured <b>14.8–15.0 ms/step</b> (median).  
> PNG plots below.

---

## Contents
- [Install](#install)
- [Quickstart](#quickstart)
- [Reference Bench (CPU/MPS/CUDA)](#reference-bench-cpumpscuda)
- [System Info for This CUDA Run (Legacy Reference)](#system-info-for-this-cuda-run-legacy-reference)
- [Call for Community CUDA Runs](#call-for-community-cuda-runs)
- [Pricing & Licensing](#pricing--licensing)
- [Known Good Settings](#known-good-settings)
- [Legacy CUDA: Quick Preset](#legacy-cuda-quick-preset)
- [Roadmap & Lessons from Legacy GPUs](#roadmap--lessons-from-legacy-gpus)

---

## Install

```bash
# dev install from this repo
pip install -e .
# or (future)
# pip install tiger-optim
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
    # premium toggles are automatically no-op on AGPL build
)

x = torch.randn(16, 32, 256)
y = torch.randn(16, 32, 256)
loss_fn = nn.MSELoss()

opt.zero_grad(set_to_none=True)
loss = loss_fn(model(x), y)
loss.backward()
opt.step()
```

---

## Acceleration: TorchScript fused primitives

Tiger ships pure-PyTorch kernels for its softsign, RMS and vector-norm
operations. When TorchScript is available (it ships with PyTorch by default)
the kernels are scripted once at import time, producing fused graphs that avoid
Python dispatch and scalar argument boxing.

The accelerated helpers fall back to the eager implementations automatically.
You can confirm what is active on your system:

```python
from tiger_optim import available_backends
print(available_backends())  # e.g. {"torchscript_softsign": True, ...}
```

TorchScript works across CPU, CUDA and MPS backends without additional build
steps, so the optimization path is now completely dependency free.

---

## Reference Bench (CPU/Mac, MPS/Mac, CUDA/Win)

**TinyMix; 120 steps (warmup 30–50).**  

| Device | Optimizer | Median step time |
|-------:|:---------:|-----------------:|
| CPU (Mac) | Tiger v2.1 (full) | **8.50 ms** |
| MPS (Mac) | Tiger v2.1 (full) | **26.84–35.53 ms** |
| CUDA (Win, GTX 1650 / CUDA 11.1) | AdamW | **5.43–6.69 ms** |
| CUDA (Win, GTX 1650 / CUDA 11.1) | Tiger v2.1 (full) | **14.8–15.0 ms** |

<p align="center">
  <img src="bench/plots/median_step_time.png" width="60%" alt="AdamW vs Tiger — Median Step Time">
  <br/>
  <img src="bench/plots/loss_curves.png" width="60%" alt="Loss Curves">
</p>

> Notes:  
> • MPS can be slower on small batches due to launch/transfer overheads; larger batch/T/D typically improves.  
> • The CUDA numbers above are **legacy GPU** reference values (see next section).

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
   python bench/bench_compare_optim.py --device cuda --steps 200 --warmup 50
   python bench/plot_bench.py --out-dir bench/plots
   ```
2. Collect and attach:
   - `bench/results/compare-*.json` (AdamW + Tiger)
   - `bench/plots/median_step_time.png`, `bench/plots/loss_curves.png`
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

## Pricing & Licensing

## License
SpiralReality and its components are licensed under the  
[GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)](https://www.gnu.org/licenses/agpl-3.0.html).  
© 2025 Ryo ∴ SpiralArchitect and SpiralReality. for the public build.  
**Commercial License** available for proprietary integration; see `docs/PRICING.md`.

### Pricing (Annual) — Conservative (current)
| Tier | Rights & Scope | Price | Support |
|---|---|---:|---|
| Growth | Single org, ≤1 product, ≤10 seats, internal use | **$3,000** | Std (email) |
| Pro | Single org, ≤3 products, ≤40 seats, internal + offline eval | **$9,000** | Priority |
| Enterprise | Org‑wide, unlimited seats/products, internal | **$25,000** | Priority+ |
| OEM | Redistribution/embedding in shipped products or SaaS | **$75,000 + royalty** | Premier |

Royalty (OEM): 0.5% Tiger‑attributable GTV **or** $0.05/MAU (higher of), floor **$50k/yr**, cap **$250k/yr**.  
Premium (Commercial): LoRA‑PID inertia + minima/recovery, QKV dual‑objective auto‑LR (γ auto‑scale + accel clip), scalarless foreach + Triton stats, pending arithmetic pipelines, Auto‑FFN asym.  
Early adopters: first 10 customers −25% (year 1).  
Contact: **kishkavsesvit@icloud.com**

> If community CUDA runs show Tiger ≥ AdamW (median) on modern GPUs by ≥5%, we’ll switch to **Assertive** pricing (Pro=$12k / Enterprise=$30k / OEM=$90k+royalty).

---

## Known Good Settings

- **Stability first (any device)**  
  ```python
  Tiger(..., update_buffer_dtype="fp32", lr=2e-4, agc_clip=0.02, trust_clip=5.0,
        rms_clip_threshold=1.0, rms_clip_granularity="param")
  ```
- **MPS (Apple Silicon)**: keep Triton flags off; prefer FP32 update buffer.  
- **CUDA (modern)**: try `use_foreach_update=True`, `bucket_standardize=True`, and Triton stats if available.

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
