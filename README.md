# Tiger Optimizer 🐅

**A PyTorch optimizer for experiments with sign-aware updates, trust ratios,
and adaptive Q/K/V and LoRA controls.**

[![CI](https://github.com/RyoSpiralArchitect/tiger-optim/actions/workflows/ci.yml/badge.svg)](https://github.com/RyoSpiralArchitect/tiger-optim/actions/workflows/ci.yml)
[![License: AGPL-3.0-only](https://img.shields.io/badge/License-AGPL--3.0--only-blue.svg)](LICENSE.txt)

Tiger is research software. Its update rules and checkpoint behavior have
regression coverage, and its learning experiments retain raw results and source
hashes. An adaptive feature being implemented does not establish a learning
advantage; the current evidence includes negative results.

## Install

From a source checkout, with a PyTorch build appropriate for your device:

```sh
git clone https://github.com/RyoSpiralArchitect/tiger-optim.git
cd tiger-optim
python -m pip install -e .
# Tests and benchmark tools:
python -m pip install -e '.[dev]'
```

Package metadata: **2.4.0**, Python **3.9+**, PyTorch **1.13+**. The recall
benchmark needs **PyTorch 2.0+** for scaled dot-product attention; CUDA BF16
runs also need compatible hardware. Julia is optional (`pip install -e '.[julia]'`).
No license-based feature gates are present in this public build.

## Start training

This example exercises tagging and a real optimizer step. Its LR illustrates
the API; it is not a validated preset for your model.

```python
import torch
from torch import nn
from tiger_optim import Tiger, build_tagged_param_groups

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(
        d_model=128, nhead=4, dim_feedforward=512,
        dropout=0.0, batch_first=True,
    ),
    num_layers=2,
).to(device)
groups = build_tagged_param_groups(model, base_lr=3e-4, base_wd=0.01)
optimizer = Tiger(
    groups, trust_space="precond", trust_clip=5.0,
    update_buffer_dtype="fp32", auto_lr=False, auto_blend=False,
)
x = torch.randn(8, 32, 128, device=device)
target = torch.randn_like(x)
optimizer.zero_grad(set_to_none=True)
loss = (model(x) - target).square().mean()
loss.backward()
optimizer.step()
```

CPU, MPS, and CUDA use native PyTorch operations. Move the model and inputs to
the desired device before constructing the optimizer. Inspect tagged groups
when using custom module names:

```python
from tiger_optim import summarize_param_groups
print(summarize_param_groups(optimizer.param_groups))
```

## What Tiger adds

| Component | Behavior | Main controls |
| --- | --- | --- |
| Sign-aware direction | Blends sign/softsign with normalized momentum and optional factored preconditioning | `sign_mode`, `sign_blend`, `factored`, `precond_alpha` |
| Trust ratios | Scales updates relative to parameter magnitude; fused QKV can use separate slice ratios | `use_trust_ratio`, `trust_space`, `trust_clip`, `qkv_trust_split` |
| Tagged groups | Assigns LR scales and decay rules to attention, embeddings, norms, FFNs, and LoRA parameters | `build_tagged_param_groups`, `tag_overrides`, `lora_overrides` |
| QKV adaptation | Adjusts slice LR multipliers from RMS/trust statistics, with optional spectral feedback | `qkv_lr_autoadapt`, `qkv_lr_interval`, `qkv_lr_gain`, `qkv_spectral_adapt` |
| LoRA adaptation | Uses density feedback, PID state, and inertia to adjust blend/clipping | `lora_density_adapt`, `lora_pid_*`, `lora_cross_adapt`, `lora_bridge` |
| Schedules and metrics | Fixed-budget tail cosine, tag warmup/decay, or reported-loss plateau controls | `TailCosineLR`, `TagWarmupDecay`, `report_metrics` |

The [constructor](src/tiger_optim/tiger.py) defines the complete option set.
QKV and LoRA mechanisms need correctly tagged parameters. LoRA adaptation has
correctness coverage but no learning advantage established by the current
retrieval experiments, which contain no LoRA adapters.

### Control Q, K, and V

For a fused QKV slice, the LR before trust and clipping is:

```text
group["lr"] × group["lr_scale"] × group["qkv_lr_scales"].get(slice, 1.0)
```

The tag builder starts Q/K/V multipliers at **0.9 / 0.8 / 1.1**. Group and slice
controls compose, including staged updates:

```python
groups = build_tagged_param_groups(
    model, base_lr=3e-4,
    tag_overrides={"attn_qkv": {"lr_scale": 0.5}},
)
optimizer = Tiger(groups, auto_lr=False, auto_blend=False)
qkv_index = next(i for i, g in enumerate(optimizer.param_groups)
                 if g.get("block_tag") == "attn_qkv")
optimizer.stage_group_update(qkv_index, {"lr_scale": 0.0})
optimizer.reflect_pending()
```

A zero group scale stops parameter updates and weight decay; moments and
adaptation counters still advance. To hold QKV multipliers fixed, pass
`qkv_lr_autoadapt=False, qkv_spectral_adapt=False`. To isolate spectral feedback,
leave QKV adaptation enabled and set only `qkv_spectral_adapt=False`.

For experiments that reduce spectral feedback during training, keep the feature
enabled and set its strength between zero and one:

```python
optimizer.set_qkv_spectral_strength(0.5)
```

`qkv_spectral_strength=1.0` is the constructor default. Strength blends the
clipped frequency and phase corrections toward one; it does not directly scale
the learning rate. Zero skips FFT collection while RMS/trust adaptation keeps
running. Spectral EMA history is retained through a pause, and the current
strength is saved with optimizer state. Change strength eagerly before a step.
The recall probe's `tiger-spectral-fade` recipe keeps full strength for half the
budget, fades to zero over the next quarter, then finishes without spectral
feedback. Its timing is an experiment, not a recommended training preset.

The default cadence is 25 updates with gain 0.02. Increasing cadence and gain
did not consistently improve the [matched Mac probe](benchmarks/evidence/2026-09-30-qkv-learning/README.md).
The [learnable Mac retrieval control](benchmarks/evidence/2026-10-01-mac-qkv-ablation/README.md)
separates fixed scales, adaptive slice LR, spectral feedback and slice trust.
At its frozen LR, asymmetric scales rescue two failed uniform-scale runs;
spectral feedback worsens final test CE in all three confirmation seeds.
The [mean-scale follow-up](benchmarks/evidence/2026-10-01-qkv-scale-factorial/README.md)
shows that lowering the mean multiplier also rescues a uniform-scale run.
With [physical QKV RMS matched each step](benchmarks/evidence/2026-10-01-qkv-rms-matched/README.md),
asymmetric allocation improves final CE in three of five new seeds, with a
mean gain driven by one seed. These results keep allocation and magnitude
effects separately observable.

The [learned CUDA follow-up](benchmarks/evidence/2026-10-02-cuda-spectral/README.md)
uses a 608,256-parameter model with eight pairs and an eight-token gap. All
four controls pass binding in seven BF16 seeds and two FP32 supplements.
Full spectral lowers final CE in 4/7 BF16 seeds but worsens the average; fading
lowers CE versus full in 3/7 and worsens both FP32 supplements. The study
supports a controllable feedback mechanism, with its learning effect still
dependent on the recipe and seed.

### Schedule and resume

`TailCosineLR` holds the initial LR and then decays toward a fraction of it:

```python
from tiger_optim import TailCosineLR

scheduler = TailCosineLR(optimizer, total_steps=1000, decay_start=900,
                         min_lr_ratio=0.1)
# In each iteration, after backward():
optimizer.step()
scheduler.step()
```

`decay_start` counts completed updates. Here, updates 1–901 use the initial LR;
the floor is installed after update 1000. The scheduler preserves group LR
ratios. Disable `auto_lr` when using an external schedule.

For plateau control instead, enable `auto_lr` and call
`optimizer.report_metrics(loss=float(loss.detach()))` each iteration. Only
reported finite losses advance its plateau history; `auto_blend` also consumes
reported metrics. These controls run eagerly.

Save model, optimizer, and scheduler together:

```python
torch.save({"model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict()}, "checkpoint.pt")

# Recreate model, tagged groups, optimizer, and scheduler first,
# keeping parameter/group order and constructor options the same.
checkpoint = torch.load("checkpoint.pt", map_location=device)
model.load_state_dict(checkpoint["model"])
optimizer.load_state_dict(checkpoint["optimizer"])
scheduler.load_state_dict(checkpoint["scheduler"])
```

Old checkpoints without Tiger's adaptive state cannot resume exactly; the
loader warns and needs newly built QKV rules when loading old QKV groups.

## Numerical behavior and acceleration

- `skip_if_nonfinite=True` rejects unsafe parameter updates before committing
  candidate moments or decay for that parameter. It is not a whole-step
  rollback or a substitute for monitoring training loss.
- For FP16/BF16 parameters, `update_buffer_dtype="fp32"` keeps update buffers
  in FP32. Low-precision storage still has its own range limits.
- Median bucket standardization is rejected for low-precision parameters with
  nonfinite protection; use `global` or `mean` instead.
- Foreach, bucket standardization, Triton, and Julia are experimental execution
  choices. Measure their effect on the target model before claiming a speedup.

The core works without Julia or Triton. Runtime backend controls are available:

```python
from tiger_optim import configure_backends, backend_diagnostics
configure_backends(preferred=["torch"])
print(backend_diagnostics())
```

`configure_backends(disabled=["all"])` selects the eager reference path.
`reset_backend_configuration()` clears process overrides;
`refresh_backend_state(reload=True, reset_metrics=True)` refreshes availability
and timing history. Environment alternatives are `TIGER_ACCEL_DISABLE` and
`TIGER_ACCEL_PREFER`.

## Learning evidence

See the [benchmark guide](benchmarks/README.md) for runnable protocols and the
archive. Open questions include whether spectral QKV feedback improves held-out
learning and whether LoRA controls help actual adapter training.

- [Matched QKV magnitude](benchmarks/evidence/2026-10-01-qkv-rms-matched/README.md):
  a 2×2 [allocation/mean-scale factorial](benchmarks/evidence/2026-10-01-qkv-scale-factorial/README.md)
  followed by five additional seeds with equal physical QKV RMS every step.
  In the factorial, low-mean recipes pass binding in 4/5 seeds and unit-mean
  recipes in 3/5, for both allocations. Under physical RMS matching, asymmetric
  allocation lowers CE in 3/5 seeds; both arms pass binding in 5/5. The earlier
  fixed-versus-uniform result combines allocation with a lower mean LR scale.
- [Mac QKV component ablation](benchmarks/evidence/2026-10-01-mac-qkv-ablation/README.md):
  102,912-parameter short retrieval control, six configurations, three paired
  confirmation seeds, and an MPS replay. Full QKV averages 99.76% original and
  99.87% rebound test accuracy; AdamW reaches 100% on both. Fixed slice scales
  have the largest observed component effect at the shared LR. Spectral
  feedback improves sampled validation CE but worsens final test CE in all
  three seeds. This is a learnable synthetic control with binding checks.
- [Mac causal Transformer](benchmarks/evidence/2026-09-27-toy-transformer-learning/README.md):
  learning and schedule checks on a short periodic-copy task.
- [QKV control audit](benchmarks/evidence/2026-09-30-qkv-learning/README.md):
  33 runs, a verified group-scale correction, and a stronger-adaptation candidate
  that improved only one of three new confirmation seeds.
- [CUDA associative recall](benchmarks/evidence/2026-09-30-cuda-recall/README.md):
  a 1.83M-parameter Transformer and QKV ablations on RTX 5090. At 1000 updates,
  Tiger full averaged 7.59% test accuracy and AdamW 8.55%; both underperformed
  a query-independent context baseline (12.42%). Spectral feedback showed no
  consistent gain. A separate 5000-update AdamW diagnostic also failed to
  establish retrieval mastery. This harder stress task still needs a successful
  learning control before it can assess an optimizer's retrieval advantage.

Historical timing notes without raw receipts have been removed from this
README. Tracked experiments remain in the archive so source revisions,
failed hypotheses, and correctness comparisons remain reproducible. None of
these results establishes a general advantage over AdamW.

## Development

```sh
python -m pip install -e '.[dev]'
python -m pytest -q
python benchmarks/bench_associative_recall.py \
  --device cpu --precision fp32 --mode tiger-full --lr 0.003 --seed 11 \
  --steps 3 --width 24 --layers 2 --heads 3 --symbols 16 --pairs 4 \
  --gap 3 --queries 3 --batch-size 4 --eval-size 8 --test-size 8 \
  --output benchmarks/results/recall-smoke.json
```

Use a fresh output path: the learning benchmark refuses to overwrite results.
CI checks Python 3.9/3.12, tests, learning/RMS matching smoke, wheel/sdist builds, and the
installed wheel. Device-specific checks skip when unavailable.

## License

**GNU AGPL-3.0-only.** See [LICENSE.txt](LICENSE.txt).
