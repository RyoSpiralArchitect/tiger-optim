# Benchmarks and evidence

## Current learning protocol: associative recall

[`bench_associative_recall.py`](bench_associative_recall.py) trains a causal
Transformer to answer four queries about 16 key/value pairs across a 64-token
gap. Training examples are generated independently for each update. Answers
never appear in the query suffix. This is synthetic retrieval, not language
modeling.

Defaults: four layers, width 192, six heads, batch 64, 1000 updates. CUDA uses
BF16 autocast with FP32 parameters and optimizer buffers. The four modes
`adamw`, `tiger-full`, `tiger-no-spectral`, and `tiger-fixed-qkv` share model/data
seeds, gradient clipping, and a warmup/cosine schedule. Tiger retains its tag
scales and trust recipe; the AdamW row is a configured reference. FFN asymmetry
and LoRA cross-adaptation are disabled to isolate the QKV comparison.

```sh
python benchmarks/bench_associative_recall.py \
  --device cuda --precision bf16 --mode tiger-full \
  --lr 0.003 --seed 211 --steps 1000 \
  --output benchmarks/results/recall-tiger-211.json
```

The LR above is an example. For comparison, use a separate development search,
lock its selection, then evaluate new confirmation seeds. `--development`
never creates or evaluates test data. Confirmation evaluates test data once
after training. A [frozen plan and results](evidence/2026-09-30-cuda-recall/README.md)
record the procedure. Runs store input/source hashes, complete update and
validation traces, final test scores, and an explicit completion status.
Timing includes synchronization and safety checks; it is diagnostic.

The archived recipes did not beat a query-independent context baseline, even
with a separate extended AdamW diagnostic. Treat this as a stress probe; a
successful learning control is still needed before using it to assess an
optimizer's retrieval advantage.

## Other tools

| Script | Purpose |
| --- | --- |
| `bench_toy_transformer_learning.py` | Periodic-copy learning and scheduler regression on CPU/MPS |
| `bench_quality_smoke.py` | Earlier held-out CPU quality smoke |
| `bench_compare_optim.py` | Timing/execution smoke; historical mode names remain for compatibility |
| `bench_profile_v21.py` | Optimizer profiling, including MPS scalar-read diagnostics |
| `summarize_results.py`, `plot_bench.py` | Reports for the timing schema, not the learning schemas |

Timing tools remain because CI and earlier evidence use them. Use a fresh
result directory for every comparison: the timing summarizer selects the latest
file per mode/device and does not estimate variation across runs.

## Evidence archive

| Record | Scope |
| --- | --- |
| [CUDA recall, Sep 30](evidence/2026-09-30-cuda-recall/README.md) | Larger retrieval task and QKV ablations |
| [QKV learning, Sep 30](evidence/2026-09-30-qkv-learning/README.md) | Group-scale correctness and rejected stronger adaptation |
| [Toy Transformer, Sep 27](evidence/2026-09-27-toy-transformer-learning/README.md) | Periodic-copy learning and tail schedule |
| [FP32 bounds, Sep 27](evidence/2026-09-27-fp32-bound-batch/README.md) | Overflow-guard batching |
| [Trust guard, Sep 27](evidence/2026-09-27-trust-guard/README.md) | Trust and finite-state protection |
| [QKV spectral views, Sep 27](evidence/2026-09-27-qkv-spectral-view/README.md) | Batched spectral processing |
| [Step-25 follow-up, Sep 26](evidence/2026-09-26-step25-followup/README.md) | Adaptation-boundary diagnostics |
| [Scalar sync, Sep 26](evidence/2026-09-26-scalar-sync/README.md) | Host/device synchronization |
| [QKV batching, Sep 26](evidence/2026-09-26-qkv-batch/README.md) | FFT batching correctness |
| [Mac step 25, Sep 26](evidence/2026-09-26-mac-step25/README.md) | Spectral adaptation timing spikes |
| [Mac diagnostics, Sep 26](evidence/2026-09-26-mac-perf/README.md) | CPU/MPS profiling |
| [CPU quality, Sep 25](evidence/2026-09-25-cpu-quality/README.md) | Early held-out learning probe |
| [CPU smoke, Sep 25](evidence/2026-09-25-cpu-smoke/README.md) | Historical paired timing smoke |

Records apply to their own source revision/environment. New speed claims need
controlled host load and repeated measurements; new quality claims need
matched budgets, held-out data, and multiple seeds. Preserve raw results,
including negative outcomes, when comparing an implementation change.
