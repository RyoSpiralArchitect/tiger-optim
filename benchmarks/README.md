# Benchmarks and evidence

## Current learning protocol: associative recall

[`bench_associative_recall.py`](bench_associative_recall.py) trains a causal
Transformer to answer queries about key/value pairs. Training examples are
generated independently for each update. Answers
never appear in the query suffix. This is synthetic retrieval, not language
modeling.

Defaults: 16 pairs, 64 symbols, a 64-token gap, four queries, four layers,
width 192, six heads, batch 64, 1000 updates. CUDA uses BF16 autocast with FP32
parameters and optimizer buffers; CPU and MPS use FP32. The eight modes
`adamw`, `tiger-full`, `tiger-no-spectral`, `tiger-fixed-qkv`,
`tiger-uniform-qkv`, `tiger-global-trust`, `tiger-normalized-qkv`, and
`tiger-uniform-low-qkv` share model/data
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
Schema 2 additionally records actual QKV controls/feedback and a final binding
check: rotate context values while preserving the value bag, keys and queries,
then evaluate the new correct answers and obsolete original answers. Development
uses validation for this check; confirmation uses test. Timing includes device
synchronization and safety checks; it is diagnostic.

### Learnable Mac control

The [Mac QKV ablation](evidence/2026-10-01-mac-qkv-ablation/README.md) uses four
pairs, 16 symbols, no gap and a 102,912-parameter Transformer. A separate LR
search qualifies learning with original/rebound accuracy and low obsolete-answer
accuracy, then locks rates for three new paired seeds and six configurations.
The component chain separates spectral feedback, adaptive slice LR, fixed
slice scales and separate versus shared trust within a fused QKV tensor.

```sh
python benchmarks/evidence/2026-10-01-mac-qkv-ablation/run_suite.py \
  --phase development --output-dir benchmarks/results/mac-qkv-fresh
python benchmarks/evidence/2026-10-01-mac-qkv-ablation/run_suite.py \
  --phase confirmation --output-dir benchmarks/results/mac-qkv-fresh
```

Full QKV averages 99.76% original and 99.87% rebound test accuracy. AdamW
reaches 100% on both. Asymmetric fixed scales rescue two uniform-scale failures
at the shared LR; spectral feedback worsens final test CE in all three seeds.
This is a conditional component comparison; ablations are not separately tuned.
A supplementary full-QKV replay also learns the binding task on MPS.

### Separate allocation from update magnitude

The [mean-scale factorial](evidence/2026-10-01-qkv-scale-factorial/README.md)
compares asymmetric/uniform allocation at arithmetic mean multipliers 14/15
and 1, using five new seeds and the inherited LR. Both low-mean cells pass
binding in 4/5 seeds; both unit-mean cells pass in 3/5. The opt-in
`--measure-qkv-updates` records actual per-slice and combined parameter-delta
RMS after every optimizer step. At equal coefficient means, asymmetric runs
still have larger mean applied RMS along their trajectories.

The [physical RMS study](evidence/2026-10-01-qkv-rms-matched/README.md) then uses
five more seeds with a reference-derived QKV RMS budget at every step. Both
comparison arms use `--qkv-rms-reference`, which rescales actual QKV deltas
after the ordinary Tiger step. It is an experimental benchmark intervention;
optimizer moments and non-QKV parameters are not rewritten. A completed
uniform reference with matching inputs/source is required. Both matched arms
pass binding in all five seeds; asymmetric allocation lowers final CE in 3/5,
with the favorable mean largely driven by one seed. Ordinary and matched
uniform trajectories differ numerically, so the primary control also uses
the wrapper.

```sh
python benchmarks/evidence/2026-10-01-qkv-rms-matched/run_suite.py \
  --output-dir benchmarks/results/qkv-rms-fresh
```

Schema 3 receipts add measured applied deltas; schema 4 adds reference
provenance and per-step candidate/target/applied RMS and rescale factors.
The source commits and protocols are frozen separately for the two studies.

### Harder CUDA stress task

The [Sep 30 CUDA recipes](evidence/2026-09-30-cuda-recall/README.md) did not beat
a query-independent context baseline, even with a separate extended AdamW
diagnostic. The shorter Mac control does not establish mastery of that harder
task, which still needs a successful learning control before it can assess an
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
| [Matched QKV RMS, Oct 1](evidence/2026-10-01-qkv-rms-matched/README.md) | Five new seeds with equal applied QKV magnitude and a common rescaling control |
| [QKV mean-scale factorial, Oct 1](evidence/2026-10-01-qkv-scale-factorial/README.md) | Allocation versus coefficient mean, actual update traces and negative MPS replay |
| [Mac QKV ablation, Oct 1](evidence/2026-10-01-mac-qkv-ablation/README.md) | Learnable retrieval, binding checks, six component controls and MPS replay |
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
