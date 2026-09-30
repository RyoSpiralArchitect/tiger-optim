"""Check retrieval labels, causality, and the actual QKV ablation boundaries."""
import importlib.util
from pathlib import Path

import torch

spec = importlib.util.spec_from_file_location("recall", Path(__file__).parents[1] / "benchmarks/bench_associative_recall.py")
recall = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recall)


def test_queries_recover_context_values_without_answer_tokens():
    tokens, targets = recall.corpus(13, 20, symbols=16, pairs=4, gap=3, queries=3)
    again = recall.corpus(13, 20, symbols=16, pairs=4, gap=3, queries=3)
    assert torch.equal(tokens, again[0]) and torch.equal(targets, again[1])
    for row, expected in zip(tokens.tolist(), targets.tolist()):
        mapping = dict(zip(row[:8:2], [value - 16 for value in row[1:8:2]]))
        assert len(mapping) == 4
        assert row[8:11] == [32] * 3
        assert row[11::2] == [33] * 3
        assert [mapping[key] for key in row[12::2]] == expected
        assert all(key < 16 for key in row[12::2])


def test_future_queries_cannot_change_earlier_predictions():
    torch.manual_seed(4)
    model = recall.RecallTransformer(symbols=16, pairs=4, gap=3, queries=3, width=24, layers=2, heads=3).cpu().eval()
    tokens, _ = recall.corpus(13, 2, symbols=16, pairs=4, gap=3, queries=3)
    altered = tokens.clone()
    altered[:, 14] = (altered[:, 14] + 1) % 16
    with torch.no_grad():
        torch.testing.assert_close(model(tokens)[:, 0], model(altered)[:, 0], rtol=0, atol=0)


def test_fixed_qkv_control_and_spectral_ablation_are_distinct():
    model = recall.RecallTransformer(symbols=16, pairs=4, gap=3, queries=3, width=24, layers=2, heads=3).cpu()
    full = recall.optimizer_for(model, "tiger-full", 0.003)
    no_spectral = recall.optimizer_for(model, "tiger-no-spectral", 0.003)
    fixed = recall.optimizer_for(model, "tiger-fixed-qkv", 0.003)
    assert full.defaults["qkv_lr_autoadapt"] and full.defaults["qkv_spectral_adapt"]
    assert no_spectral.defaults["qkv_lr_autoadapt"] and not no_spectral.defaults["qkv_spectral_adapt"]
    assert not fixed.defaults["qkv_lr_autoadapt"] and not fixed.defaults["qkv_spectral_adapt"]
    for optimizer in (full, no_spectral, fixed):
        group = next(g for g in optimizer.param_groups if g["block_tag"] == "attn_qkv")
        assert len(group["qkv_rules"]) == 2
        assert group["qkv_lr_scales"] == {"q": 0.9, "k": 0.8, "v": 1.1}
