"""Check retrieval labels, causality, and the actual QKV ablation boundaries."""
import importlib.util
from pathlib import Path
import math

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


def test_spectral_fade_keeps_initial_full_recipe_and_completed_step_boundaries():
    model = recall.RecallTransformer(symbols=16, pairs=4, gap=0, queries=2, width=24, layers=2, heads=3).cpu()
    full = recall.optimizer_for(model, "tiger-full", 0.003)
    fade = recall.optimizer_for(model, "tiger-spectral-fade", 0.003)
    assert full.defaults == fade.defaults
    assert recall.spectral_strength_factor(0, 100) == 1.0
    assert recall.spectral_strength_factor(50, 100) == 1.0
    assert math.isclose(recall.spectral_strength_factor(62.5, 100), 0.5)
    assert recall.spectral_strength_factor(75, 100) == 0.0
    assert recall.spectral_strength_factor(100, 100) == 0.0


def test_rebinding_preserves_value_bag_and_queries_but_changes_associations():
    tokens, targets = recall.corpus(13, 20, symbols=16, pairs=4, gap=3, queries=3)
    rebound, rebound_targets = recall.rebind_values((tokens, targets), symbols=16, pairs=4, gap=3, queries=3)
    assert torch.equal(tokens[:, :8:2], rebound[:, :8:2])
    assert torch.equal(tokens[:, 8:], rebound[:, 8:])
    assert torch.equal(tokens[:, 1:8:2].sort(dim=1).values, rebound[:, 1:8:2].sort(dim=1).values)
    assert (targets != rebound_targets).any()
    for row, expected in zip(rebound.tolist(), rebound_targets.tolist()):
        mapping = dict(zip(row[:8:2], [value - 16 for value in row[1:8:2]]))
        assert [mapping[key] for key in row[12::2]] == expected


def test_uniform_scale_and_shared_trust_change_one_control_at_a_time():
    model = recall.RecallTransformer(symbols=16, pairs=4, gap=3, queries=3, width=24, layers=2, heads=3).cpu()
    optimizers = [recall.optimizer_for(model, mode, 0.003)
                  for mode in ("tiger-fixed-qkv", "tiger-uniform-qkv", "tiger-global-trust")]
    groups = [next(g for g in opt.param_groups if g["block_tag"] == "attn_qkv") for opt in optimizers]
    assert groups[0]["qkv_lr_scales"] == {"q": 0.9, "k": 0.8, "v": 1.1}
    assert groups[1]["qkv_lr_scales"] == groups[2]["qkv_lr_scales"] == {"q": 1.0, "k": 1.0, "v": 1.0}
    assert groups[0]["qkv_trust_split"] and groups[1]["qkv_trust_split"]
    assert not groups[2]["qkv_trust_split"]
    assert all(not opt.defaults["qkv_lr_autoadapt"] and not opt.defaults["qkv_spectral_adapt"] for opt in optimizers)


def test_qkv_update_metrics_pool_applied_deltas_across_layers():
    model = recall.RecallTransformer(symbols=16, pairs=4, gap=0, queries=2, width=6, layers=2, heads=2).cpu()
    before = [torch.zeros_like(block.qkv.weight) for block in model.blocks]
    with torch.no_grad():
        for layer, block in enumerate(model.blocks):
            for index, part in enumerate(block.qkv.weight.chunk(3, dim=0)):
                part.fill_(3 * layer + index + 1)
    after = [block.qkv.weight.detach().clone() for block in model.blocks]
    metrics = recall.qkv_update_metrics(model, before)
    for index, key in enumerate(("q", "k", "v")):
        expected = math.sqrt(((index + 1) ** 2 + (index + 4) ** 2) / 2)
        assert math.isclose(metrics["rms_by_slice"][key], expected, rel_tol=1e-6)
    assert math.isclose(metrics["combined_rms"], math.sqrt(91 / 6), rel_tol=1e-6)
    assert all(torch.equal(block.qkv.weight, saved) for block, saved in zip(model.blocks, after))


def test_qkv_rms_matching_preserves_delta_direction_and_other_weights():
    model = recall.RecallTransformer(symbols=16, pairs=4, gap=0, queries=2, width=6, layers=2, heads=2).cpu()
    before = [block.qkv.weight.detach().clone() for block in model.blocks]
    other = model.output.weight.detach().clone()
    with torch.no_grad():
        for layer, block in enumerate(model.blocks):
            for index, part in enumerate(block.qkv.weight.chunk(3, dim=0)):
                part.add_(3 * layer + index + 1)
    candidate = [block.qkv.weight.detach().clone() for block in model.blocks]
    target = recall.qkv_update_metrics(model, before)["combined_rms"] * 0.5
    match = recall.match_qkv_update_rms(model, before, target)
    assert math.isclose(match["applied_rms"], target, rel_tol=1e-6)
    assert math.isclose(match["rescale_factor"], 0.5, rel_tol=1e-6)
    for block, old, update in zip(model.blocks, before, candidate):
        torch.testing.assert_close(block.qkv.weight - old, (update - old) * 0.5)
    assert torch.equal(model.output.weight, other)
