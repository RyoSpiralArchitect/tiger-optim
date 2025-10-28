import math

import pytest
import torch
import torch.nn as nn

from tiger_optim import (
    ParamGroupSummary,
    ParamTagAggregate,
    aggregate_param_group_stats,
    build_tagged_param_groups,
    collect_param_group_stats,
    summarize_param_groups,
)


class _TinyDemo(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 3, bias=False)
        self.norm = nn.LayerNorm(3)


@pytest.fixture()
def demo_param_groups():
    model = _TinyDemo()
    return build_tagged_param_groups(model)


def test_collect_param_group_stats_shapes(demo_param_groups):
    stats = collect_param_group_stats(demo_param_groups)
    assert all(isinstance(item, ParamGroupSummary) for item in stats)

    total_params = sum(item.n_params for item in stats)
    expected_total = sum(
        int(p.numel()) for group in demo_param_groups for p in group["params"] if isinstance(p, torch.Tensor)
    )
    assert total_params == expected_total
    assert sum(item.param_ratio for item in stats) == pytest.approx(1.0)

    tags = {item.tag: item for item in stats}
    assert "mlp" in tags
    assert tags["mlp"].n_params == 12  # 4 * 3 matrix without bias


def test_summarize_param_groups_includes_share_column(demo_param_groups):
    summary = summarize_param_groups(demo_param_groups, include_share=True, sort_by="index", precision=2)
    header = summary.splitlines()[0].lower()
    assert "share" in header

    total_params = sum(
        int(p.numel()) for group in demo_param_groups for p in group["params"] if isinstance(p, torch.Tensor)
    )
    assert f"Total params: {total_params:,}" in summary


def test_collect_param_group_stats_empty_handles_gracefully():
    assert collect_param_group_stats([]) == []
    assert summarize_param_groups([], include_share=True) == "(no parameter groups)"


def test_aggregate_param_group_stats_basic(demo_param_groups):
    aggregates = aggregate_param_group_stats(demo_param_groups)
    assert all(isinstance(item, ParamTagAggregate) for item in aggregates)

    mapping = {item.tag: item for item in aggregates}
    assert set(mapping) == {group["block_tag"] for group in demo_param_groups}

    total_params = sum(item.total_params for item in aggregates)
    expected = sum(
        int(p.numel()) for group in demo_param_groups for p in group["params"] if isinstance(p, torch.Tensor)
    )
    assert total_params == expected
    assert sum(item.param_ratio for item in aggregates) == pytest.approx(1.0)

    mlp_group = mapping["mlp"]
    assert mlp_group.groups == 1
    assert mlp_group.total_tensors > 0
    assert mlp_group.avg_lr > 0
    assert mlp_group.avg_weight_decay >= 0
    assert mlp_group.avg_lr_scale >= 0
    assert mlp_group.avg_effective_lr == pytest.approx(mlp_group.avg_lr * mlp_group.avg_lr_scale)
    assert mlp_group.lr_std == pytest.approx(0.0)
    assert mlp_group.weight_decay_std == pytest.approx(0.0)
    assert mlp_group.lr_scale_std == pytest.approx(0.0)
    assert mlp_group.effective_lr_std == pytest.approx(0.0)
    assert mlp_group.min_lr == mlp_group.max_lr == mlp_group.avg_lr
    assert mlp_group.min_weight_decay == mlp_group.max_weight_decay == mlp_group.avg_weight_decay
    assert mlp_group.min_lr_scale == mlp_group.max_lr_scale == mlp_group.avg_lr_scale
    assert mlp_group.min_effective_lr == mlp_group.max_effective_lr == mlp_group.avg_effective_lr


def test_aggregate_param_group_stats_accepts_precomputed_summaries(demo_param_groups):
    summaries = collect_param_group_stats(demo_param_groups)
    direct = aggregate_param_group_stats(demo_param_groups)
    from_summaries = aggregate_param_group_stats(summaries)

    assert direct == from_summaries


def test_aggregate_param_group_stats_rejects_mixed_input(demo_param_groups):
    summaries = collect_param_group_stats(demo_param_groups)

    with pytest.raises(TypeError) as exc:
        aggregate_param_group_stats([demo_param_groups[0], summaries[0]])

    assert "mixed input" in str(exc.value)


def test_aggregate_param_group_stats_rejects_unknown_items():
    with pytest.raises(TypeError) as exc:
        aggregate_param_group_stats([object()])

    message = str(exc.value)
    assert "mapping-like" in message
    assert "object" in message


def test_aggregate_param_group_stats_empty_returns_empty():
    assert aggregate_param_group_stats([]) == []


def test_aggregate_param_group_stats_tracks_extrema():
    base = dict(params=[torch.zeros(1)], block_tag="shared", lr_scale=1.0)
    groups = [
        {**base, "lr": 0.1, "weight_decay": 0.0},
        {**base, "lr": 0.2, "weight_decay": 0.01},
        {**base, "lr": 0.05, "weight_decay": 0.001},
    ]

    aggregates = aggregate_param_group_stats(groups)
    assert len(aggregates) == 1

    agg = aggregates[0]
    assert agg.min_lr == pytest.approx(0.05)
    assert agg.max_lr == pytest.approx(0.2)
    assert agg.min_weight_decay == pytest.approx(0.0)
    assert agg.max_weight_decay == pytest.approx(0.01)
    assert agg.min_lr_scale == agg.max_lr_scale == pytest.approx(1.0)
    assert agg.min_effective_lr == pytest.approx(0.05)
    assert agg.max_effective_lr == pytest.approx(0.2)
    assert agg.lr_std == pytest.approx(0.06236095644623236)
    assert agg.weight_decay_std == pytest.approx(0.004496912521077348)
    assert agg.lr_scale_std == pytest.approx(0.0)
    assert agg.effective_lr_std == pytest.approx(0.06236095644623236)


def test_aggregate_param_group_stats_zero_param_groups_fallback_to_simple_average():
    groups = [
        {"params": [], "block_tag": "empty", "lr": 0.1, "weight_decay": 0.02, "lr_scale": 0.5},
        {"params": [], "block_tag": "empty", "lr": 0.2, "weight_decay": 0.04, "lr_scale": 1.5},
    ]

    aggregates = aggregate_param_group_stats(groups)
    assert len(aggregates) == 1

    agg = aggregates[0]
    assert agg.avg_lr == pytest.approx(0.15)
    assert agg.avg_weight_decay == pytest.approx(0.03)
    assert agg.avg_lr_scale == pytest.approx(1.0)
    assert agg.avg_effective_lr == pytest.approx(0.175)
    assert agg.total_params == 0
    assert agg.param_ratio == pytest.approx(0.0)
    assert agg.lr_std == pytest.approx(0.05)
    assert agg.weight_decay_std == pytest.approx(0.01)
    assert agg.lr_scale_std == pytest.approx(0.5)
    assert agg.effective_lr_std == pytest.approx(0.125)


def test_aggregate_param_group_stats_weighted_dispersion_respects_param_counts():
    heavy = torch.zeros(10)
    light = torch.zeros(1)
    groups = [
        {"params": [heavy], "block_tag": "weighted", "lr": 0.1, "weight_decay": 0.02, "lr_scale": 0.5},
        {"params": [light], "block_tag": "weighted", "lr": 0.4, "weight_decay": 0.05, "lr_scale": 0.5},
    ]

    aggregates = aggregate_param_group_stats(groups)
    assert len(aggregates) == 1

    agg = aggregates[0]
    total = heavy.numel() + light.numel()
    expected_lr = ((0.1 * heavy.numel()) + (0.4 * light.numel())) / total
    expected_lr_sq = ((0.1 ** 2) * heavy.numel() + (0.4 ** 2) * light.numel()) / total
    expected_lr_std = math.sqrt(max(0.0, expected_lr_sq - expected_lr ** 2))
    expected_wd = ((0.02 * heavy.numel()) + (0.05 * light.numel())) / total
    expected_wd_sq = ((0.02 ** 2) * heavy.numel() + (0.05 ** 2) * light.numel()) / total
    expected_wd_std = math.sqrt(max(0.0, expected_wd_sq - expected_wd ** 2))
    expected_effective_lr = ((0.1 * 0.5 * heavy.numel()) + (0.4 * 0.5 * light.numel())) / total
    expected_effective_lr_sq = (((0.1 * 0.5) ** 2) * heavy.numel() + ((0.4 * 0.5) ** 2) * light.numel()) / total
    expected_effective_lr_std = math.sqrt(max(0.0, expected_effective_lr_sq - expected_effective_lr ** 2))

    assert agg.avg_lr == pytest.approx(expected_lr)
    assert agg.lr_std == pytest.approx(expected_lr_std)
    assert agg.avg_weight_decay == pytest.approx(expected_wd)
    assert agg.weight_decay_std == pytest.approx(expected_wd_std)
    assert agg.avg_effective_lr == pytest.approx(expected_effective_lr)
    assert agg.effective_lr_std == pytest.approx(expected_effective_lr_std)


def _extract_summary_tags(summary: str) -> list[str]:
    tags: list[str] = []
    for line in summary.splitlines()[2:]:
        if line.startswith("Total params"):
            break
        columns = line.split()
        if columns:
            tags.append(columns[1])
    return tags


def _extract_summary_column(summary: str, header: str) -> list[str]:
    lines = summary.splitlines()
    headers = lines[0].split()
    try:
        index = headers.index(header)
    except ValueError as exc:  # pragma: no cover - defensive guard for debugging
        raise AssertionError(f"Column {header!r} not found in summary header: {headers}") from exc

    values: list[str] = []
    for line in lines[2:]:
        if line.startswith("Total params"):
            break
        columns = line.split()
        if columns:
            values.append(columns[index])
    return values


def test_summarize_param_groups_sort_by_lr_scale_descending(demo_param_groups):
    summary = summarize_param_groups(demo_param_groups, sort_by="lr_scale")
    tags = _extract_summary_tags(summary)
    assert tags[0] == "mlp"


def test_summarize_param_groups_sort_by_lr_scale_ascending(demo_param_groups):
    summary = summarize_param_groups(demo_param_groups, sort_by="lr_scale", descending=False)
    tags = _extract_summary_tags(summary)
    assert tags[-1] == "mlp"


def test_summarize_param_groups_unknown_sort_falls_back_to_index(demo_param_groups):
    summary = summarize_param_groups(demo_param_groups, sort_by="unknown")
    tags = _extract_summary_tags(summary)
    assert tags == ["mlp", "norm", "bias"]


def test_summarize_param_groups_tag_descending_order(demo_param_groups):
    summary = summarize_param_groups(demo_param_groups, sort_by="tag", descending=True)
    tags = _extract_summary_tags(summary)
    assert tags == ["norm", "mlp", "bias"]


def test_summarize_param_groups_params_alias_descends(demo_param_groups):
    summary = summarize_param_groups(demo_param_groups, sort_by="params")
    tags = _extract_summary_tags(summary)
    assert tags[0] == "mlp"


def test_summarize_param_groups_weight_decay_alias(demo_param_groups):
    summary = summarize_param_groups(demo_param_groups, sort_by="wd", descending=False)
    tags = _extract_summary_tags(summary)
    assert tags[-1] == "mlp"


def test_summarize_param_groups_share_defaults_descending(demo_param_groups):
    summary = summarize_param_groups(demo_param_groups, include_share=True, sort_by="share")
    shares = [float(value.rstrip("%")) for value in _extract_summary_column(summary, "share")]
    assert shares == sorted(shares, reverse=True)
