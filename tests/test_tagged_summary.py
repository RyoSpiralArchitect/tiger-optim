import pytest
import torch
import torch.nn as nn

from tiger_optim import (
    ParamGroupSummary,
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
