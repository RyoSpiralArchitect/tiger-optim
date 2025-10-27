# ====================================================================
# Copyright (C) 2025  Ryo ∴ SpiralArchitect and SpiralReality
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# ====================================================================

from .tiger import Tiger
from .tagged import (
    ParamGroupSummary,
    ParamTagAggregate,
    aggregate_param_group_stats,
    build_tagged_param_groups,
    collect_param_group_stats,
    summarize_param_groups,
)
from .schedulers import TagWarmupDecay
from .accel import (
    available_backends,
    backend_diagnostics,
    configure_backends,
    current_backend_priority,
    refresh_backend_state,
    reset_backend_configuration,
)

__all__ = [
    "Tiger",
    "ParamGroupSummary",
    "ParamTagAggregate",
    "aggregate_param_group_stats",
    "build_tagged_param_groups",
    "collect_param_group_stats",
    "summarize_param_groups",
    "TagWarmupDecay",
    "available_backends",
    "configure_backends",
    "reset_backend_configuration",
    "refresh_backend_state",
    "backend_diagnostics",
    "current_backend_priority",
]
__version__ = "2.4.0"
