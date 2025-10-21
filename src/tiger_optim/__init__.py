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
from .tagged import build_tagged_param_groups, summarize_param_groups
from .schedulers import TagWarmupDecay

__all__ = ["Tiger","build_tagged_param_groups","summarize_param_groups","TagWarmupDecay"]
__version__ = "2.1.0"
