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
from __future__ import annotations
import math
from torch.optim.lr_scheduler import _LRScheduler

class TagWarmupDecay(_LRScheduler):
    def __init__(self, optimizer, *, default_warmup: int = 1000, default_total: int = 100_000,
                 default_schedule: str = "cosine", default_min_ratio: float = 0.1, last_epoch: int=-1):
        self.w = default_warmup; self.T = default_total; self.sched = default_schedule; self.m = float(default_min_ratio)
        super().__init__(optimizer, last_epoch)
    def get_lr(self):
        s = self.last_epoch + 1
        if s < self.w: k = s / max(1, self.w)
        else:
            t = min(1.0, (s - self.w) / max(1, self.T - self.w))
            k = self.m + (1.0 - self.m) * (0.5 * (1.0 + math.cos(math.pi * t)) if self.sched=="cosine" else (1.0 - t))
        return [base * k for base in self.base_lrs]
