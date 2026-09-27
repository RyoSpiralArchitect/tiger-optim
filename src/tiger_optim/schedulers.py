# ====================================================================
# Copyright (C) 2025  Ryo ∴ SpiralArchitect and SpiralReality
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, version 3 of the License.
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
from numbers import Integral
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


class TailCosineLR(_LRScheduler):
    """Hold each group LR, then decay it during the final updates.

    ``decay_start`` counts completed optimizer updates. Call ``step()`` after
    ``optimizer.step()``: with ``total_steps=100`` and ``decay_start=90``,
    updates 1 through 91 use the initial LR and update 100 uses the factor
    computed at completed update 99. The floor is set after update 100.

    The same factor applies to every group's initial LR, preserving their
    ratios. Save this scheduler's state alongside the optimizer to resume.
    """

    def __init__(
        self,
        optimizer,
        *,
        total_steps: int,
        decay_start: int,
        min_lr_ratio: float = 0.1,
        last_epoch: int = -1,
    ):
        if (
            isinstance(total_steps, bool)
            or not isinstance(total_steps, Integral)
            or total_steps < 1
        ):
            raise ValueError("total_steps must be a positive integer")
        if (
            isinstance(decay_start, bool)
            or not isinstance(decay_start, Integral)
            or not 0 <= decay_start < total_steps
        ):
            raise ValueError("decay_start must be an integer in [0, total_steps)")
        try:
            if isinstance(min_lr_ratio, bool):
                raise TypeError("boolean ratio")
            min_lr_ratio = float(min_lr_ratio)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("min_lr_ratio must be finite and in [0, 1]") from exc
        if not math.isfinite(min_lr_ratio) or not 0.0 <= min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be finite and in [0, 1]")
        self.total_steps = int(total_steps)
        self.decay_start = int(decay_start)
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        progress = min(
            1.0,
            max(
                0.0,
                (self.last_epoch - self.decay_start) / (self.total_steps - self.decay_start),
            ),
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        factor = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine
        return [base_lr * factor for base_lr in self.base_lrs]
