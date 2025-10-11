
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
