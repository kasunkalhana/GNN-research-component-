"""Lightweight timing utilities for profiling."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class TimerStats:
    """Aggregated timing statistics."""

    count: int = 0
    total: float = 0.0

    def update(self, duration: float) -> None:
        self.count += 1
        self.total += duration

    @property
    def avg(self) -> float:
        return self.total / self.count if self.count else 0.0


@dataclass
class TimerPool:
    """Manage a collection of timers identified by names."""

    stats: Dict[str, TimerStats] = field(default_factory=dict)

    @contextmanager
    def track(self, name: str):
        start = time.perf_counter()
        yield
        duration = time.perf_counter() - start
        self.stats.setdefault(name, TimerStats()).update(duration)


__all__ = ["TimerPool", "TimerStats"]
