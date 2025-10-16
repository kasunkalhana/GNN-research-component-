"""Feature normalization utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import torch


@dataclass
class RunningNorm:
    """Track running mean and std for online normalization."""

    count: int = 0
    mean: torch.Tensor | None = None
    m2: torch.Tensor | None = None

    def update(self, batch: torch.Tensor) -> None:
        if batch.numel() == 0:
            return
        if self.mean is None:
            self.mean = batch.mean(dim=0)
            self.m2 = torch.zeros_like(self.mean)
            self.count = batch.shape[0]
            return
        self.count += batch.shape[0]
        delta = batch.mean(dim=0) - self.mean
        self.mean += delta * batch.shape[0] / self.count
        self.m2 += batch.var(dim=0, unbiased=False)

    def normalize(self, batch: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.m2 is None:
            return batch
        std = torch.sqrt(self.m2 / max(self.count, 1)) + 1e-6
        return (batch - self.mean) / std


@dataclass
class NormalizerRegistry:
    """Keep normalizers for each node type."""

    norms: Dict[str, RunningNorm] = field(default_factory=dict)

    def update(self, node_type: str, batch: torch.Tensor) -> None:
        self.norms.setdefault(node_type, RunningNorm()).update(batch)

    def normalize(self, node_type: str, batch: torch.Tensor) -> torch.Tensor:
        return self.norms.setdefault(node_type, RunningNorm()).normalize(batch)


__all__ = ["RunningNorm", "NormalizerRegistry"]
