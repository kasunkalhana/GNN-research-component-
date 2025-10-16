"""Random seed helpers."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def seed_everything(seed: Optional[int] = None) -> int:
    """Seed Python, NumPy, and torch RNGs.

    Args:
        seed: Optional manual seed. If ``None``, one will be sampled from ``os.urandom``.
    Returns:
        The seed used.
    """

    if seed is None:
        seed = int.from_bytes(os.urandom(4), "little")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - depends on GPU
        torch.cuda.manual_seed_all(seed)
    return seed


__all__ = ["seed_everything"]
