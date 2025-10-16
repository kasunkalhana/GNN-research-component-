"""Simple experience replay buffer."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple

import torch


@dataclass
class Transition:
    state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_state: torch.Tensor
    done: torch.Tensor


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.storage: Deque[Transition] = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self.storage.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        idx = torch.randperm(len(self.storage))[:batch_size]
        return [list(self.storage)[i] for i in idx]

    def __len__(self) -> int:
        return len(self.storage)


__all__ = ["Transition", "ReplayBuffer"]
