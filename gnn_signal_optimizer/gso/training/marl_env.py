"""Multi-agent environment wrapper for MARL fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch


@dataclass
class RewardWeights:
    wait: float = 1.0
    queue: float = 0.5
    stops: float = 0.1


class JunctionMARLEnv:
    """Lightweight environment used for algorithm prototyping."""

    def __init__(self, num_junctions: int, action_dim: int, reward_weights: RewardWeights | None = None):
        self.num_junctions = num_junctions
        self.action_dim = action_dim
        self.reward_weights = reward_weights or RewardWeights()
        self.state = torch.zeros(num_junctions, 6)
        self.step_count = 0

    def reset(self) -> torch.Tensor:
        self.step_count = 0
        self.state = torch.zeros_like(self.state)
        return self.state

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, bool, Dict[str, float]]:
        self.step_count += 1
        noise = torch.randn_like(self.state) * 0.1
        self.state = self.state + noise + actions.unsqueeze(-1) * 0.01
        reward = -self.state.mean(dim=1)
        done = self.step_count >= 100
        info = {"step": self.step_count}
        return self.state, reward, done, info


__all__ = ["JunctionMARLEnv", "RewardWeights"]
