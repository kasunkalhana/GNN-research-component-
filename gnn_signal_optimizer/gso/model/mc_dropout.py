"""Utilities for Monte Carlo Dropout inference."""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import torch


@torch.no_grad()
def mc_dropout_inference(
    model: torch.nn.Module,
    forward_fn: Callable[[torch.nn.Module], torch.Tensor],
    passes: int = 30,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run multiple stochastic forward passes with dropout enabled."""

    model.train()
    logits: list[torch.Tensor] = []
    for _ in range(passes):
        logits.append(forward_fn(model))
    stacked = torch.stack(logits, dim=0)
    mean = stacked.mean(dim=0)
    var = stacked.var(dim=0, unbiased=False)
    return mean, var


def predictive_entropy(mean_logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(mean_logits, dim=-1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)
    return entropy


__all__ = ["mc_dropout_inference", "predictive_entropy"]
