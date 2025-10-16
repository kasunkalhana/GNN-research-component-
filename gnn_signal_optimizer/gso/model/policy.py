"""Action selection helpers."""

from __future__ import annotations

from typing import Dict, Optional

import torch


def mask_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = logits.clone()
    masked[mask <= 0] = -1e9
    return masked


def select_actions(
    logits: torch.Tensor,
    legal_masks: Dict[str, torch.Tensor],
    node_order: Dict[str, list[str]],
    epsilon: float = 0.0,
) -> Dict[str, int]:
    actions: Dict[str, int] = {}
    for idx, jid in enumerate(node_order["junction"]):
        mask = legal_masks.get(jid, torch.ones_like(logits[idx]))
        masked = mask_logits(logits[idx], mask)
        if epsilon > 0 and torch.rand(1).item() < epsilon:
            valid = mask.nonzero(as_tuple=False).flatten()
            if valid.numel() == 0:
                actions[jid] = int(torch.argmax(masked).item())
            else:
                actions[jid] = int(valid[torch.randint(len(valid), (1,))].item())
        else:
            actions[jid] = int(torch.argmax(masked).item())
    return actions


__all__ = ["mask_logits", "select_actions"]
