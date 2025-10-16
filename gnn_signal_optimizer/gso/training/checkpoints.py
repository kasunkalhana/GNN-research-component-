"""Checkpointing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch


def save_checkpoint(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: Path) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu")


__all__ = ["save_checkpoint", "load_checkpoint"]
