"""Export the HGAT policy to ONNX."""

from __future__ import annotations

from pathlib import Path

import torch

from gso.model.hgat import RecurrentHGAT, HGATConfig


def export(model: RecurrentHGAT, dummy_input: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, (dummy_input,), path)


if __name__ == "__main__":
    cfg = HGATConfig()
    model = RecurrentHGAT({"junction": 6, "lane": 5, "vehicle": 3}, cfg)
    dummy = torch.zeros(1, cfg.num_actions)
    export(model, dummy, Path("model.onnx"))
