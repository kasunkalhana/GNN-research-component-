from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")

from gso.model.mc_dropout import mc_dropout_inference


def test_mc_dropout_variance():
    model = torch.nn.Sequential(torch.nn.Dropout(p=0.5), torch.nn.Linear(4, 2))

    def forward_fn(m):
        return m(torch.ones(1, 4))

    _, var = mc_dropout_inference(model, forward_fn, passes=10)
    assert var.sum() > 0
