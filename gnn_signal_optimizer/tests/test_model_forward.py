from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from gso.graph.builder import GraphBuilder
from gso.model.hgat import HGATConfig, RecurrentHGAT
from gso.sim.state_extractor import StructuredState


def test_model_forward_shapes():
    builder = GraphBuilder(lane_to_junction={"L0": "J0"})
    struct = StructuredState(
        junction_features={"J0": np.ones(6, dtype=np.float32)},
        lane_features={"L0": np.ones(5, dtype=np.float32)},
        vehicle_features={},
        legal_actions={"J0": np.array([1, 1], dtype=np.float32)},
    )
    data = builder.build(struct, time=0.0)
    model = RecurrentHGAT({"junction": 6, "lane": 5, "vehicle": 3}, HGATConfig(hidden_dim=16, num_actions=2))
    logits, hidden = model(data)
    assert logits.shape == (1, 2)
    assert hidden.shape == (1, 16)
