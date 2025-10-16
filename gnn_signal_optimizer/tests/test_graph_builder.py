from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from gso.graph.builder import GraphBuilder
from gso.sim.state_extractor import StructuredState


def make_state(step: int) -> StructuredState:
    junction_features = {"J0": np.ones(6, dtype=np.float32) * step, "J1": np.ones(6, dtype=np.float32) * (step + 1)}
    lane_features = {"L0": np.ones(5, dtype=np.float32), "L1": np.ones(5, dtype=np.float32) * 2}
    vehicle_features = {}
    legal = {"J0": np.array([1, 1]), "J1": np.array([1, 0])}
    return StructuredState(junction_features, lane_features, vehicle_features, legal)


def test_builder_id_stability():
    builder = GraphBuilder(lane_to_junction={"L0": "J0", "L1": "J1"})
    state1 = make_state(0)
    data1 = builder.build(state1, time=0.0)
    state2 = make_state(1)
    data2 = builder.build(state2, time=1.0)
    assert data1[("junction", "x")].shape == data2[("junction", "x")].shape
    assert builder.node_order["junction"] == ["J0", "J1"]
    assert builder.node_order["lane"] == ["L0", "L1"]


def test_builder_masks():
    builder = GraphBuilder(lane_to_junction={"L0": "J0"})
    state = make_state(0)
    data = builder.build(state, time=0.0)
    mask = data[("junction", "mask")]
    assert mask.sum().item() == 2
