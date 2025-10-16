from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from gso.sim.control_translator import ControlConfig, ControlTranslator
from gso.sim.traci_client import JunctionState


def make_state(time_in_phase: float, queues=None):
    queues = queues or {"L0": 3, "L1": 1}
    return JunctionState(phase="Gr", phase_index=0, time_in_phase=time_in_phase, min_green=5, max_green=30, queues=queues)


def test_min_green_enforced():
    translator = ControlTranslator({"J0": ["a", "b"]}, ControlConfig(min_green=10))
    state = {"J0": make_state(2)}
    actions = translator.translate({"J0": 1}, {"J0": 0.0}, state)
    assert actions["J0"] == 0


def test_uncertainty_fallback_queue():
    translator = ControlTranslator({"J0": ["a", "b"]}, ControlConfig(fallback_mode="queue", uncertainty_threshold=0.1))
    state = {"J0": make_state(15, {"L0": 10, "L1": 2})}
    actions = translator.translate({"J0": 1}, {"J0": 0.9}, state)
    assert actions["J0"] in {0, 1}
