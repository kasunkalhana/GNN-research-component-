"""Convert raw TraCI subscriptions into structured tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .traci_client import JunctionState, LaneState, RawState


@dataclass
class StructuredState:
    """Structured features ready for graph construction."""

    junction_features: Dict[str, np.ndarray]
    lane_features: Dict[str, np.ndarray]
    vehicle_features: Dict[str, np.ndarray]
    legal_actions: Dict[str, np.ndarray]


class StateExtractor:
    """Convert :class:`RawState` into numerical features."""

    def __init__(self, phases: Dict[str, List[str]]):
        self.phases = phases

    def extract(self, raw: RawState) -> StructuredState:
        junction_features = {jid: self._junction_features(jid, state) for jid, state in raw.junctions.items()}
        lane_features = {lid: self._lane_features(state) for lid, state in raw.lanes.items()}
        vehicle_features = {vid: self._vehicle_features(state) for vid, state in raw.vehicles.items()}
        legal_actions = {jid: self._legal_action_mask(jid, raw.junctions.get(jid)) for jid in raw.junctions}
        return StructuredState(
            junction_features=junction_features,
            lane_features=lane_features,
            vehicle_features=vehicle_features,
            legal_actions=legal_actions,
        )

    # feature helpers ------------------------------------------------------
    def _junction_features(self, jid: str, state: JunctionState) -> np.ndarray:
        queue_values = np.array(list(state.queues.values()), dtype=np.float32)
        q_mean = float(queue_values.mean()) if queue_values.size else 0.0
        q_std = float(queue_values.std()) if queue_values.size else 0.0
        feats = np.array(
            [
                q_mean,
                q_std,
                float(state.phase_index),
                float(state.time_in_phase),
                float(sum(state.queues.values())),
                float(len(state.queues)),
            ],
            dtype=np.float32,
        )
        return feats

    def _lane_features(self, state: LaneState) -> np.ndarray:
        return np.array(
            [
                float(state.vehicle_count),
                float(state.mean_speed),
                float(state.density),
                float(state.length),
                float(state.speed_limit),
            ],
            dtype=np.float32,
        )

    def _vehicle_features(self, state) -> np.ndarray:
        return np.array(
            [
                float(state.position),
                float(state.speed),
                float(state.waiting_time),
            ],
            dtype=np.float32,
        )

    def _legal_action_mask(self, jid: str, state: JunctionState | None) -> np.ndarray:
        phases = self.phases.get(jid, [])
        mask = np.zeros(len(phases), dtype=np.float32)
        if not phases:
            return mask
        if state is None:
            mask[:] = 1.0
            return mask
        # simple rule: only allow next phase index or staying
        mask[state.phase_index % len(phases)] = 1.0
        mask[(state.phase_index + 1) % len(phases)] = 1.0
        return mask


__all__ = ["StateExtractor", "StructuredState"]
