"""Translate network actions into SUMO-compatible TLS commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .traci_client import JunctionState


@dataclass
class ControlConfig:
    """Safety thresholds and fallback logic."""

    min_green: float = 5.0
    max_green: float = 60.0
    yellow_duration: float = 3.0
    uncertainty_threshold: float = 0.25
    fallback_mode: str = "queue"


class ControlTranslator:
    """Apply safety guards and fallbacks on top of policy outputs."""

    def __init__(self, phases: Dict[str, list[str]], config: ControlConfig):
        self.phases = phases
        self.config = config

    def translate(
        self,
        desired_phase: Dict[str, int],
        uncertainty: Dict[str, float],
        junction_state: Dict[str, JunctionState],
    ) -> Dict[str, int]:
        actions: Dict[str, int] = {}
        for jid, state in junction_state.items():
            phases = self.phases.get(jid, [])
            if not phases:
                continue
            proposed = desired_phase.get(jid, state.phase_index)
            guard_action = self._enforce_timing(jid, state, proposed)
            guard_action = self._enforce_uncertainty(jid, guard_action, uncertainty.get(jid, 0.0), state)
            actions[jid] = guard_action % len(phases)
        return actions

    def _enforce_timing(self, jid: str, state: JunctionState, proposed: int) -> int:
        phases = self.phases.get(jid, [])
        if not phases:
            return proposed
        current_idx = state.phase_index % len(phases)
        if state.time_in_phase < self.config.min_green:
            return current_idx
        if state.time_in_phase > self.config.max_green:
            return (current_idx + 1) % len(phases)
        return proposed

    def _enforce_uncertainty(
        self,
        jid: str,
        proposed: int,
        uncertainty: float,
        state: JunctionState,
    ) -> int:
        phases = self.phases.get(jid, [])
        if not phases:
            return proposed
        if uncertainty <= self.config.uncertainty_threshold:
            return proposed
        if self.config.fallback_mode == "fixed":
            return 0
        if self.config.fallback_mode == "queue":
            if not state.queues:
                return proposed
            busiest_lane = max(state.queues.items(), key=lambda kv: kv[1])[0]
            target = hash(busiest_lane) % len(phases)
            return target
        if self.config.fallback_mode == "external":
            return proposed
        return proposed


__all__ = ["ControlTranslator", "ControlConfig"]
