"""Wrapper around SUMO's TraCI interface."""

from __future__ import annotations

import contextlib
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:  # pragma: no cover - optional dependency
    import traci  # type: ignore
except Exception:  # pragma: no cover - gracefully degrade for environments without SUMO
    traci = None  # type: ignore


@dataclass
class JunctionState:
    """Traffic light controller state."""

    phase: str
    phase_index: int
    time_in_phase: float
    min_green: float
    max_green: float
    queues: Dict[str, float]


@dataclass
class LaneState:
    """Aggregated lane features."""

    vehicle_count: int
    mean_speed: float
    density: float
    length: float
    speed_limit: float


@dataclass
class VehicleState:
    """Tracked vehicle features."""

    lane_id: str
    position: float
    speed: float
    waiting_time: float
    vtype: str


@dataclass
class RawState:
    """Snapshot returned every SUMO step."""

    junctions: Dict[str, JunctionState] = field(default_factory=dict)
    lanes: Dict[str, LaneState] = field(default_factory=dict)
    vehicles: Dict[str, VehicleState] = field(default_factory=dict)
    time: float = 0.0


class TraciClient:
    """Lifecycle manager for SUMO and TraCI subscriptions."""

    def __init__(self, sumocfg: Path, gui: bool = False, step_length: float = 1.0):
        self.sumocfg = Path(sumocfg)
        self.gui = gui
        self.step_length = step_length
        self._process: Optional[subprocess.Popen[str]] = None
        self._connected: bool = False
        self._subscriptions_initialized = False

    @property
    def connected(self) -> bool:
        return self._connected

    def start(self) -> None:
        """Start SUMO and connect via TraCI."""

        if traci is None:  # pragma: no cover - runtime guard
            raise RuntimeError("TraCI is not available in this environment")
        binary = "sumo-gui" if self.gui else "sumo"
        cmd = [binary, "-c", str(self.sumocfg), "--step-length", str(self.step_length)]
        env = os.environ.copy()
        env.setdefault("SUMO_HOME", str(self.sumocfg.parent.resolve()))
        self._process = subprocess.Popen(cmd, env=env)
        traci.start(["traci", "--step-length", str(self.step_length)], label="gnn")
        self._connected = True
        self._init_subscriptions()

    def close(self) -> None:
        """Close TraCI session and terminate SUMO."""

        if traci is not None and self._connected:  # pragma: no branch - best effort cleanup
            with contextlib.suppress(Exception):
                traci.close()
        if self._process is not None:
            with contextlib.suppress(Exception):
                self._process.terminate()
        self._connected = False

    # context manager support
    def __enter__(self) -> "TraciClient":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - resource cleanup
        self.close()

    # subscription helpers -------------------------------------------------
    def _init_subscriptions(self) -> None:
        if traci is None:
            return
        if self._subscriptions_initialized:
            return
        for lane_id in traci.lane.getIDList():
            traci.lane.subscribe(lane_id, (traci.constants.LANE_LAST_STEP_MEAN_SPEED, traci.constants.LANE_LAST_STEP_VEHICLE_NUMBER))
        for tls_id in traci.trafficlight.getIDList():
            traci.trafficlight.subscribe(tls_id, (traci.constants.TL_CURRENT_PHASE, traci.constants.TL_NEXT_SWITCH))
        for veh_id in traci.vehicle.getIDList():
            traci.vehicle.subscribe(veh_id, (traci.constants.VAR_LANE_ID, traci.constants.VAR_POSITION, traci.constants.VAR_SPEED, traci.constants.VAR_WAITING_TIME))
        self._subscriptions_initialized = True

    # public api ------------------------------------------------------------
    def step(self) -> RawState:
        """Advance the simulation by one step and return a snapshot."""

        if traci is None:  # pragma: no cover
            raise RuntimeError("TraCI is not available")
        traci.simulationStep()
        time = traci.simulation.getTime()
        lanes = self._collect_lanes()
        junctions = self._collect_tls(lanes)
        vehicles = self._collect_vehicles()
        return RawState(junctions=junctions, lanes=lanes, vehicles=vehicles, time=time)

    def _collect_lanes(self) -> Dict[str, LaneState]:
        if traci is None:
            return {}
        lane_states: Dict[str, LaneState] = {}
        for lane_id in traci.lane.getIDList():
            veh_count = traci.lane.getLastStepVehicleNumber(lane_id)
            mean_speed = traci.lane.getLastStepMeanSpeed(lane_id)
            length = traci.lane.getLength(lane_id)
            speed_limit = traci.lane.getMaxSpeed(lane_id)
            density = veh_count / max(length, 1e-3)
            lane_states[lane_id] = LaneState(
                vehicle_count=veh_count,
                mean_speed=mean_speed,
                density=density,
                length=length,
                speed_limit=speed_limit,
            )
        return lane_states

    def _collect_tls(self, lanes: Dict[str, LaneState]) -> Dict[str, JunctionState]:
        if traci is None:
            return {}
        tls_states: Dict[str, JunctionState] = {}
        for tls_id in traci.trafficlight.getIDList():
            phase = traci.trafficlight.getRedYellowGreenState(tls_id)
            phase_index = traci.trafficlight.getPhase(tls_id)
            next_switch = traci.trafficlight.getNextSwitch(tls_id)
            current_time = traci.simulation.getTime()
            time_in_phase = max(0.0, next_switch - current_time)
            logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0]
            min_green = min(p.duration for p in logic.phases if "y" not in p.state.lower())
            max_green = max(p.maxDuration for p in logic.phases if p.maxDuration > 0)
            queues = {lane_id: lanes[lane_id].vehicle_count for lane_id in traci.trafficlight.getControlledLanes(tls_id) if lane_id in lanes}
            tls_states[tls_id] = JunctionState(
                phase=phase,
                phase_index=phase_index,
                time_in_phase=time_in_phase,
                min_green=min_green,
                max_green=max_green,
                queues=queues,
            )
        return tls_states

    def _collect_vehicles(self) -> Dict[str, VehicleState]:
        if traci is None:
            return {}
        vehicle_states: Dict[str, VehicleState] = {}
        for veh_id in traci.vehicle.getIDList():
            lane_id = traci.vehicle.getLaneID(veh_id)
            position = traci.vehicle.getLanePosition(veh_id)
            speed = traci.vehicle.getSpeed(veh_id)
            waiting_time = traci.vehicle.getWaitingTime(veh_id)
            vtype = traci.vehicle.getTypeID(veh_id)
            vehicle_states[veh_id] = VehicleState(
                lane_id=lane_id,
                position=position,
                speed=speed,
                waiting_time=waiting_time,
                vtype=vtype,
            )
        return vehicle_states


__all__ = [
    "TraciClient",
    "RawState",
    "JunctionState",
    "LaneState",
    "VehicleState",
]
