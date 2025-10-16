"""Key performance indicator computations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from .traci_client import RawState


@dataclass
class KPIState:
    """Maintain aggregated KPI values."""

    travel_time: float = 0.0
    delay: float = 0.0
    stops: int = 0
    throughput: int = 0
    emissions_proxy: float = 0.0
    steps: int = 0

    def update(self, snapshot: RawState) -> None:
        lane_speeds = np.array([lane.mean_speed for lane in snapshot.lanes.values()], dtype=np.float32)
        lane_counts = np.array([lane.vehicle_count for lane in snapshot.lanes.values()], dtype=np.float32)
        if lane_counts.size:
            self.delay += float(np.maximum(0.0, 13.9 - lane_speeds).sum())
            self.throughput += int(lane_counts.sum())
            self.emissions_proxy += float(lane_speeds.var())
        vehicle_waits = [veh.waiting_time for veh in snapshot.vehicles.values()]
        self.stops += sum(1 for w in vehicle_waits if w > 1.0)
        self.travel_time += float(np.sum(vehicle_waits))
        self.steps += 1

    def as_dict(self) -> Dict[str, float]:
        return {
            "travel_time": self.travel_time,
            "delay": self.delay,
            "stops": float(self.stops),
            "throughput": float(self.throughput),
            "emissions_proxy": self.emissions_proxy,
            "steps": float(self.steps),
        }


__all__ = ["KPIState"]
