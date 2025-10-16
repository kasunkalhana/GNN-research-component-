"""Pydantic models for FastAPI I/O."""

from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel


class KPIModel(BaseModel):
    travel_time: float
    delay: float
    stops: float
    throughput: float
    emissions_proxy: float
    steps: float


class ActionSummary(BaseModel):
    junction_id: str
    action: int
    uncertainty: float


class ControlOverride(BaseModel):
    junction_id: str
    phase: int
    duration: float = 10.0


class ControlPolicyUpdate(BaseModel):
    fallback_mode: str
    uncertainty_threshold: float


class StateSummary(BaseModel):
    time: float
    actions: List[ActionSummary]
    kpis: KPIModel


__all__ = [
    "KPIModel",
    "ActionSummary",
    "ControlOverride",
    "ControlPolicyUpdate",
    "StateSummary",
]
