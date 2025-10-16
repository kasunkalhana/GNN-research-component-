"""FastAPI application exposing KPIs and control endpoints."""

from __future__ import annotations

import asyncio
from typing import Dict

from fastapi import Depends, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from ..sim.control_translator import ControlConfig
from ..sim.kpis import KPIState
from .broadcaster import Broadcaster
from .schemas import ActionSummary, ControlOverride, ControlPolicyUpdate, KPIModel, StateSummary


class AppState:
    def __init__(self):
        self.kpis = KPIState()
        self.actions: Dict[str, int] = {}
        self.uncertainty: Dict[str, float] = {}
        self.control_config = ControlConfig()
        self.broadcaster = Broadcaster()


def create_app() -> FastAPI:
    app = FastAPI(title="GNN Signal Optimizer API")
    state = AppState()
    app.state.runtime = state

    @app.get("/health")
    async def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/kpis/current", response_model=KPIModel)
    async def get_kpis() -> KPIModel:
        return KPIModel(**state.kpis.as_dict())

    @app.get("/state/summary", response_model=StateSummary)
    async def get_state() -> StateSummary:
        actions = [
            ActionSummary(junction_id=jid, action=act, uncertainty=state.uncertainty.get(jid, 0.0))
            for jid, act in state.actions.items()
        ]
        return StateSummary(time=float(state.kpis.steps), actions=actions, kpis=KPIModel(**state.kpis.as_dict()))

    @app.post("/override")
    async def override(payload: ControlOverride) -> Dict[str, str]:
        state.actions[payload.junction_id] = payload.phase
        return {"status": "override", "junction_id": payload.junction_id}

    @app.post("/control/policy")
    async def control_policy(payload: ControlPolicyUpdate) -> Dict[str, str]:
        state.control_config.fallback_mode = payload.fallback_mode
        state.control_config.uncertainty_threshold = payload.uncertainty_threshold
        return {"status": "updated"}

    @app.websocket("/ws/stream")
    async def ws_stream(socket: WebSocket):
        await socket.accept()
        queue = await state.broadcaster.register()
        try:
            while True:
                payload = await queue.get()
                await socket.send_json(payload)
        except WebSocketDisconnect:
            await state.broadcaster.unregister(queue)
        finally:
            await state.broadcaster.unregister(queue)

    async def push_update(message: Dict[str, object]) -> None:
        state.actions = {jid: message["actions"][jid] for jid in message.get("actions", {})}
        state.uncertainty = message.get("uncertainty", {})
        await state.broadcaster.broadcast(message)

    app.state.push_update = push_update
    return app


app = create_app()


__all__ = ["create_app", "app"]
