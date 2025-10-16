from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from gso.serve.api import create_app


def test_health_endpoint():
    app = create_app()
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_websocket_stream():
    app = create_app()
    client = TestClient(app)

    async def push():
        await app.state.push_update({"actions": {"J0": 1}, "uncertainty": {"J0": 0.5}})

    with client.websocket_connect("/ws/stream") as websocket:
        asyncio.get_event_loop().run_until_complete(push())
        message = websocket.receive_json()
        assert message["actions"]["J0"] == 1
