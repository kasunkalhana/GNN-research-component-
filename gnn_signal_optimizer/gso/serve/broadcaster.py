"""Simple websocket broadcaster using asyncio queues."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List


class Broadcaster:
    def __init__(self):
        self.connections: List[asyncio.Queue] = []
        self._lock = asyncio.Lock()

    async def register(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        async with self._lock:
            self.connections.append(queue)
        return queue

    async def unregister(self, queue: asyncio.Queue) -> None:
        async with self._lock:
            if queue in self.connections:
                self.connections.remove(queue)

    async def broadcast(self, message: Dict[str, Any]) -> None:
        async with self._lock:
            for queue in self.connections:
                await queue.put(message)


__all__ = ["Broadcaster"]
