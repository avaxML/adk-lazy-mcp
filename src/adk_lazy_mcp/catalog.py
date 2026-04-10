from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .config import RegistryConfig, ServerConfig


class ServerState(str, Enum):
    UNSEEN = "unseen"
    HYDRATING = "hydrating"
    READY = "ready"
    DEGRADED = "degraded"
    STALE = "stale"
    COOLING_OFF = "cooling_off"
    CLOSED = "closed"


@dataclass
class ToolSchema:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class CatalogEntry:
    config: ServerConfig
    state: ServerState = ServerState.UNSEEN
    tools: dict[str, ToolSchema] = field(default_factory=dict)
    refreshed_at: float = 0.0
    last_error: str | None = None
    version: str = ""


class CatalogManager:
    def __init__(self, registry_config: RegistryConfig) -> None:
        self._cfg = registry_config
        self._entries: dict[str, CatalogEntry] = {}
        self._lock = asyncio.Lock()

    def register_server(self, server: ServerConfig) -> None:
        self._entries[server.name] = CatalogEntry(config=server)

    def get_entry(self, server: str) -> CatalogEntry:
        return self._entries[server]

    async def hydrate_server(self, server: str, tools: list[dict[str, Any]]) -> None:
        async with self._lock:
            entry = self._entries[server]
            entry.state = ServerState.HYDRATING
            mapped: dict[str, ToolSchema] = {}
            for t in tools:
                mapped[t["name"]] = ToolSchema(
                    name=t["name"],
                    description=t.get("description", ""),
                    input_schema=t.get("inputSchema", {"type": "object", "properties": {}}),
                )
            entry.tools = mapped
            entry.refreshed_at = time.time()
            entry.version = self._catalog_hash(mapped)
            entry.last_error = None
            entry.state = ServerState.READY

    def mark_error(self, server: str, error: str) -> None:
        e = self._entries[server]
        e.last_error = error
        e.state = ServerState.DEGRADED

    def is_stale(self, server: str) -> bool:
        e = self._entries[server]
        return (time.time() - e.refreshed_at) > self._cfg.summary_ttl_s

    def discover(self, query: str, server: str | None = None) -> list[dict[str, str]]:
        q = query.strip().lower()
        results: list[tuple[int, dict[str, str]]] = []
        for name, e in self._entries.items():
            if server and name != server:
                continue
            for tool in e.tools.values():
                score = self._score(q, tool)
                if score >= 0:
                    results.append(
                        (
                            score,
                            {
                                "server": name,
                                "tool": tool.name,
                                "description": tool.description,
                            },
                        )
                    )
        results.sort(key=lambda x: (-x[0], x[1]["tool"]))
        return [x[1] for x in results]

    @staticmethod
    def _score(query: str, tool: ToolSchema) -> int:
        if not query:
            return 1
        name = tool.name.lower()
        desc = tool.description.lower()
        if name == query:
            return 100
        if name.startswith(query):
            return 80
        if query in name:
            return 60
        if query in desc:
            return 40
        return -1

    @staticmethod
    def _catalog_hash(tools: dict[str, ToolSchema]) -> str:
        payload = "|".join(sorted(f"{k}:{v.description}" for k, v in tools.items()))
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()
