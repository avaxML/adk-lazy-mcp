from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Iterator
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .config import RegistryConfig, ServerConfig


class ServerState(str, Enum):
    UNSEEN = "unseen"
    READY = "ready"
    DEGRADED = "degraded"
    CLOSED = "closed"


class ToolSchema(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]
    schema_hash: str = ""
    name_lower: str = ""
    description_lower: str = ""

    def model_post_init(self, __context: Any) -> None:
        if not self.schema_hash:
            self.schema_hash = (
                "sha256:"
                + hashlib.sha256(_canonical_json(self.input_schema).encode("utf-8")).hexdigest()
            )
        self.name_lower = self.name.lower()
        self.description_lower = self.description.lower()


class CatalogEntry(BaseModel):
    config: ServerConfig
    state: ServerState = ServerState.UNSEEN
    tools: dict[str, ToolSchema] = Field(default_factory=dict)
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

    def iter_entries(self) -> Iterator[tuple[str, CatalogEntry]]:
        return iter(self._entries.items())

    def server_names(self) -> list[str]:
        return sorted(self._entries.keys())

    async def hydrate_server(self, server: str, tools: list[dict[str, Any]]) -> None:
        mapped: dict[str, ToolSchema] = {}
        for t in tools:
            mapped[t["name"]] = ToolSchema(
                name=t["name"],
                description=t.get("description", ""),
                input_schema=t.get("inputSchema", {"type": "object", "properties": {}}),
            )
        version = self._catalog_hash(mapped)
        async with self._lock:
            entry = self._entries[server]
            entry.tools = mapped
            entry.refreshed_at = time.time()
            entry.version = version
            entry.last_error = None
            entry.state = ServerState.READY

    def mark_error(self, server: str, error: str) -> None:
        e = self._entries[server]
        e.last_error = error
        e.state = ServerState.DEGRADED

    def is_stale(self, server: str) -> bool:
        e = self._entries[server]
        if e.refreshed_at == 0.0:
            return True
        return (time.time() - e.refreshed_at) > self._cfg.summary_ttl_s

    def discover(self, query: str, server: str | None = None) -> list[dict[str, str]]:
        q = query.strip().lower()
        results: list[tuple[int, str, dict[str, str]]] = []
        entries = ((server, self._entries[server]),) if server else self._entries.items()
        for name, e in entries:
            if e.state != ServerState.READY:
                continue
            for tool in e.tools.values():
                score = self._score(q, tool)
                if score < 0:
                    continue
                results.append(
                    (
                        score,
                        tool.name,
                        {
                            "server": name,
                            "tool": tool.name,
                            "description": tool.description,
                        },
                    )
                )
        results.sort(key=lambda x: (-x[0], x[1]))
        return [r[2] for r in results]

    @staticmethod
    def _score(query: str, tool: ToolSchema) -> int:
        if not query:
            return 1
        name = tool.name_lower
        if name == query:
            return 100
        if name.startswith(query):
            return 80
        if query in name:
            return 60
        if query in tool.description_lower:
            return 40
        return -1

    @staticmethod
    def _catalog_hash(tools: dict[str, ToolSchema]) -> str:
        payload = "|".join(sorted(f"{k}:{v.description}" for k, v in tools.items()))
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    """Return a stable JSON string so semantically equal schemas hash the same."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
