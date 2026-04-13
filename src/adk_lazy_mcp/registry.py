from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Awaitable, Callable
from typing import Any

from .catalog import CatalogEntry, CatalogManager, ServerState, ToolSchema
from .config import RegistryConfig, ServerConfig
from .errors import PolicyDeniedError, RegistryClosedError, ToolNotFoundError, ValidationError
from .policy import PolicyEngine
from .results import normalize_result
from .session_manager import SessionManager
from .telemetry import Telemetry

McpExecutor = Callable[[str, str, dict[str, Any]], Awaitable[dict[str, Any]]]
McpLister = Callable[[str], Awaitable[list[dict[str, Any]]]]

_TYPE_MAP: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "object": dict,
    "array": list,
    "null": type(None),
}


class Registry:
    def __init__(
        self,
        server_configs: list[ServerConfig],
        cfg: RegistryConfig,
        *,
        list_tools: McpLister,
        execute_tool: McpExecutor,
        policy: PolicyEngine | None = None,
        telemetry: Telemetry | None = None,
    ) -> None:
        duplicate_names = [s.name for s in server_configs]
        if len(duplicate_names) != len(set(duplicate_names)):
            raise ValueError("server_configs must use unique server names")
        self.cfg = cfg
        self._list_tools = list_tools
        self._execute_tool = execute_tool
        self._policy = policy or PolicyEngine()
        self._telemetry = telemetry or Telemetry()
        self._catalog = CatalogManager(cfg)
        self._sessions: dict[str, SessionManager] = {}
        self._server_configs = {s.name: s for s in server_configs}
        self._closed = False
        self._warm_task: asyncio.Task[None] | None = None
        self._hydrate_locks: dict[str, asyncio.Lock] = {}
        for s in server_configs:
            self._catalog.register_server(s)
            self._sessions[s.name] = SessionManager(s)
            self._hydrate_locks[s.name] = asyncio.Lock()

    async def start(self) -> None:
        if self.cfg.warm_mode == "background":
            self._warm_task = asyncio.create_task(self._hydrate_all())
        elif self.cfg.warm_mode == "eager":
            await self._hydrate_all()

    async def _hydrate_all(self) -> None:
        await asyncio.gather(
            *(self._hydrate_server(name) for name in self._server_configs),
            return_exceptions=True,
        )

    async def _hydrate_server(self, name: str) -> None:
        lock = self._hydrate_locks[name]
        async with lock:
            cfg = self._server_configs[name]
            decision = self._policy.validate_server(cfg)
            if not decision.allowed:
                self._catalog.mark_error(name, decision.reason or "policy_denied")
                return
            try:
                tools = await self._list_tools(name)
                await self._catalog.hydrate_server(name, tools)
                self._telemetry.incr(f"catalog_refresh_total:{name}:success")
            except Exception as exc:
                self._catalog.mark_error(name, str(exc))
                self._telemetry.incr(f"catalog_refresh_total:{name}:error")

    async def _ensure_fresh(self, server: str) -> None:
        entry = self._catalog.get_entry(server)
        if entry.state is ServerState.UNSEEN or self._catalog.is_stale(server):
            await self._hydrate_server(server)

    async def discover(self, query: str = "", server: str = "", limit: int = 20) -> dict[str, Any]:
        self._ensure_open()
        cap = min(max(limit, 1), self.cfg.hard_discover_cap, self.cfg.max_discover_results)
        if server:
            if server not in self._server_configs:
                raise ToolNotFoundError(f"Unknown server: {server}")
            await self._ensure_fresh(server)
        else:
            await asyncio.gather(
                *(self._ensure_fresh(name) for name in self._catalog.server_names()),
                return_exceptions=False,
            )
        matches = self._catalog.discover(query, server or None)
        unavailable: list[str] = []
        for name, entry in self._catalog.iter_entries():
            if entry.state in {ServerState.DEGRADED, ServerState.UNSEEN}:
                unavailable.append(name)
        returned = matches[:cap]
        has_more = len(matches) > len(returned)
        return {
            "status": "success",
            "tools": returned,
            "total_matches": len(matches),
            "returned": len(returned),
            "has_more": has_more,
            "servers_considered": [server] if server else self._catalog.server_names(),
            "unavailable_servers": unavailable,
            "hint": (
                "Narrow the query or pass a server name to reduce results."
                if has_more
                else "Call inspect_mcp_tool before executing so you can see the input schema."
            ),
        }

    async def inspect(self, server: str, tool: str, refresh: bool = False) -> dict[str, Any]:
        self._ensure_open()
        if server not in self._server_configs:
            raise ToolNotFoundError(f"Unknown server: {server}")
        if refresh or self._catalog.is_stale(server):
            await self._hydrate_server(server)
        entry = self._catalog.get_entry(server)
        ts = entry.tools.get(tool)
        if not ts:
            raise ToolNotFoundError(f"Unknown tool: {server}/{tool}")
        return self._format_inspection(entry, ts)

    async def execute(
        self,
        server: str,
        tool: str,
        arguments: dict[str, Any] | None = None,
        timeout_ms: int | None = None,
    ) -> dict[str, Any]:
        self._ensure_open()
        if server not in self._server_configs:
            raise ToolNotFoundError(f"Unknown server: {server}")

        args = arguments or {}
        cfg = self._server_configs[server]

        pol = self._policy.validate_tool(cfg, tool)
        if not pol.allowed:
            raise PolicyDeniedError(pol.reason or "policy_denied")

        # Fast-path tool lookup: hydrate only if we don't know it yet.
        entry = self._catalog.get_entry(server)
        ts = entry.tools.get(tool)
        if ts is None or self._catalog.is_stale(server):
            await self._hydrate_server(server)
            entry = self._catalog.get_entry(server)
            ts = entry.tools.get(tool)
        if ts is None:
            raise ToolNotFoundError(f"Unknown tool: {server}/{tool}")

        if self.cfg.enable_client_validation:
            err = _validate_args(args, ts.input_schema)
            if err:
                raise ValidationError(err)
        effective_timeout_ms = cfg.call_timeout_ms if timeout_ms is None else timeout_ms
        if effective_timeout_ms <= 0:
            raise ValidationError("timeout_ms must be positive")

        session = self._sessions[server]
        started = time.perf_counter()
        try:
            raw = await session.execute(
                lambda: self._execute_tool(server, tool, args),
                timeout_ms=effective_timeout_ms,
                allow_retry=cfg.trusted,
            )
            if not isinstance(raw, dict):
                raise TypeError("execute_tool must return a dict-like MCP result")
        finally:
            duration_ms = (time.perf_counter() - started) * 1_000
            self._telemetry.time_ms(f"execute_duration_ms:{server}:{tool}", duration_ms)
        return {
            "status": "success",
            "server": server,
            "tool": tool,
            "duration_ms": int(duration_ms),
            "tool_result": normalize_result(raw, max_inline_bytes=cfg.max_inline_bytes),
        }

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._warm_task:
            self._warm_task.cancel()
            with contextlib.suppress(BaseException):
                await self._warm_task
        await asyncio.gather(
            *(s.close() for s in self._sessions.values()),
            return_exceptions=True,
        )

    def health_snapshot(self) -> dict[str, Any]:
        servers: list[dict[str, Any]] = []
        status = "ready"
        for name, entry in self._catalog.iter_entries():
            session = self._sessions[name]
            servers.append(
                {
                    "name": name,
                    "state": entry.state.value,
                    "tool_count": len(entry.tools),
                    "breaker": "open" if session.breaker_open else "closed",
                    "last_error": entry.last_error,
                }
            )
            if entry.state is ServerState.DEGRADED or session.breaker_open:
                status = "degraded"
        return {"status": status, "servers": servers}

    def _ensure_open(self) -> None:
        if self._closed:
            raise RegistryClosedError("registry_closed")

    @staticmethod
    def _format_inspection(entry: CatalogEntry, ts: ToolSchema) -> dict[str, Any]:
        return {
            "status": "success",
            "server": entry.config.name,
            "tool": ts.name,
            "description": ts.description,
            "input_schema": ts.input_schema,
            "required_fields": list(ts.input_schema.get("required", [])),
            "accepts_no_args": not bool(ts.input_schema.get("required")),
            "schema_hash": ts.schema_hash,
            "refreshed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(entry.refreshed_at)),
            "hint": "Call execute_mcp_tool with an arguments object that matches this schema exactly.",
        }


def _validate_args(args: dict[str, Any], schema: dict[str, Any]) -> str | None:
    if not isinstance(args, dict):
        return "arguments must be an object"
    required = schema.get("required") or []
    for field_name in required:
        if field_name not in args:
            return f"{field_name} is a required property"
    props = schema.get("properties") or {}
    for key, value in args.items():
        prop = props.get(key)
        if not prop:
            continue
        expected = prop.get("type")
        if not expected:
            continue
        py_type = _TYPE_MAP.get(expected)
        if py_type is None:
            continue
        # JSON booleans are a subtype of int in Python; disallow that coercion.
        if expected in {"integer", "number"} and isinstance(value, bool):
            return f"{key} should be of type {expected}"
        if not isinstance(value, py_type):
            return f"{key} should be of type {expected}"
    return None
