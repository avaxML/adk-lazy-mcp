from __future__ import annotations

import asyncio
import hashlib
import time
from typing import Any, Awaitable, Callable


from .catalog import CatalogManager, ServerState
from .config import RegistryConfig, ServerConfig
from .errors import PolicyDeniedError, RegistryClosedError, ToolNotFoundError, ValidationError
from .policy import PolicyEngine
from .results import normalize_result
from .session_manager import SessionManager
from .telemetry import Telemetry

McpExecutor = Callable[[str, str, dict[str, Any]], Awaitable[dict[str, Any]]]
McpLister = Callable[[str], Awaitable[list[dict[str, Any]]]]


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
        for s in server_configs:
            self._catalog.register_server(s)
            self._sessions[s.name] = SessionManager(s)

    async def start(self) -> None:
        if self.cfg.warm_mode == "background":
            self._warm_task = asyncio.create_task(self._hydrate_all())
        elif self.cfg.warm_mode == "eager":
            await self._hydrate_all()

    async def _hydrate_all(self) -> None:
        await asyncio.gather(*(self._hydrate_server(name) for name in self._server_configs), return_exceptions=True)

    async def _hydrate_server(self, name: str) -> None:
        cfg = self._server_configs[name]
        decision = self._policy.validate_server(cfg)
        if not decision.allowed:
            self._catalog.mark_error(name, decision.reason or "policy_denied")
            return
        try:
            tools = await self._list_tools(name)
            await self._catalog.hydrate_server(name, tools)
            self._telemetry.incr(f"catalog_refresh_total:{name}:success")
        except Exception as exc:  # noqa: BLE001
            self._catalog.mark_error(name, str(exc))
            self._telemetry.incr(f"catalog_refresh_total:{name}:error")

    async def discover(self, query: str = "", server: str = "", limit: int = 20) -> dict[str, Any]:
        self._ensure_open()
        cap = min(max(limit, 1), self.cfg.hard_discover_cap, self.cfg.max_discover_results)
        if server and self._catalog.get_entry(server).state in {ServerState.UNSEEN, ServerState.STALE}:
            await self._hydrate_server(server)
        matches = self._catalog.discover(query, server or None)
        unavailable = [name for name, e in self._catalog._entries.items() if e.state in {ServerState.DEGRADED, ServerState.HYDRATING}]
        returned = matches[:cap]
        return {
            "status": "success",
            "tools": returned,
            "total_matches": len(matches),
            "returned": len(returned),
            "has_more": len(matches) > len(returned),
            "servers_considered": [server] if server else sorted(self._server_configs.keys()),
            "unavailable_servers": unavailable,
            "hint": "Refine the query or specify a server before inspecting a tool." if len(matches) > len(returned) else "Inspect a tool before executing.",
        }

    async def inspect(self, server: str, tool: str, refresh: bool = False) -> dict[str, Any]:
        self._ensure_open()
        if refresh or self._catalog.is_stale(server):
            await self._hydrate_server(server)
        entry = self._catalog.get_entry(server)
        ts = entry.tools.get(tool)
        if not ts:
            raise ToolNotFoundError(f"Unknown tool: {server}/{tool}")
        schema_hash = "sha256:" + hashlib.sha256(str(ts.input_schema).encode("utf-8")).hexdigest()
        return {
            "status": "success",
            "server": server,
            "tool": tool,
            "input_schema": ts.input_schema,
            "required_fields": ts.input_schema.get("required", []),
            "accepts_no_args": not bool(ts.input_schema.get("required")),
            "schema_hash": schema_hash,
            "refreshed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(entry.refreshed_at)),
            "hint": "Call execute_mcp_tool with an arguments object that matches this schema exactly.",
        }

    async def execute(
        self,
        server: str,
        tool: str,
        arguments: dict[str, Any] | None = None,
        timeout_ms: int | None = None,
    ) -> dict[str, Any]:
        self._ensure_open()
        args = arguments or {}
        cfg = self._server_configs[server]
        pol = self._policy.validate_tool(cfg, tool)
        if not pol.allowed:
            raise PolicyDeniedError(pol.reason or "policy_denied")

        inspection = await self.inspect(server, tool)
        if self.cfg.enable_client_validation:
            err = _validate_args(args, inspection["input_schema"])
            if err:
                raise ValidationError(err)

        session = self._sessions[server]
        timer = self._telemetry.timer(f"execute_duration_ms:{server}:{tool}")
        started = time.perf_counter()
        raw = await session.execute(
            lambda: self._execute_tool(server, tool, args),
            timeout_ms=timeout_ms or cfg.call_timeout_ms,
            allow_retry=cfg.trusted,
        )
        timer()
        duration_ms = int((time.perf_counter() - started) * 1000)
        return {
            "status": "success",
            "server": server,
            "tool": tool,
            "duration_ms": duration_ms,
            "tool_result": normalize_result(raw, max_inline_bytes=cfg.max_inline_bytes),
        }

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._warm_task:
            self._warm_task.cancel()
            with contextlib.suppress(Exception):
                await self._warm_task
        await asyncio.gather(*(s.close() for s in self._sessions.values()))

    def health_snapshot(self) -> dict[str, Any]:
        servers = []
        status = "ready"
        for name, entry in self._catalog._entries.items():
            servers.append(
                {
                    "name": name,
                    "state": entry.state.value,
                    "tool_count": len(entry.tools),
                    "breaker": "open" if self._sessions[name]._breaker.open else "closed",
                    "last_error": entry.last_error,
                }
            )
            if entry.state in {ServerState.DEGRADED, ServerState.COOLING_OFF}:
                status = "degraded"
        return {"status": status, "servers": servers}

    def _ensure_open(self) -> None:
        if self._closed:
            raise RegistryClosedError("registry_closed")


import contextlib  # noqa: E402


def _validate_args(args: dict[str, Any], schema: dict[str, Any]) -> str | None:
    required = schema.get("required", [])
    props = schema.get("properties", {})
    for field in required:
        if field not in args:
            return f"{field} is a required property"
    type_map = {"string": str, "number": (int, float), "integer": int, "boolean": bool, "object": dict, "array": list}
    for key, value in args.items():
        expected = props.get(key, {}).get("type")
        if expected and expected in type_map and not isinstance(value, type_map[expected]):
            return f"{key} should be of type {expected}"
    return None
