from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from .config import RegistryConfig, ServerConfig
from .errors import LazyMCPError, PolicyDeniedError, ToolNotFoundError, ValidationError
from .policy import PolicyEngine
from .registry import Registry
from .telemetry import Telemetry


class LazyMCPToolset:
    """ADK-native wrapper exposing three meta-tools for MCP interactions.

    The toolset is designed to plug into Google ADK's ``BaseToolset`` contract:
    ``get_tools`` accepts an optional ``readonly_context`` and returns bound
    callables whose signatures and docstrings the ADK auto-function-calling
    layer turns into model-facing tool descriptions.
    """

    def __init__(
        self,
        server_configs: list[ServerConfig],
        registry_config: RegistryConfig | None = None,
        *,
        list_tools: Callable[[str], Awaitable[list[dict[str, Any]]]],
        execute_tool: Callable[[str, str, dict[str, Any]], Awaitable[dict[str, Any]]],
        policy: PolicyEngine | None = None,
        telemetry: Telemetry | None = None,
    ) -> None:
        self._registry = Registry(
            server_configs,
            registry_config or RegistryConfig(),
            list_tools=list_tools,
            execute_tool=execute_tool,
            policy=policy,
            telemetry=telemetry,
        )
        self._started = False
        self._start_lock = asyncio.Lock()

    async def get_tools(self, readonly_context: Any = None) -> list[Callable[..., Any]]:
        """Return the three meta-tools exposed to the model.

        Accepts an optional ``readonly_context`` to match Google ADK's
        ``BaseToolset.get_tools`` contract. The first call lazily starts the
        underlying registry (and, depending on ``warm_mode``, may schedule
        background hydration).
        """
        if not self._started:
            async with self._start_lock:
                if not self._started:
                    await self._registry.start()
                    self._started = True
        return [self.discover_mcp_tools, self.inspect_mcp_tool, self.execute_mcp_tool]

    async def discover_mcp_tools(
        self,
        query: str = "",
        server: str = "",
        limit: int = 20,
    ) -> dict[str, Any]:
        """Search the MCP catalog for tools matching a query.

        Use this as the first step of the three-step MCP workflow. The
        response contains a ranked list of ``{server, tool, description}``
        entries plus pagination hints. Pass ``server`` to restrict the
        search to a single MCP server, and ``limit`` to cap the number of
        results (the registry enforces a hard upper bound).

        Args:
            query: Case-insensitive keyword to match against tool names and
                descriptions. Leave empty to list everything.
            server: Optional MCP server name to restrict the search.
            limit: Maximum number of tools to return (default 20).

        Returns:
            A dict with keys ``status``, ``tools``, ``total_matches``,
            ``returned``, ``has_more``, ``servers_considered``,
            ``unavailable_servers``, and ``hint``.
        """
        try:
            return await self._registry.discover(query=query, server=server, limit=limit)
        except ToolNotFoundError as exc:
            return self._error("tool_not_found", str(exc), True, server=server)
        except LazyMCPError as exc:
            return self._error("discover_error", str(exc), True)

    async def inspect_mcp_tool(
        self,
        server: str,
        tool: str,
        refresh: bool = False,
    ) -> dict[str, Any]:
        """Return the input schema and metadata for a single MCP tool.

        Call this before the first execution of a tool so the model can
        build a valid arguments object. The response includes the full
        JSON schema, a list of required fields, and a stable
        ``schema_hash`` suitable for caching.

        Args:
            server: MCP server name (as returned by ``discover_mcp_tools``).
            tool: Tool name on that server.
            refresh: If True, force a catalog refresh before reading the
                schema. Use sparingly - the registry already refreshes
                stale entries automatically.

        Returns:
            A dict with keys ``status``, ``server``, ``tool``,
            ``description``, ``input_schema``, ``required_fields``,
            ``accepts_no_args``, ``schema_hash``, ``refreshed_at``, and
            ``hint``. On failure, returns an error envelope.
        """
        try:
            return await self._registry.inspect(server=server, tool=tool, refresh=refresh)
        except ToolNotFoundError as exc:
            return self._error("tool_not_found", str(exc), True, server=server, tool=tool)
        except LazyMCPError as exc:
            return self._error("inspect_error", str(exc), True, server=server, tool=tool)

    async def execute_mcp_tool(
        self,
        server: str,
        tool: str,
        arguments: dict[str, Any] | None = None,
        timeout_ms: int | None = None,
    ) -> dict[str, Any]:
        """Execute an MCP tool and return a normalized result envelope.

        Arguments are validated client-side against the cached JSON schema
        before any network call is made, so invalid shapes fail fast with
        a ``validation_error`` envelope. The result payload contains
        ``content`` items, optional ``structured_data``, and
        ``artifact_refs`` for offloaded binary payloads.

        Args:
            server: MCP server name.
            tool: Tool name on that server.
            arguments: JSON-serializable arguments object matching the
                tool's input schema. Defaults to an empty object.
            timeout_ms: Optional per-call timeout override in
                milliseconds. Defaults to the server's ``call_timeout_ms``.

        Returns:
            A dict with keys ``status``, ``server``, ``tool``,
            ``duration_ms``, and ``tool_result``. On failure, returns an
            error envelope with ``error_type`` set to one of
            ``validation_error``, ``policy_denied``, ``tool_not_found``,
            or ``execution_error``.
        """
        try:
            return await self._registry.execute(
                server=server,
                tool=tool,
                arguments=arguments,
                timeout_ms=timeout_ms,
            )
        except ValidationError as exc:
            return self._error(
                "validation_error",
                str(exc),
                True,
                server=server,
                tool=tool,
                hint="Inspect the tool schema and retry with matching arguments.",
            )
        except PolicyDeniedError as exc:
            return self._error("policy_denied", str(exc), False, server=server, tool=tool)
        except ToolNotFoundError as exc:
            return self._error("tool_not_found", str(exc), True, server=server, tool=tool)
        except Exception as exc:
            return self._error("execution_error", str(exc), True, server=server, tool=tool)

    def health_snapshot(self) -> dict[str, Any]:
        """Return a health snapshot of the underlying registry."""
        return self._registry.health_snapshot()

    async def close(self) -> None:
        await self._registry.close()

    async def __aenter__(self) -> LazyMCPToolset:
        await self.get_tools()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()

    @staticmethod
    def default_instruction() -> str:
        return (
            "Use MCP tools through a three-step workflow:\n"
            "1. Call discover_mcp_tools to find relevant capabilities.\n"
            "2. Call inspect_mcp_tool before the first execution of a tool so you can see its input schema.\n"
            "3. Call execute_mcp_tool with an arguments object that matches the schema exactly.\n"
            "Prefer the smallest relevant tool and do not invent tool names or arguments."
        )

    @staticmethod
    def _error(error_type: str, error: str, recoverable: bool, **extra: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": "error",
            "error_type": error_type,
            "error": error,
            "recoverable": recoverable,
        }
        payload.update(extra)
        return payload
