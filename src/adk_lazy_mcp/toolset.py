from __future__ import annotations

from typing import Any, Callable, Awaitable

from .config import RegistryConfig, ServerConfig
from .errors import LazyMCPError, PolicyDeniedError, ToolNotFoundError, ValidationError
from .registry import Registry


class LazyMCPToolset:
    """ADK-native wrapper exposing three meta-tools for MCP interactions."""

    def __init__(
        self,
        server_configs: list[ServerConfig],
        registry_config: RegistryConfig | None = None,
        *,
        list_tools: Callable[[str], Awaitable[list[dict[str, Any]]]],
        execute_tool: Callable[[str, str, dict[str, Any]], Awaitable[dict[str, Any]]],
    ) -> None:
        self._registry = Registry(
            server_configs,
            registry_config or RegistryConfig(),
            list_tools=list_tools,
            execute_tool=execute_tool,
        )
        self._started = False

    async def get_tools(self, tool_filter: set[str] | None = None) -> list[Callable[..., Any]]:
        if not self._started:
            await self._registry.start()
            self._started = True
        all_tools = [self.discover_mcp_tools, self.inspect_mcp_tool, self.execute_mcp_tool]
        if not tool_filter:
            return all_tools
        return [t for t in all_tools if t.__name__ in tool_filter]

    async def discover_mcp_tools(self, query: str = "", server: str = "", limit: int = 20) -> dict[str, Any]:
        try:
            return await self._registry.discover(query=query, server=server, limit=limit)
        except LazyMCPError as exc:
            return self._error("discover_error", str(exc), True)

    async def inspect_mcp_tool(self, server: str, tool: str, refresh: bool = False) -> dict[str, Any]:
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
        try:
            return await self._registry.execute(server=server, tool=tool, arguments=arguments, timeout_ms=timeout_ms)
        except ValidationError as exc:
            return self._error("validation_error", str(exc), True, server=server, tool=tool, hint="Inspect the tool schema and retry with matching arguments.")
        except PolicyDeniedError as exc:
            return self._error("policy_denied", str(exc), False, server=server, tool=tool)
        except ToolNotFoundError as exc:
            return self._error("tool_not_found", str(exc), True, server=server, tool=tool)
        except Exception as exc:  # noqa: BLE001
            return self._error("execution_error", str(exc), True, server=server, tool=tool)

    async def close(self) -> None:
        await self._registry.close()

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
        payload = {
            "status": "error",
            "error_type": error_type,
            "error": error,
            "recoverable": recoverable,
        }
        payload.update(extra)
        return payload
