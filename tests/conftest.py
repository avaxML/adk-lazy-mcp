"""Shared fixtures for adk-lazy-mcp tests."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from adk_lazy_mcp import LazyMCPToolset, RegistryConfig, ServerConfig

ListTools = Callable[[str], Awaitable[list[dict[str, Any]]]]
ExecuteTool = Callable[[str, str, dict[str, Any]], Awaitable[dict[str, Any]]]


READ_FILE_TOOL: dict[str, Any] = {
    "name": "read_file",
    "description": "Read file text from disk",
    "inputSchema": {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "encoding": {"type": "string"},
        },
        "required": ["path"],
    },
}

WRITE_FILE_TOOL: dict[str, Any] = {
    "name": "write_file",
    "description": "Write text to a file",
    "inputSchema": {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "text": {"type": "string"},
        },
        "required": ["path", "text"],
    },
}

LIST_DIR_TOOL: dict[str, Any] = {
    "name": "list_dir",
    "description": "List contents of a directory",
    "inputSchema": {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
}

SEARCH_TOOL: dict[str, Any] = {
    "name": "search",
    "description": "Full-text search across files",
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer"},
        },
        "required": ["query"],
    },
}


@pytest.fixture
def filesystem_tools() -> list[dict[str, Any]]:
    """A canned multi-tool MCP tool list for a fake filesystem server."""
    return [READ_FILE_TOOL, WRITE_FILE_TOOL, LIST_DIR_TOOL, SEARCH_TOOL]


@pytest.fixture
def make_list_tools(filesystem_tools: list[dict[str, Any]]) -> Callable[..., ListTools]:
    """Factory building an async ``list_tools`` callback from a per-server map."""

    def _factory(
        mapping: dict[str, list[dict[str, Any]]] | None = None,
        *,
        default: list[dict[str, Any]] | None = None,
        errors: dict[str, Exception] | None = None,
    ) -> ListTools:
        servers = mapping or {"filesystem": filesystem_tools}
        fallback = default if default is not None else filesystem_tools
        failures = errors or {}

        async def list_tools(server: str) -> list[dict[str, Any]]:
            if server in failures:
                raise failures[server]
            return servers.get(server, fallback)

        return list_tools

    return _factory


@pytest.fixture
def make_execute_tool() -> Callable[..., ExecuteTool]:
    """Factory producing an async ``execute_tool`` callback with programmable results."""

    def _factory(
        responder: Callable[[str, str, dict[str, Any]], dict[str, Any]] | None = None,
    ) -> ExecuteTool:
        def _default(server: str, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
            return {
                "isError": False,
                "content": [{"type": "text", "text": f"{server}:{tool}:{arguments}"}],
            }

        fn = responder or _default

        async def execute_tool(server: str, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
            return fn(server, tool, arguments)

        return execute_tool

    return _factory


@pytest.fixture
def make_toolset(
    make_list_tools: Callable[..., ListTools],
    make_execute_tool: Callable[..., ExecuteTool],
) -> Callable[..., LazyMCPToolset]:
    """Factory that builds a fully wired :class:`LazyMCPToolset` for tests."""

    def _factory(
        *,
        servers: list[ServerConfig] | None = None,
        registry_config: RegistryConfig | None = None,
        list_tools: ListTools | None = None,
        execute_tool: ExecuteTool | None = None,
    ) -> LazyMCPToolset:
        return LazyMCPToolset(
            servers or [ServerConfig(name="filesystem")],
            registry_config or RegistryConfig(warm_mode="eager"),
            list_tools=list_tools or make_list_tools(),
            execute_tool=execute_tool or make_execute_tool(),
        )

    return _factory
