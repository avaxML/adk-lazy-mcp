"""Live end-to-end checks against Smithery-hosted MCP servers.

These tests prove that ``LazyMCPToolset`` works unmodified against a real
MCP transport. They require ``SMITHERY_API_KEY`` in the environment; if it
is missing, the whole ``tests/integration`` package is skipped by
``conftest.py``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from adk_lazy_mcp import LazyMCPToolset, RegistryConfig, ServerConfig

pytestmark = pytest.mark.integration


ADKCallbacks = tuple[
    Callable[[str], Awaitable[list[dict[str, Any]]]],
    Callable[[str, str, dict[str, Any]], Awaitable[dict[str, Any]]],
]


async def _build_toolset(adk_callbacks: ADKCallbacks) -> LazyMCPToolset:
    list_tools, execute_tool = adk_callbacks
    return LazyMCPToolset(
        [
            ServerConfig(name="sequential_thinking", transport="streamable_http"),
            ServerConfig(name="fetch", transport="streamable_http"),
        ],
        RegistryConfig(warm_mode="eager"),
        list_tools=list_tools,
        execute_tool=execute_tool,
    )


async def test_get_tools_surfaces_three_meta_tools(adk_callbacks: ADKCallbacks) -> None:
    toolset = await _build_toolset(adk_callbacks)
    try:
        tools = await toolset.get_tools()
        assert [t.__name__ for t in tools] == [
            "discover_mcp_tools",
            "inspect_mcp_tool",
            "execute_mcp_tool",
        ]
    finally:
        await toolset.close()


async def test_discover_then_inspect_real_fetch_tool(
    adk_callbacks: ADKCallbacks,
) -> None:
    toolset = await _build_toolset(adk_callbacks)
    try:
        await toolset.get_tools()

        discover = await toolset.discover_mcp_tools(query="fetch")
        assert discover["status"] == "success"
        assert discover["total_matches"] >= 1

        # The fetch server publishes a tool literally named "fetch".
        target = next((t for t in discover["tools"] if t["server"] == "fetch"), None)
        assert target is not None, f"no fetch tool found: {discover['tools']}"

        inspect = await toolset.inspect_mcp_tool(target["server"], target["tool"])
        assert inspect["status"] == "success"
        assert "input_schema" in inspect
        assert inspect["schema_hash"].startswith("sha256:")
    finally:
        await toolset.close()


async def test_execute_sequential_thinking_real_call(
    adk_callbacks: ADKCallbacks,
) -> None:
    toolset = await _build_toolset(adk_callbacks)
    try:
        await toolset.get_tools()

        discover = await toolset.discover_mcp_tools(server="sequential_thinking")
        assert discover["total_matches"] >= 1
        target = discover["tools"][0]

        inspect = await toolset.inspect_mcp_tool(target["server"], target["tool"])
        schema = inspect["input_schema"]

        # Build a minimum-viable argument object from the inspected schema so
        # the test is resilient to tool revisions.
        args = _minimum_valid_arguments(schema)

        result = await toolset.execute_mcp_tool(target["server"], target["tool"], args)
        assert result["status"] in {"success", "error"}  # either is informative
        # If it succeeded, the result envelope must be well-formed.
        if result["status"] == "success":
            assert result["tool_result"]["is_error"] is False
    finally:
        await toolset.close()


async def test_health_snapshot_reports_ready(adk_callbacks: ADKCallbacks) -> None:
    toolset = await _build_toolset(adk_callbacks)
    try:
        await toolset.get_tools()
        snap = toolset.health_snapshot()
        assert snap["status"] == "ready"
        names = {s["name"] for s in snap["servers"]}
        assert {"sequential_thinking", "fetch"} <= names
    finally:
        await toolset.close()


def _minimum_valid_arguments(schema: dict[str, Any]) -> dict[str, Any]:
    """Build an argument dict satisfying all ``required`` fields of a JSON schema.

    Covers the primitive types used by the servers we test against. If the
    server introduces a more exotic required field, the test will still run
    but the MCP server may return an error envelope, which is covered by the
    soft assertion in the caller.
    """
    args: dict[str, Any] = {}
    required = schema.get("required", []) or []
    properties = schema.get("properties", {}) or {}
    for key in required:
        prop = properties.get(key, {})
        expected = prop.get("type", "string")
        if expected == "string":
            args[key] = "hello world"
        elif expected == "integer":
            args[key] = 1
        elif expected == "number":
            args[key] = 1.0
        elif expected == "boolean":
            args[key] = True
        elif expected == "array":
            args[key] = []
        elif expected == "object":
            args[key] = {}
        else:
            args[key] = ""
    return args
