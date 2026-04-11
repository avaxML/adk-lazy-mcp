"""Efficiency benchmark: lazy discover/inspect/execute vs. dumping all tools.

Against real Smithery-hosted MCPs this test measures how many bytes/tokens a
traditional "dump every MCP tool schema into the model prompt" setup would
cost, vs. the three meta-tools ``LazyMCPToolset`` exposes. It prints a
markdown summary table to stdout (visible in CI logs via ``pytest -s``) and
asserts a minimum reduction ratio to guard against regressions.
"""

from __future__ import annotations

import inspect
import json
import time
from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from adk_lazy_mcp import LazyMCPToolset, RegistryConfig, ServerConfig

pytestmark = [pytest.mark.integration, pytest.mark.benchmark]

ADKCallbacks = tuple[
    Callable[[str], Awaitable[list[dict[str, Any]]]],
    Callable[[str, str, dict[str, Any]], Awaitable[dict[str, Any]]],
]

# Rough 4-chars-per-token approximation so we don't take a tiktoken dependency
# just for a relative comparison. It is directionally accurate for short
# JSON-ish strings, which is what we measure here.
_CHARS_PER_TOKEN = 4


def _approx_tokens(payload: str) -> int:
    return (len(payload) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN


def _lazy_tool_surface(toolset: LazyMCPToolset) -> list[dict[str, Any]]:
    """Shape the three meta-tools the way an ADK prompt would see them."""
    out: list[dict[str, Any]] = []
    for fn in (
        toolset.discover_mcp_tools,
        toolset.inspect_mcp_tool,
        toolset.execute_mcp_tool,
    ):
        sig = inspect.signature(fn)
        out.append(
            {
                "name": fn.__name__,
                "description": (fn.__doc__ or "").strip(),
                "parameters": {
                    name: str(param.annotation) for name, param in sig.parameters.items()
                },
            }
        )
    return out


async def _naive_tool_surface(
    servers: list[str], list_tools: Callable[[str], Awaitable[list[dict[str, Any]]]]
) -> list[dict[str, Any]]:
    """What an ADK agent would see if you dumped every MCP tool upfront."""
    surface: list[dict[str, Any]] = []
    for server in servers:
        tools = await list_tools(server)
        for tool in tools:
            surface.append(
                {
                    "server": server,
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "inputSchema": tool.get("inputSchema", {}),
                }
            )
    return surface


async def test_lazy_surface_is_smaller_than_naive(
    adk_callbacks: ADKCallbacks, capsys: pytest.CaptureFixture[str]
) -> None:
    list_tools, execute_tool = adk_callbacks
    server_configs = [
        ServerConfig(name="sequential_thinking", transport="streamable_http"),
        ServerConfig(name="fetch", transport="streamable_http"),
    ]
    toolset = LazyMCPToolset(
        server_configs,
        RegistryConfig(warm_mode="eager"),
        list_tools=list_tools,
        execute_tool=execute_tool,
    )
    try:
        await toolset.get_tools()

        # --- size comparison -------------------------------------------------
        lazy_surface = _lazy_tool_surface(toolset)
        naive_surface = await _naive_tool_surface([c.name for c in server_configs], list_tools)

        lazy_json = json.dumps(lazy_surface)
        naive_json = json.dumps(naive_surface)

        lazy_bytes = len(lazy_json.encode("utf-8"))
        naive_bytes = len(naive_json.encode("utf-8"))
        lazy_tokens = _approx_tokens(lazy_json)
        naive_tokens = _approx_tokens(naive_json)

        # Plus the discover payload the model actually sees when it searches.
        discover = await toolset.discover_mcp_tools(query="fetch")
        discover_json = json.dumps(discover)
        discover_bytes = len(discover_json.encode("utf-8"))
        discover_tokens = _approx_tokens(discover_json)

        ratio = naive_bytes / max(lazy_bytes, 1)

        # --- latency comparison ---------------------------------------------
        # Lazy path: discover(query) -> inspect -> execute on a specific tool.
        target = next((t for t in discover["tools"] if t["server"] == "fetch"), None)
        assert target is not None, "expected the fetch server to expose a tool"

        lazy_start = time.perf_counter()
        await toolset.discover_mcp_tools(query=target["tool"])
        await toolset.inspect_mcp_tool(target["server"], target["tool"])
        inspect_result = await toolset.inspect_mcp_tool(target["server"], target["tool"])
        await toolset.execute_mcp_tool(
            target["server"],
            target["tool"],
            _min_args(inspect_result["input_schema"]),
        )
        lazy_duration_ms = (time.perf_counter() - lazy_start) * 1_000

        # Naive path: list every tool from every server, then execute.
        naive_start = time.perf_counter()
        for cfg in server_configs:
            await list_tools(cfg.name)
        await execute_tool(
            target["server"],
            target["tool"],
            _min_args(inspect_result["input_schema"]),
        )
        naive_duration_ms = (time.perf_counter() - naive_start) * 1_000

        # --- report ---------------------------------------------------------
        with capsys.disabled():
            print()
            print("### adk-lazy-mcp efficiency report")
            print()
            print("| metric | naive (dump all tools) | lazy (three meta-tools) | reduction |")
            print("|---|---|---|---|")
            print(f"| tool surface bytes | {naive_bytes} | {lazy_bytes} | {ratio:.1f}x smaller |")
            print(
                f"| tool surface ~tokens | {naive_tokens} | {lazy_tokens} | "
                f"{naive_tokens / max(lazy_tokens, 1):.1f}x smaller |"
            )
            print(
                f"| discover payload bytes | n/a | {discover_bytes} (~{discover_tokens} tokens) | |"
            )
            print(
                f"| discover→inspect→execute latency | {naive_duration_ms:.0f} ms "
                f"(list + call) | {lazy_duration_ms:.0f} ms | |"
            )
            print()
            print(f"total tools exposed by the MCPs under test: {len(naive_surface)}")
            print()

        # Regression guards. With two non-trivial MCPs, we expect the naive
        # surface to be at least 3x bigger than the three-meta-tool surface.
        assert ratio >= 3.0, (
            f"expected >=3x reduction in tool surface, got {ratio:.2f}x "
            f"(naive={naive_bytes} bytes, lazy={lazy_bytes} bytes)"
        )
        # And a lazy "tool surface" should fit comfortably in a prompt.
        assert lazy_tokens < 2_000, f"lazy tool surface unexpectedly large: ~{lazy_tokens} tokens"
    finally:
        await toolset.close()


def _min_args(schema: dict[str, Any]) -> dict[str, Any]:
    args: dict[str, Any] = {}
    required = schema.get("required", []) or []
    properties = schema.get("properties", {}) or {}
    for key in required:
        prop = properties.get(key, {})
        expected = prop.get("type", "string")
        if expected == "string":
            # Use a stable, harmless URL for the fetch server.
            args[key] = "https://example.com"
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
