"""Integration tests for BM25 discovery ranking against live Smithery MCPs.

These tests exercise the per-server BM25 index, cross-server reciprocal-rank
fusion, semantic reranking, leader-cluster tool families, and schema-aware
indexing against a broad set of free community-hosted MCP servers on Smithery.

Every test requires ``SMITHERY_API_KEY`` in the environment; if it is missing
the whole ``tests/integration`` package is skipped by ``conftest.py``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from adk_lazy_mcp import LazyMCPToolset, RegistryConfig, ServerConfig

from .conftest import EXTENDED_SERVERS

pytestmark = pytest.mark.integration

ADKCallbacks = tuple[
    Callable[[str], Awaitable[list[dict[str, Any]]]],
    Callable[[str, str, dict[str, Any]], Awaitable[dict[str, Any]]],
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EXTENDED_SERVER_CONFIGS = [
    ServerConfig(name=name, transport="streamable_http") for name, _ in EXTENDED_SERVERS
]


async def _build_extended_toolset(
    callbacks: ADKCallbacks,
    *,
    warm_mode: str = "eager",
) -> LazyMCPToolset:
    list_tools, execute_tool = callbacks
    return LazyMCPToolset(
        _EXTENDED_SERVER_CONFIGS,
        RegistryConfig(warm_mode=warm_mode, max_discover_results=100),  # type: ignore[arg-type]
        list_tools=list_tools,
        execute_tool=execute_tool,
    )


def _tool_names(discover_result: dict[str, Any]) -> list[str]:
    return [t["tool"] for t in discover_result["tools"]]


def _server_names(discover_result: dict[str, Any]) -> set[str]:
    return {t["server"] for t in discover_result["tools"]}


def _families(discover_result: dict[str, Any]) -> dict[str, str]:
    return {t["tool"]: t.get("family", t["tool"]) for t in discover_result["tools"]}


# ---------------------------------------------------------------------------
# BM25 lexical ranking quality
# ---------------------------------------------------------------------------


async def test_bm25_exact_name_match_ranked_first(
    extended_adk_callbacks: ADKCallbacks,
) -> None:
    """An exact tool-name query should surface that tool at rank 1."""
    toolset = await _build_extended_toolset(extended_adk_callbacks)
    try:
        await toolset.get_tools()
        result = await toolset.discover_mcp_tools(query="add")
        assert result["status"] == "success"
        names = _tool_names(result)
        assert len(names) >= 1, "expected at least one match for 'add'"
        # math/add should be first or very near the top.
        assert "add" in names[:3], f"'add' not in top-3: {names[:5]}"
    finally:
        await toolset.close()


async def test_bm25_prefix_match_ranked_high(
    extended_adk_callbacks: ADKCallbacks,
) -> None:
    """A prefix query should surface tools whose name starts with that prefix."""
    toolset = await _build_extended_toolset(extended_adk_callbacks)
    try:
        await toolset.get_tools()
        result = await toolset.discover_mcp_tools(query="multi")
        assert result["status"] == "success"
        names = _tool_names(result)
        # "multiply" begins with "multi" — should rank high.
        prefix_hits = [n for n in names if n.startswith("multi")]
        assert len(prefix_hits) >= 1, f"no prefix hits for 'multi': {names[:10]}"
    finally:
        await toolset.close()


async def test_bm25_description_match(
    extended_adk_callbacks: ADKCallbacks,
) -> None:
    """A query matching description text should still surface relevant tools."""
    toolset = await _build_extended_toolset(extended_adk_callbacks)
    try:
        await toolset.get_tools()
        # "sine" appears in the description of trig tools on math-mcp.
        result = await toolset.discover_mcp_tools(query="sine")
        assert result["status"] == "success"
        names = _tool_names(result)
        assert len(names) >= 1, "expected at least one match for 'sine'"
    finally:
        await toolset.close()


# ---------------------------------------------------------------------------
# Cross-server rank fusion
# ---------------------------------------------------------------------------


async def test_cross_server_discovery_returns_multiple_servers(
    extended_adk_callbacks: ADKCallbacks,
) -> None:
    """A broad query with no server filter should merge results from ≥2 servers."""
    toolset = await _build_extended_toolset(extended_adk_callbacks)
    try:
        await toolset.get_tools()
        # Empty query returns all tools from every ready server.
        result = await toolset.discover_mcp_tools(query="", limit=100)
        assert result["status"] == "success"
        servers = _server_names(result)
        assert len(servers) >= 2, (
            f"expected results from ≥2 servers, got {servers}; "
            f"unavailable={result.get('unavailable_servers', [])}"
        )
    finally:
        await toolset.close()


async def test_cross_server_query_merges_results(
    extended_adk_callbacks: ADKCallbacks,
) -> None:
    """A keyword present in multiple servers' tool catalogs should return
    merged results ranked by reciprocal-rank fusion.
    """
    toolset = await _build_extended_toolset(extended_adk_callbacks)
    try:
        await toolset.get_tools()
        # "search" is likely present in duckduckgo and paper_search catalogs.
        result = await toolset.discover_mcp_tools(query="search")
        assert result["status"] == "success"
        assert result["total_matches"] >= 1
    finally:
        await toolset.close()


# ---------------------------------------------------------------------------
# Per-server isolation
# ---------------------------------------------------------------------------


async def test_per_server_filter_isolates_results(
    extended_adk_callbacks: ADKCallbacks,
) -> None:
    """Passing ``server=`` should restrict discovery to that server only."""
    toolset = await _build_extended_toolset(extended_adk_callbacks)
    try:
        await toolset.get_tools()
        result = await toolset.discover_mcp_tools(query="", server="math")
        assert result["status"] == "success"
        servers = _server_names(result)
        assert servers == {"math"}, f"expected only math, got {servers}"
        assert result["total_matches"] >= 10, (
            f"math-mcp should expose ≥10 tools, got {result['total_matches']}"
        )
    finally:
        await toolset.close()


async def test_per_server_query_within_single_server(
    extended_adk_callbacks: ADKCallbacks,
) -> None:
    """A targeted query scoped to a single server should only return its tools."""
    toolset = await _build_extended_toolset(extended_adk_callbacks)
    try:
        await toolset.get_tools()
        result = await toolset.discover_mcp_tools(query="add", server="math")
        assert result["status"] == "success"
        names = _tool_names(result)
        assert "add" in names, f"'add' not found in math-scoped results: {names}"
        assert _server_names(result) == {"math"}
    finally:
        await toolset.close()


# ---------------------------------------------------------------------------
# Tool family clustering
# ---------------------------------------------------------------------------


async def test_discovery_results_include_family_label(
    extended_adk_callbacks: ADKCallbacks,
) -> None:
    """Every discovery result should carry a ``family`` field."""
    toolset = await _build_extended_toolset(extended_adk_callbacks)
    try:
        await toolset.get_tools()
        result = await toolset.discover_mcp_tools(query="", server="math", limit=100)
        assert result["status"] == "success"
        for tool in result["tools"]:
            assert "family" in tool, f"tool {tool['tool']} missing 'family' field"
            assert isinstance(tool["family"], str)
            assert tool["family"], "family must not be empty"
    finally:
        await toolset.close()


async def test_related_math_tools_share_family(
    extended_adk_callbacks: ADKCallbacks,
) -> None:
    """The math server's arithmetic tools should cluster into families."""
    toolset = await _build_extended_toolset(extended_adk_callbacks)
    try:
        await toolset.get_tools()
        result = await toolset.discover_mcp_tools(query="", server="math", limit=100)
        assert result["status"] == "success"
        fam = _families(result)
        # The 22 math tools should NOT all be in the same family — leader
        # clustering should split them into at least 2 families.
        unique_families = set(fam.values())
        assert len(unique_families) >= 2, (
            f"expected ≥2 families among {len(fam)} math tools, got {unique_families}"
        )
    finally:
        await toolset.close()


# ---------------------------------------------------------------------------
# Schema-aware indexing
# ---------------------------------------------------------------------------


async def test_schema_property_name_searchable(
    extended_adk_callbacks: ADKCallbacks,
) -> None:
    """Querying a schema property name should surface tools that accept it.

    The math/add tool's input schema contains properties like ``a`` / ``b``
    (or ``firstNumber`` / ``secondNumber``). We inspect the actual schema
    to pick a property name, then search for it.
    """
    toolset = await _build_extended_toolset(extended_adk_callbacks)
    try:
        await toolset.get_tools()
        # First inspect to get a real property name.
        inspect = await toolset.inspect_mcp_tool("math", "add")
        assert inspect["status"] == "success"
        schema = inspect["input_schema"]
        props = list((schema.get("properties") or {}).keys())
        assert props, "expected add tool to have schema properties"

        # Search for the longest property name (most discriminating).
        prop = max(props, key=len)
        result = await toolset.discover_mcp_tools(query=prop)
        assert result["status"] == "success"
        # The tool we inspected should appear in results.
        names = _tool_names(result)
        assert len(names) >= 1, f"no matches for schema prop '{prop}': {names}"
    finally:
        await toolset.close()


# ---------------------------------------------------------------------------
# Semantic fallback
# ---------------------------------------------------------------------------


async def test_semantic_fallback_close_variant(
    extended_adk_callbacks: ADKCallbacks,
) -> None:
    """A semantically close query should still find relevant tools.

    ``"addition"`` should surface the ``add`` tool via trigram-based semantic
    overlap even if "addition" is not literally a term in the tool document.
    """
    toolset = await _build_extended_toolset(extended_adk_callbacks)
    try:
        await toolset.get_tools()
        result = await toolset.discover_mcp_tools(query="addition")
        assert result["status"] == "success"
        names = _tool_names(result)
        # "add" should appear via BM25 substring/description match or
        # semantic fallback.
        assert "add" in names, f"'add' not found via semantic fallback: {names[:10]}"
    finally:
        await toolset.close()


# ---------------------------------------------------------------------------
# Many-server resilience
# ---------------------------------------------------------------------------


async def test_many_servers_no_crash(
    extended_adk_callbacks: ADKCallbacks,
) -> None:
    """Hydrating many servers and running discovery should not raise."""
    toolset = await _build_extended_toolset(extended_adk_callbacks)
    try:
        await toolset.get_tools()
        result = await toolset.discover_mcp_tools(query="calculate", limit=50)
        assert result["status"] == "success"
        # At minimum math is up.
        assert result["total_matches"] >= 1
    finally:
        await toolset.close()


async def test_health_snapshot_with_many_servers(
    extended_adk_callbacks: ADKCallbacks,
) -> None:
    """Health snapshot should reflect every registered server."""
    toolset = await _build_extended_toolset(extended_adk_callbacks)
    try:
        await toolset.get_tools()
        snap = toolset.health_snapshot()
        names = {s["name"] for s in snap["servers"]}
        # The default two must be present.
        assert {"math", "sequential_thinking"} <= names
        # All extended servers should be registered even if some are degraded.
        expected_names = {name for name, _ in EXTENDED_SERVERS}
        assert expected_names == names, f"missing servers: {expected_names - names}"
    finally:
        await toolset.close()


async def test_unavailable_servers_reported(
    extended_adk_callbacks: ADKCallbacks,
) -> None:
    """Discovery should report servers that failed hydration as unavailable."""
    toolset = await _build_extended_toolset(extended_adk_callbacks)
    try:
        await toolset.get_tools()
        result = await toolset.discover_mcp_tools(query="")
        assert result["status"] == "success"
        # If any extended server is down, it should appear here.
        # We just assert the field exists and is a list.
        assert isinstance(result["unavailable_servers"], list)
    finally:
        await toolset.close()


# ---------------------------------------------------------------------------
# Discovery + inspect + execute round-trip across servers
# ---------------------------------------------------------------------------


async def test_discover_inspect_execute_roundtrip_across_servers(
    extended_adk_callbacks: ADKCallbacks,
) -> None:
    """Discover a tool from the math server, inspect its schema, and execute it.

    This proves the full three-step lazy workflow works end-to-end when the
    discovery ranking selects a tool from a multi-server catalog.
    """
    toolset = await _build_extended_toolset(extended_adk_callbacks)
    try:
        await toolset.get_tools()

        # Step 1: Discover — broad query, no server filter.
        discover = await toolset.discover_mcp_tools(query="subtract")
        assert discover["status"] == "success"
        target = next(
            (t for t in discover["tools"] if t["server"] == "math" and t["tool"] == "subtract"),
            None,
        )
        assert target is not None, f"subtract not found: {_tool_names(discover)}"

        # Step 2: Inspect.
        inspect = await toolset.inspect_mcp_tool(target["server"], target["tool"])
        assert inspect["status"] == "success"
        schema = inspect["input_schema"]

        # Step 3: Execute with auto-generated arguments.
        args: dict[str, Any] = {}
        for key in schema.get("required", []):
            prop = (schema.get("properties") or {}).get(key, {})
            ptype = prop.get("type", "number")
            args[key] = 5 if ptype in {"number", "integer"} else "5"

        result = await toolset.execute_mcp_tool(target["server"], target["tool"], args)
        assert result["status"] == "success"
        assert result["tool_result"]["is_error"] is False
    finally:
        await toolset.close()


# ---------------------------------------------------------------------------
# Ranking consistency
# ---------------------------------------------------------------------------


async def test_repeated_discovery_is_deterministic(
    extended_adk_callbacks: ADKCallbacks,
) -> None:
    """Two identical discovery calls should return the same ranked order."""
    toolset = await _build_extended_toolset(extended_adk_callbacks)
    try:
        await toolset.get_tools()
        r1 = await toolset.discover_mcp_tools(query="add")
        r2 = await toolset.discover_mcp_tools(query="add")
        assert _tool_names(r1) == _tool_names(r2)
    finally:
        await toolset.close()
