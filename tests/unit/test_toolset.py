"""Unit tests for :mod:`adk_lazy_mcp.toolset`."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from adk_lazy_mcp import LazyMCPToolset, RegistryConfig, ServerConfig

MakeToolset = Callable[..., LazyMCPToolset]


class TestGetTools:
    async def test_returns_exactly_the_three_meta_tools(self, make_toolset: MakeToolset) -> None:
        toolset = make_toolset()
        tools = await toolset.get_tools()
        assert len(tools) == 3
        names = [t.__name__ for t in tools]
        assert names == [
            "discover_mcp_tools",
            "inspect_mcp_tool",
            "execute_mcp_tool",
        ]

    async def test_get_tools_accepts_readonly_context(self, make_toolset: MakeToolset) -> None:
        toolset = make_toolset()
        tools = await toolset.get_tools(readonly_context={"agent": "adk"})
        assert len(tools) == 3

    async def test_lazy_start_only_runs_once(
        self,
        make_list_tools: Callable[..., Callable[[str], Awaitable[list[dict[str, Any]]]]],
        make_execute_tool: Callable[..., Callable[..., Awaitable[dict[str, Any]]]],
    ) -> None:
        calls: list[str] = []

        async def tracking_list(server: str) -> list[dict[str, Any]]:
            calls.append(server)
            return []

        toolset = LazyMCPToolset(
            [ServerConfig(name="filesystem")],
            RegistryConfig(warm_mode="eager"),
            list_tools=tracking_list,
            execute_tool=make_execute_tool(),
        )
        await toolset.get_tools()
        await toolset.get_tools()
        await toolset.get_tools()
        # One hydration only: get_tools starts the registry once, and `eager`
        # warmup triggers a single call per server.
        assert calls == ["filesystem"]


class TestThreeStepWorkflow:
    async def test_discover_inspect_execute_happy_path(self, make_toolset: MakeToolset) -> None:
        toolset = make_toolset()
        await toolset.get_tools()

        discover = await toolset.discover_mcp_tools(query="read")
        assert discover["status"] == "success"
        assert discover["total_matches"] >= 1
        assert any(t["tool"] == "read_file" for t in discover["tools"])

        inspect = await toolset.inspect_mcp_tool("filesystem", "read_file")
        assert inspect["status"] == "success"
        assert inspect["required_fields"] == ["path"]
        assert inspect["schema_hash"].startswith("sha256:")

        run = await toolset.execute_mcp_tool("filesystem", "read_file", {"path": "README.md"})
        assert run["status"] == "success"
        assert run["tool_result"]["is_error"] is False


class TestErrorEnvelopes:
    async def test_validation_error_is_structured(self, make_toolset: MakeToolset) -> None:
        toolset = make_toolset()
        await toolset.get_tools()
        run = await toolset.execute_mcp_tool("filesystem", "read_file", {})
        assert run["status"] == "error"
        assert run["error_type"] == "validation_error"
        assert run["recoverable"] is True
        assert "Inspect the tool schema" in run["hint"]

    async def test_tool_not_found_on_execute(self, make_toolset: MakeToolset) -> None:
        toolset = make_toolset()
        await toolset.get_tools()
        run = await toolset.execute_mcp_tool("filesystem", "does_not_exist", {})
        assert run["status"] == "error"
        assert run["error_type"] == "tool_not_found"

    async def test_policy_denied_on_execute(self, make_toolset: MakeToolset) -> None:
        toolset = make_toolset(
            servers=[ServerConfig(name="filesystem", deny_tools=("write_file",))]
        )
        await toolset.get_tools()
        run = await toolset.execute_mcp_tool(
            "filesystem", "write_file", {"path": "/a", "text": "b"}
        )
        assert run["status"] == "error"
        assert run["error_type"] == "policy_denied"
        assert run["recoverable"] is False

    async def test_execution_error_on_unexpected_exception(
        self,
        make_list_tools: Callable[..., Callable[[str], Awaitable[list[dict[str, Any]]]]],
        make_execute_tool: Callable[..., Callable[..., Awaitable[dict[str, Any]]]],
    ) -> None:
        async def crash(server: str, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
            raise ValueError("kaboom")

        toolset = LazyMCPToolset(
            [ServerConfig(name="filesystem")],
            RegistryConfig(warm_mode="eager"),
            list_tools=make_list_tools(),
            execute_tool=crash,
        )
        await toolset.get_tools()
        run = await toolset.execute_mcp_tool("filesystem", "read_file", {"path": "/a"})
        assert run["status"] == "error"
        assert run["error_type"] == "execution_error"
        assert "kaboom" in run["error"]

    async def test_discover_tool_not_found_for_unknown_server(
        self, make_toolset: MakeToolset
    ) -> None:
        toolset = make_toolset()
        await toolset.get_tools()
        result = await toolset.discover_mcp_tools(server="ghost")
        assert result["status"] == "error"
        assert result["error_type"] == "tool_not_found"

    async def test_inspect_tool_not_found(self, make_toolset: MakeToolset) -> None:
        toolset = make_toolset()
        await toolset.get_tools()
        result = await toolset.inspect_mcp_tool("filesystem", "nope")
        assert result["status"] == "error"
        assert result["error_type"] == "tool_not_found"


class TestContextManager:
    async def test_async_context_manager_starts_and_closes(self, make_toolset: MakeToolset) -> None:
        toolset = make_toolset()
        async with toolset as t:
            result = await t.discover_mcp_tools()
            assert result["status"] == "success"
        # After close, a follow-up discover surfaces a structured error envelope.
        err = await toolset.discover_mcp_tools()
        assert err["status"] == "error"
        assert err["error_type"] == "discover_error"


class TestMiscellaneous:
    def test_default_instruction_mentions_three_tools(self) -> None:
        instr = LazyMCPToolset.default_instruction()
        assert "discover_mcp_tools" in instr
        assert "inspect_mcp_tool" in instr
        assert "execute_mcp_tool" in instr
        assert instr.count("\n") >= 3  # multi-line

    async def test_health_snapshot_shape(self, make_toolset: MakeToolset) -> None:
        toolset = make_toolset()
        await toolset.get_tools()
        snap = toolset.health_snapshot()
        assert snap["status"] in {"ready", "degraded"}
        assert isinstance(snap["servers"], list)
        assert {"name", "state", "tool_count", "breaker"} <= set(snap["servers"][0])


@pytest.mark.parametrize("warm_mode", ["eager", "background", "on_demand"])
async def test_workflow_against_all_warm_modes(make_toolset: MakeToolset, warm_mode: str) -> None:
    toolset = make_toolset(registry_config=RegistryConfig(warm_mode=warm_mode))  # type: ignore[arg-type]
    await toolset.get_tools()
    result = await toolset.execute_mcp_tool("filesystem", "read_file", {"path": "/tmp/a"})
    assert result["status"] == "success"
