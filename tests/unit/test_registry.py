"""Unit tests for :mod:`adk_lazy_mcp.registry`."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from adk_lazy_mcp.config import RegistryConfig, ServerConfig
from adk_lazy_mcp.errors import (
    PolicyDeniedError,
    RegistryClosedError,
    ToolNotFoundError,
    ValidationError,
)
from adk_lazy_mcp.registry import Registry, _validate_args
from adk_lazy_mcp.telemetry import Telemetry

# ---------------------------------------------------------------------------
# _validate_args: every branch of the client-side validator.
# ---------------------------------------------------------------------------

OBJECT_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "count": {"type": "integer"},
        "ratio": {"type": "number"},
        "flag": {"type": "boolean"},
        "meta": {"type": "object"},
        "tags": {"type": "array"},
        "nothing": {"type": "null"},
    },
    "required": ["path"],
}


class TestValidateArgs:
    def test_all_valid(self) -> None:
        assert (
            _validate_args(
                {
                    "path": "/tmp/a",
                    "count": 3,
                    "ratio": 1.5,
                    "flag": True,
                    "meta": {"k": "v"},
                    "tags": [1, 2],
                    "nothing": None,
                },
                OBJECT_SCHEMA,
            )
            is None
        )

    def test_missing_required(self) -> None:
        err = _validate_args({}, OBJECT_SCHEMA)
        assert err == "path is a required property"

    def test_not_a_dict(self) -> None:
        assert _validate_args("oops", OBJECT_SCHEMA) == "arguments must be an object"  # type: ignore[arg-type]

    def test_wrong_string_type(self) -> None:
        assert _validate_args({"path": 3}, OBJECT_SCHEMA) == "path should be of type string"

    def test_bool_is_not_accepted_as_int(self) -> None:
        err = _validate_args({"path": "/a", "count": True}, OBJECT_SCHEMA)
        assert err == "count should be of type integer"

    def test_bool_is_not_accepted_as_number(self) -> None:
        err = _validate_args({"path": "/a", "ratio": False}, OBJECT_SCHEMA)
        assert err == "ratio should be of type number"

    def test_int_accepted_as_number(self) -> None:
        assert _validate_args({"path": "/a", "ratio": 3}, OBJECT_SCHEMA) is None

    def test_unknown_property_is_ignored(self) -> None:
        assert _validate_args({"path": "/a", "unknown": 99}, OBJECT_SCHEMA) is None

    def test_property_without_type_is_ignored(self) -> None:
        schema = {"type": "object", "properties": {"x": {}}}
        assert _validate_args({"x": "anything"}, schema) is None

    def test_unknown_json_type_is_ignored(self) -> None:
        schema = {"type": "object", "properties": {"x": {"type": "weird"}}}
        assert _validate_args({"x": "anything"}, schema) is None


# ---------------------------------------------------------------------------
# Registry: warm modes, discover, inspect, execute, close, telemetry.
# ---------------------------------------------------------------------------

READ_TOOL: dict[str, Any] = {
    "name": "read_file",
    "description": "Read file contents",
    "inputSchema": {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
}

WRITE_TOOL: dict[str, Any] = {
    "name": "write_file",
    "description": "Write file contents",
    "inputSchema": {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "text": {"type": "string"},
        },
        "required": ["path", "text"],
    },
}


def _make_registry(
    *,
    tools: dict[str, list[dict[str, Any]]] | None = None,
    errors: dict[str, Exception] | None = None,
    warm_mode: str = "eager",
    enable_client_validation: bool = True,
    max_discover: int = 20,
    hard_cap: int = 100,
    servers: list[ServerConfig] | None = None,
    telemetry: Telemetry | None = None,
) -> Registry:
    mapping = tools or {"filesystem": [READ_TOOL, WRITE_TOOL]}
    failures = errors or {}

    async def list_tools(server: str) -> list[dict[str, Any]]:
        if server in failures:
            raise failures[server]
        return mapping.get(server, [])

    async def execute_tool(server: str, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return {
            "isError": False,
            "content": [{"type": "text", "text": f"{server}:{tool}:{arguments}"}],
        }

    return Registry(
        servers or [ServerConfig(name="filesystem")],
        RegistryConfig(
            warm_mode=warm_mode,  # type: ignore[arg-type]
            enable_client_validation=enable_client_validation,
            max_discover_results=max_discover,
            hard_discover_cap=hard_cap,
        ),
        list_tools=list_tools,
        execute_tool=execute_tool,
        telemetry=telemetry,
    )


class TestWarmModes:
    async def test_eager_hydrates_before_start_returns(self) -> None:
        reg = _make_registry(warm_mode="eager")
        await reg.start()
        result = await reg.discover()
        assert result["unavailable_servers"] == []
        assert result["total_matches"] == 2

    async def test_on_demand_does_not_hydrate_until_first_use(self) -> None:
        calls: list[str] = []

        async def list_tools(server: str) -> list[dict[str, Any]]:
            calls.append(server)
            return [READ_TOOL]

        reg = Registry(
            [ServerConfig(name="filesystem")],
            RegistryConfig(warm_mode="on_demand"),
            list_tools=list_tools,
            execute_tool=_null_executor,
        )
        await reg.start()
        assert calls == []  # no hydration yet

        await reg.discover(server="filesystem")
        assert calls == ["filesystem"]

    async def test_on_demand_discover_without_server_hydrates_all_servers(self) -> None:
        calls: list[str] = []

        async def list_tools(server: str) -> list[dict[str, Any]]:
            calls.append(server)
            return [READ_TOOL]

        reg = Registry(
            [ServerConfig(name="filesystem"), ServerConfig(name="search")],
            RegistryConfig(warm_mode="on_demand"),
            list_tools=list_tools,
            execute_tool=_null_executor,
        )
        await reg.start()

        result = await reg.discover()
        assert result["total_matches"] == 2
        assert result["unavailable_servers"] == []
        assert calls == ["filesystem", "search"]

    async def test_background_hydrates_eventually(self) -> None:
        reg = _make_registry(warm_mode="background")
        await reg.start()
        # Yield so the background task can run.
        for _ in range(10):
            await asyncio.sleep(0)
        assert reg._warm_task is not None
        await reg._warm_task
        result = await reg.discover()
        assert result["total_matches"] == 2


class TestDiscover:
    async def test_reports_unavailable_servers(self) -> None:
        reg = _make_registry(
            servers=[ServerConfig(name="filesystem"), ServerConfig(name="broken")],
            tools={"filesystem": [READ_TOOL]},
            errors={"broken": RuntimeError("boom")},
        )
        await reg.start()
        result = await reg.discover()
        assert "broken" in result["unavailable_servers"]
        assert result["total_matches"] == 1

    async def test_limit_is_capped_by_hard_cap(self) -> None:
        tools = [
            {"name": f"t{i}", "description": "", "inputSchema": {"type": "object"}}
            for i in range(50)
        ]
        reg = _make_registry(tools={"filesystem": tools}, max_discover=100, hard_cap=5)
        await reg.start()
        result = await reg.discover(limit=40)
        assert result["returned"] == 5
        assert result["has_more"] is True

    async def test_limit_is_capped_by_max_discover_results(self) -> None:
        tools = [
            {"name": f"t{i}", "description": "", "inputSchema": {"type": "object"}}
            for i in range(50)
        ]
        reg = _make_registry(tools={"filesystem": tools}, max_discover=3, hard_cap=100)
        await reg.start()
        result = await reg.discover(limit=40)
        assert result["returned"] == 3

    async def test_discover_unknown_server_raises(self) -> None:
        reg = _make_registry()
        await reg.start()
        with pytest.raises(ToolNotFoundError, match="Unknown server"):
            await reg.discover(server="ghost")

    async def test_discover_returns_hint_when_more_available(self) -> None:
        tools = [
            {"name": f"t{i}", "description": "", "inputSchema": {"type": "object"}}
            for i in range(10)
        ]
        reg = _make_registry(tools={"filesystem": tools}, max_discover=3)
        await reg.start()
        result = await reg.discover()
        assert "Narrow the query" in result["hint"]


class TestInspect:
    async def test_inspect_returns_schema(self) -> None:
        reg = _make_registry()
        await reg.start()
        result = await reg.inspect("filesystem", "read_file")
        assert result["status"] == "success"
        assert result["required_fields"] == ["path"]
        assert result["accepts_no_args"] is False
        assert result["schema_hash"].startswith("sha256:")

    async def test_inspect_unknown_server_raises(self) -> None:
        reg = _make_registry()
        await reg.start()
        with pytest.raises(ToolNotFoundError, match="Unknown server"):
            await reg.inspect("ghost", "read_file")

    async def test_inspect_unknown_tool_raises(self) -> None:
        reg = _make_registry()
        await reg.start()
        with pytest.raises(ToolNotFoundError, match="Unknown tool"):
            await reg.inspect("filesystem", "no_such_tool")


class TestExecute:
    async def test_happy_path(self) -> None:
        reg = _make_registry()
        await reg.start()
        result = await reg.execute("filesystem", "read_file", {"path": "/etc/hosts"})
        assert result["status"] == "success"
        assert result["tool_result"]["is_error"] is False
        assert result["server"] == "filesystem"
        assert result["tool"] == "read_file"

    async def test_missing_required_raises_validation_error(self) -> None:
        reg = _make_registry()
        await reg.start()
        with pytest.raises(ValidationError):
            await reg.execute("filesystem", "read_file", {})

    async def test_wrong_type_raises_validation_error(self) -> None:
        reg = _make_registry()
        await reg.start()
        with pytest.raises(ValidationError):
            await reg.execute("filesystem", "read_file", {"path": 42})

    async def test_validation_can_be_disabled(self) -> None:
        reg = _make_registry(enable_client_validation=False)
        await reg.start()
        # Without validation, the wrong type flows through to the executor.
        result = await reg.execute("filesystem", "read_file", {"path": 42})
        assert result["status"] == "success"

    async def test_denylisted_tool_raises_policy_denied(self) -> None:
        reg = _make_registry(
            servers=[ServerConfig(name="filesystem", deny_tools=("write_file",))],
        )
        await reg.start()
        with pytest.raises(PolicyDeniedError):
            await reg.execute("filesystem", "write_file", {"path": "/tmp/x", "text": "hi"})

    async def test_unknown_server_raises(self) -> None:
        reg = _make_registry()
        await reg.start()
        with pytest.raises(ToolNotFoundError, match="Unknown server"):
            await reg.execute("ghost", "x", {})

    async def test_unknown_tool_raises(self) -> None:
        reg = _make_registry()
        await reg.start()
        with pytest.raises(ToolNotFoundError, match="Unknown tool"):
            await reg.execute("filesystem", "no_such_tool", {"path": "/a"})


class TestClose:
    async def test_closed_registry_rejects_discover(self) -> None:
        reg = _make_registry()
        await reg.start()
        await reg.close()
        with pytest.raises(RegistryClosedError):
            await reg.discover()

    async def test_closed_registry_rejects_inspect(self) -> None:
        reg = _make_registry()
        await reg.start()
        await reg.close()
        with pytest.raises(RegistryClosedError):
            await reg.inspect("filesystem", "read_file")

    async def test_closed_registry_rejects_execute(self) -> None:
        reg = _make_registry()
        await reg.start()
        await reg.close()
        with pytest.raises(RegistryClosedError):
            await reg.execute("filesystem", "read_file", {"path": "/a"})

    async def test_close_is_idempotent(self) -> None:
        reg = _make_registry()
        await reg.start()
        await reg.close()
        await reg.close()  # should not raise

    async def test_close_cancels_background_warm_task(self) -> None:
        reg = _make_registry(warm_mode="background")
        await reg.start()
        await reg.close()
        # Warm task is cancelled and awaited without leaking.


class TestTelemetry:
    async def test_success_increments_counter(self) -> None:
        tel = Telemetry()
        reg = _make_registry(telemetry=tel)
        await reg.start()
        assert tel.counters["catalog_refresh_total:filesystem:success"] == 1

    async def test_error_increments_error_counter(self) -> None:
        tel = Telemetry()
        reg = _make_registry(
            errors={"filesystem": RuntimeError("nope")},
            telemetry=tel,
        )
        await reg.start()
        assert tel.counters["catalog_refresh_total:filesystem:error"] == 1

    async def test_execute_records_timing(self) -> None:
        tel = Telemetry()
        reg = _make_registry(telemetry=tel)
        await reg.start()
        await reg.execute("filesystem", "read_file", {"path": "/a"})
        timings = tel.timings_ms["execute_duration_ms:filesystem:read_file"]
        assert len(timings) == 1
        assert timings[0] >= 0


class TestHealthSnapshot:
    async def test_ready_state(self) -> None:
        reg = _make_registry()
        await reg.start()
        snap = reg.health_snapshot()
        assert snap["status"] == "ready"
        assert len(snap["servers"]) == 1
        assert snap["servers"][0]["state"] == "ready"
        assert snap["servers"][0]["tool_count"] == 2

    async def test_degraded_state_when_list_fails(self) -> None:
        reg = _make_registry(
            servers=[ServerConfig(name="filesystem"), ServerConfig(name="broken")],
            tools={"filesystem": [READ_TOOL]},
            errors={"broken": RuntimeError("boom")},
        )
        await reg.start()
        snap = reg.health_snapshot()
        assert snap["status"] == "degraded"
        states = {s["name"]: s["state"] for s in snap["servers"]}
        assert states["broken"] == "degraded"
        assert states["filesystem"] == "ready"


async def _null_executor(server: str, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {"isError": False, "content": []}
