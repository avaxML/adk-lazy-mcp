"""Unit tests for :mod:`adk_lazy_mcp.policy`."""

from __future__ import annotations

import pytest

from adk_lazy_mcp.config import ServerConfig
from adk_lazy_mcp.policy import PolicyDecision, PolicyEngine


class TestValidateServer:
    def test_stdio_is_always_allowed(self) -> None:
        engine = PolicyEngine(strict_https=True, allowed_hosts={"nothing"})
        cfg = ServerConfig(name="fs", transport="stdio", command="mcp-fs")
        assert engine.validate_server(cfg) == PolicyDecision(True)

    def test_http_requires_https_when_strict(self) -> None:
        engine = PolicyEngine(strict_https=True)
        cfg = ServerConfig(
            name="remote",
            transport="streamable_http",
            url="http://example.com/mcp",
        )
        decision = engine.validate_server(cfg)
        assert decision.allowed is False
        assert decision.reason == "require_https"

    def test_http_allowed_when_strict_https_disabled(self) -> None:
        engine = PolicyEngine(strict_https=False)
        cfg = ServerConfig(
            name="remote",
            transport="streamable_http",
            url="http://example.com/mcp",
        )
        assert engine.validate_server(cfg).allowed is True

    def test_https_is_allowed_when_strict(self) -> None:
        engine = PolicyEngine(strict_https=True)
        cfg = ServerConfig(
            name="remote",
            transport="streamable_http",
            url="https://example.com/mcp",
        )
        assert engine.validate_server(cfg).allowed is True

    def test_host_allowlist_blocks_unlisted(self) -> None:
        engine = PolicyEngine(allowed_hosts={"trusted.example.com"})
        cfg = ServerConfig(
            name="remote",
            transport="streamable_http",
            url="https://evil.example.com/mcp",
        )
        decision = engine.validate_server(cfg)
        assert decision.allowed is False
        assert decision.reason == "host_not_allowed"

    def test_host_allowlist_allows_listed(self) -> None:
        engine = PolicyEngine(allowed_hosts={"trusted.example.com"})
        cfg = ServerConfig(
            name="remote",
            transport="streamable_http",
            url="https://trusted.example.com/mcp",
        )
        assert engine.validate_server(cfg).allowed is True

    def test_sse_legacy_gets_same_policy_as_streamable_http(self) -> None:
        engine = PolicyEngine(strict_https=True)
        cfg = ServerConfig(
            name="remote",
            transport="sse_legacy",
            url="http://example.com/mcp",
        )
        assert engine.validate_server(cfg).allowed is False


class TestValidateTool:
    def test_no_allowlist_no_denylist(self) -> None:
        engine = PolicyEngine()
        cfg = ServerConfig(name="fs")
        assert engine.validate_tool(cfg, "anything").allowed is True

    def test_allowlist_permits_listed(self) -> None:
        engine = PolicyEngine()
        cfg = ServerConfig(name="fs", allow_tools=("read", "list"))
        assert engine.validate_tool(cfg, "read").allowed is True

    def test_allowlist_blocks_unlisted(self) -> None:
        engine = PolicyEngine()
        cfg = ServerConfig(name="fs", allow_tools=("read",))
        decision = engine.validate_tool(cfg, "write")
        assert decision.allowed is False
        assert decision.reason == "tool_not_allowlisted"

    def test_empty_allowlist_blocks_everything(self) -> None:
        engine = PolicyEngine()
        cfg = ServerConfig(name="fs", allow_tools=())
        assert engine.validate_tool(cfg, "read").allowed is False

    def test_denylist_blocks_listed(self) -> None:
        engine = PolicyEngine()
        cfg = ServerConfig(name="fs", deny_tools=("rm_rf",))
        decision = engine.validate_tool(cfg, "rm_rf")
        assert decision.allowed is False
        assert decision.reason == "tool_denylisted"

    def test_denylist_permits_unlisted(self) -> None:
        engine = PolicyEngine()
        cfg = ServerConfig(name="fs", deny_tools=("rm_rf",))
        assert engine.validate_tool(cfg, "read").allowed is True

    def test_allowlist_takes_precedence_over_deny(self) -> None:
        # An allowlisted-but-also-denylisted tool should be denied: denylist
        # is the final gate after the allowlist check passes.
        engine = PolicyEngine()
        cfg = ServerConfig(name="fs", allow_tools=("a", "b"), deny_tools=("b",))
        assert engine.validate_tool(cfg, "a").allowed is True
        assert engine.validate_tool(cfg, "b").allowed is False


@pytest.mark.parametrize("scheme", ["http", "ftp", "ws"])
def test_non_https_schemes_blocked_when_strict(scheme: str) -> None:
    engine = PolicyEngine(strict_https=True)
    cfg = ServerConfig(
        name="remote",
        transport="streamable_http",
        url=f"{scheme}://example.com/mcp",
    )
    assert engine.validate_server(cfg).allowed is False
