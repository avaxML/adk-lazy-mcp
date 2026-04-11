"""Unit tests for :mod:`adk_lazy_mcp.config`."""

from __future__ import annotations

import pytest

from adk_lazy_mcp.config import RegistryConfig, ServerConfig, resolve_env_vars


class TestServerConfig:
    def test_minimal_stdio_config(self) -> None:
        cfg = ServerConfig(name="filesystem")
        assert cfg.transport == "stdio"
        assert cfg.connect_timeout_ms == 5_000
        assert cfg.call_timeout_ms == 30_000
        assert cfg.max_concurrency is None
        assert cfg.trusted is False

    def test_empty_name_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="name is required"):
            ServerConfig(name="")

    def test_unknown_transport_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown transport"):
            ServerConfig(name="srv", transport="grpc")  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"connect_timeout_ms": 0}, "timeouts must be positive"),
            ({"connect_timeout_ms": -1}, "timeouts must be positive"),
            ({"call_timeout_ms": 0}, "timeouts must be positive"),
            ({"call_timeout_ms": -50}, "timeouts must be positive"),
        ],
    )
    def test_non_positive_timeouts_are_rejected(self, kwargs: dict[str, int], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            ServerConfig(name="srv", **kwargs)

    def test_bad_concurrency_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_concurrency must be >= 1"):
            ServerConfig(name="srv", max_concurrency=0)

    def test_concurrency_of_one_is_allowed(self) -> None:
        cfg = ServerConfig(name="srv", max_concurrency=1)
        assert cfg.max_concurrency == 1

    def test_negative_inline_bytes_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_inline_bytes must be non-negative"):
            ServerConfig(name="srv", max_inline_bytes=-1)

    def test_inline_bytes_of_zero_is_allowed(self) -> None:
        cfg = ServerConfig(name="srv", max_inline_bytes=0)
        assert cfg.max_inline_bytes == 0

    def test_is_frozen(self) -> None:
        cfg = ServerConfig(name="srv")
        with pytest.raises(AttributeError):
            cfg.name = "other"  # type: ignore[misc]


class TestRegistryConfig:
    def test_defaults(self) -> None:
        cfg = RegistryConfig()
        assert cfg.warm_mode == "background"
        assert cfg.summary_ttl_s == 300
        assert cfg.max_discover_results == 20
        assert cfg.hard_discover_cap == 100
        assert cfg.enable_client_validation is True

    def test_is_frozen(self) -> None:
        cfg = RegistryConfig()
        with pytest.raises(AttributeError):
            cfg.warm_mode = "eager"  # type: ignore[misc]


class TestResolveEnvVars:
    def test_plain_value_is_passthrough(self) -> None:
        assert resolve_env_vars("no-placeholders-here") == "no-placeholders-here"

    def test_simple_substitution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_TOKEN", "secret-abc")
        assert resolve_env_vars("Bearer ${MY_TOKEN}") == "Bearer secret-abc"

    def test_default_when_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MISSING_VAR", raising=False)
        assert resolve_env_vars("${MISSING_VAR:-fallback}") == "fallback"

    def test_default_ignored_when_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PRESENT_VAR", "real-value")
        assert resolve_env_vars("${PRESENT_VAR:-fallback}") == "real-value"

    def test_strict_missing_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NOPE", raising=False)
        with pytest.raises(ValueError, match="Missing required environment variable"):
            resolve_env_vars("${NOPE}")

    def test_lenient_missing_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NOPE", raising=False)
        assert resolve_env_vars("[${NOPE}]", strict=False) == "[]"

    def test_multiple_substitutions(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HOST", "example.com")
        monkeypatch.setenv("PORT", "8443")
        assert resolve_env_vars("https://${HOST}:${PORT}/x") == "https://example.com:8443/x"
