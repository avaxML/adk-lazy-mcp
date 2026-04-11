"""Unit tests for :mod:`adk_lazy_mcp.session_manager`."""

from __future__ import annotations

import asyncio

import pytest

from adk_lazy_mcp.config import ServerConfig
from adk_lazy_mcp.session_manager import SessionManager


async def _ok() -> dict[str, str]:
    return {"ok": True}


class TestDefaults:
    def test_stdio_concurrency_is_one(self) -> None:
        cfg = ServerConfig(name="fs", transport="stdio")
        sm = SessionManager(cfg)
        assert sm._sem._value == 1  # type: ignore[attr-defined]

    def test_streamable_http_concurrency_is_four(self) -> None:
        cfg = ServerConfig(name="fs", transport="streamable_http", url="https://example.com/mcp")
        sm = SessionManager(cfg)
        assert sm._sem._value == 4  # type: ignore[attr-defined]

    def test_explicit_max_concurrency_is_respected(self) -> None:
        cfg = ServerConfig(name="fs", transport="stdio", max_concurrency=7)
        sm = SessionManager(cfg)
        assert sm._sem._value == 7  # type: ignore[attr-defined]

    def test_breaker_starts_closed(self) -> None:
        sm = SessionManager(ServerConfig(name="fs"))
        assert sm.breaker_open is False


class TestExecute:
    async def test_happy_path_returns_result(self) -> None:
        sm = SessionManager(ServerConfig(name="fs"))
        result = await sm.execute(_ok, timeout_ms=1000, allow_retry=False)
        assert result == {"ok": True}

    async def test_timeout_raises_and_records_failure(self) -> None:
        sm = SessionManager(ServerConfig(name="fs"))

        async def _slow() -> None:
            await asyncio.sleep(0.5)

        with pytest.raises(asyncio.TimeoutError):
            await sm.execute(_slow, timeout_ms=10, allow_retry=False)
        assert sm.breaker_failures == 1

    async def test_breaker_opens_after_three_failures(self) -> None:
        sm = SessionManager(ServerConfig(name="fs"))

        async def _boom() -> None:
            raise RuntimeError("explode")

        for _ in range(3):
            with pytest.raises(RuntimeError, match="explode"):
                await sm.execute(_boom, timeout_ms=500, allow_retry=False)

        assert sm.breaker_open is True
        with pytest.raises(RuntimeError, match="circuit_open"):
            await sm.execute(_ok, timeout_ms=500, allow_retry=False)

    async def test_breaker_recovers_after_cooldown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        now = 100.0
        monkeypatch.setattr("adk_lazy_mcp.session_manager.time.monotonic", lambda: now)
        sm = SessionManager(ServerConfig(name="fs"))

        async def _boom() -> None:
            raise RuntimeError("explode")

        for _ in range(3):
            with pytest.raises(RuntimeError, match="explode"):
                await sm.execute(_boom, timeout_ms=500, allow_retry=False)

        with pytest.raises(RuntimeError, match="circuit_open"):
            await sm.execute(_ok, timeout_ms=500, allow_retry=False)
        assert sm.breaker_opened_at == 100.0

        now += 30.0
        result = await sm.execute(_ok, timeout_ms=500, allow_retry=False)
        assert result == {"ok": True}
        assert sm.breaker_open is False
        assert sm.breaker_failures == 0

    async def test_allow_retry_succeeds_on_second_attempt(self) -> None:
        sm = SessionManager(ServerConfig(name="fs"))
        calls = {"n": 0}

        async def _flaky() -> str:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("broken_pipe")
            return "ok"

        result = await sm.execute(_flaky, timeout_ms=500, allow_retry=True)
        assert result == "ok"
        assert calls["n"] == 2
        # Success after retry resets the breaker counter.
        assert sm.breaker_failures == 0
        assert sm.breaker_open is False

    async def test_allow_retry_false_does_not_retry(self) -> None:
        sm = SessionManager(ServerConfig(name="fs"))
        calls = {"n": 0}

        async def _flaky() -> str:
            calls["n"] += 1
            raise RuntimeError("broken_pipe")

        with pytest.raises(RuntimeError, match="broken_pipe"):
            await sm.execute(_flaky, timeout_ms=500, allow_retry=False)
        assert calls["n"] == 1

    async def test_retry_only_happens_for_retryable_errors(self) -> None:
        sm = SessionManager(ServerConfig(name="fs"))
        calls = {"n": 0}

        async def _other() -> None:
            calls["n"] += 1
            raise RuntimeError("not_retryable")

        with pytest.raises(RuntimeError, match="not_retryable"):
            await sm.execute(_other, timeout_ms=500, allow_retry=True)
        assert calls["n"] == 1

    async def test_retry_that_also_fails_records_failure(self) -> None:
        sm = SessionManager(ServerConfig(name="fs"))
        calls = {"n": 0}

        async def _always() -> None:
            calls["n"] += 1
            raise RuntimeError("broken_pipe")

        with pytest.raises(RuntimeError, match="broken_pipe"):
            await sm.execute(_always, timeout_ms=500, allow_retry=True)
        assert calls["n"] == 2
        assert sm.breaker_failures == 1

    async def test_closed_session_rejects_execute(self) -> None:
        sm = SessionManager(ServerConfig(name="fs"))
        await sm.close()
        with pytest.raises(RuntimeError, match="session_closed"):
            await sm.execute(_ok, timeout_ms=100, allow_retry=False)

    async def test_success_resets_failure_counter(self) -> None:
        sm = SessionManager(ServerConfig(name="fs"))

        async def _boom() -> None:
            raise RuntimeError("explode")

        with pytest.raises(RuntimeError, match="explode"):
            await sm.execute(_boom, timeout_ms=500, allow_retry=False)
        assert sm.breaker_failures == 1

        await sm.execute(_ok, timeout_ms=500, allow_retry=False)
        assert sm.breaker_failures == 0

    async def test_concurrency_limit_is_enforced(self) -> None:
        sm = SessionManager(ServerConfig(name="fs", max_concurrency=2))
        running = {"count": 0, "peak": 0}

        async def _slow() -> str:
            running["count"] += 1
            running["peak"] = max(running["peak"], running["count"])
            await asyncio.sleep(0.02)
            running["count"] -= 1
            return "ok"

        tasks = [sm.execute(_slow, timeout_ms=1000, allow_retry=False) for _ in range(5)]
        await asyncio.gather(*tasks)
        assert running["peak"] <= 2
