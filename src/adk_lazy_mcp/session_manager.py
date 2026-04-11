from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from .config import ServerConfig

_DEFAULT_CONCURRENCY = {
    "stdio": 1,
    "streamable_http": 4,
    "sse_legacy": 1,
}

_FAILURE_THRESHOLD = 3
_BREAKER_RESET_TIMEOUT_S = 30.0


@dataclass
class BreakerState:
    failures: int = 0
    open: bool = False
    opened_at: float | None = None


class SessionManager:
    """Transport-aware execution wrapper.

    This implementation intentionally abstracts over MCP transport; callers provide
    an executor coroutine used for real calls or test doubles.
    """

    def __init__(self, config: ServerConfig) -> None:
        max_c = config.max_concurrency or _DEFAULT_CONCURRENCY.get(config.transport, 1)
        self._sem = asyncio.Semaphore(max_c)
        self._breaker = BreakerState()
        self._closed = False

    @property
    def breaker_open(self) -> bool:
        return self._breaker.open

    @property
    def breaker_failures(self) -> int:
        return self._breaker.failures

    async def execute(
        self,
        call_coro: Callable[[], Awaitable[Any]],
        *,
        timeout_ms: int,
        allow_retry: bool,
    ) -> Any:
        if self._closed:
            raise RuntimeError("session_closed")
        if self._breaker.open:
            if not self._breaker_ready():
                raise RuntimeError("circuit_open")
            self._reset_breaker()

        timeout_s = timeout_ms / 1000
        async with self._sem:
            try:
                result = await asyncio.wait_for(call_coro(), timeout=timeout_s)
            except TimeoutError:
                self._record_failure()
                raise
            except RuntimeError as exc:
                if allow_retry and self._is_retryable(exc):
                    try:
                        result = await asyncio.wait_for(call_coro(), timeout=timeout_s)
                    except Exception:
                        self._record_failure()
                        raise
                    self._breaker.failures = 0
                    return result
                self._record_failure()
                raise
            self._breaker.failures = 0
            return result

    async def close(self) -> None:
        self._closed = True

    def _record_failure(self) -> None:
        self._breaker.failures += 1
        if self._breaker.failures >= _FAILURE_THRESHOLD:
            self._breaker.open = True
            self._breaker.opened_at = time.monotonic()

    def _reset_breaker(self) -> None:
        self._breaker.failures = 0
        self._breaker.open = False
        self._breaker.opened_at = None

    def _breaker_ready(self) -> bool:
        opened_at = self._breaker.opened_at
        if opened_at is None:
            return False
        return (time.monotonic() - opened_at) >= _BREAKER_RESET_TIMEOUT_S

    @staticmethod
    def _is_retryable(exc: RuntimeError) -> bool:
        return str(exc) in {"broken_pipe", "connection_reset", "session_closed"}
