from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from .config import ServerConfig

_DEFAULT_CONCURRENCY = {
    "stdio": 1,
    "streamable_http": 4,
    "sse_legacy": 1,
}

_FAILURE_THRESHOLD = 3


@dataclass
class BreakerState:
    failures: int = 0
    open: bool = False


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

    async def execute(
        self,
        call_coro: Any,
        *,
        timeout_ms: int,
        allow_retry: bool,
    ) -> Any:
        if self._closed:
            raise RuntimeError("session_closed")
        if self._breaker.open:
            raise RuntimeError("circuit_open")

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

    @staticmethod
    def _is_retryable(exc: RuntimeError) -> bool:
        return str(exc) in {"broken_pipe", "connection_reset", "session_closed"}
