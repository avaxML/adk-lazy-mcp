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
        max_c = config.max_concurrency or _DEFAULT_CONCURRENCY[config.transport]
        self._sem = asyncio.Semaphore(max_c)
        self._breaker = BreakerState()
        self._closed = False

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

        async with self._sem:
            try:
                return await asyncio.wait_for(call_coro(), timeout=timeout_ms / 1000)
            except TimeoutError:
                self._breaker.failures += 1
                raise
            except RuntimeError as exc:
                if allow_retry and self._is_retryable(exc):
                    return await asyncio.wait_for(call_coro(), timeout=timeout_ms / 1000)
                self._breaker.failures += 1
                if self._breaker.failures >= 3:
                    self._breaker.open = True
                raise

    async def close(self) -> None:
        self._closed = True

    @staticmethod
    def _is_retryable(exc: RuntimeError) -> bool:
        return str(exc) in {"broken_pipe", "connection_reset", "session_closed"}
