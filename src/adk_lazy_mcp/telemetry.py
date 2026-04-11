from __future__ import annotations

import time
from collections import defaultdict


class Telemetry:
    """Lightweight in-memory telemetry sink."""

    def __init__(self) -> None:
        self.counters: dict[str, int] = defaultdict(int)
        self.timings_ms: dict[str, list[float]] = defaultdict(list)

    def incr(self, name: str, value: int = 1) -> None:
        self.counters[name] += value

    def time_ms(self, name: str, duration_ms: float) -> None:
        self.timings_ms[name].append(duration_ms)

    def timer(self, name: str):
        start = time.perf_counter()

        def _finish() -> None:
            self.time_ms(name, (time.perf_counter() - start) * 1_000)

        return _finish
