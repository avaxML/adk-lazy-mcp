from __future__ import annotations

import os
import re
from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

_ENV_RE = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}")


class ServerConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    transport: Literal["stdio", "streamable_http", "sse_legacy"] = "stdio"
    command: str | None = None
    args: tuple[str, ...] = ()
    env: Mapping[str, str] | None = None
    cwd: str | None = None
    url: str | None = None
    headers: Mapping[str, str] | None = None
    trusted: bool = False
    connect_timeout_ms: int = 5_000
    call_timeout_ms: int = 30_000
    max_concurrency: int | None = None
    allow_tools: tuple[str, ...] | None = None
    deny_tools: tuple[str, ...] = ()
    max_inline_bytes: int = 16_384

    @model_validator(mode="after")
    def _validate(self) -> "ServerConfig":
        if not self.name:
            raise ValueError("ServerConfig.name is required")
        if self.connect_timeout_ms <= 0 or self.call_timeout_ms <= 0:
            raise ValueError(f"{self.name}: timeouts must be positive")
        if self.max_concurrency is not None and self.max_concurrency < 1:
            raise ValueError(f"{self.name}: max_concurrency must be >= 1")
        if self.max_inline_bytes < 0:
            raise ValueError(f"{self.name}: max_inline_bytes must be non-negative")
        return self


class RegistryConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    warm_mode: Literal["background", "eager", "on_demand"] = "background"
    summary_ttl_s: int = 300
    max_discover_results: int = 20
    hard_discover_cap: int = 100
    enable_client_validation: bool = True


def resolve_env_vars(value: str, *, strict: bool = True) -> str:
    """Resolve ${VAR} and ${VAR:-default} placeholders."""

    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        default = match.group(2)
        if key in os.environ:
            return os.environ[key]
        if default is not None:
            return default
        if strict:
            raise ValueError(f"Missing required environment variable: {key}")
        return ""

    return _ENV_RE.sub(_replace, value)
