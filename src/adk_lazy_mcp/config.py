from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Literal, Mapping

_ENV_RE = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}")


@dataclass(frozen=True)
class ServerConfig:
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


@dataclass(frozen=True)
class RegistryConfig:
    warm_mode: Literal["background", "eager", "on_demand"] = "background"
    summary_ttl_s: int = 300
    schema_ttl_s: int = 300
    schema_cache_mode: Literal["retain", "summary_only"] = "retain"
    max_discover_results: int = 20
    hard_discover_cap: int = 100
    strict_env: bool = True
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
