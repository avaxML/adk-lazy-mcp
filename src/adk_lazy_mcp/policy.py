from __future__ import annotations

from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict

from .config import ServerConfig


class PolicyDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    allowed: bool
    reason: str | None = None


class PolicyEngine:
    def __init__(self, *, strict_https: bool = True, allowed_hosts: set[str] | None = None):
        self._strict_https = strict_https
        self._allowed_hosts = allowed_hosts or set()

    def validate_server(self, cfg: ServerConfig) -> PolicyDecision:
        if cfg.transport in {"streamable_http", "sse_legacy"} and cfg.url:
            parsed = urlparse(cfg.url)
            if self._strict_https and parsed.scheme != "https":
                return PolicyDecision(allowed=False, reason="require_https")
            if self._allowed_hosts and parsed.hostname not in self._allowed_hosts:
                return PolicyDecision(allowed=False, reason="host_not_allowed")
        return PolicyDecision(allowed=True)

    def validate_tool(self, cfg: ServerConfig, tool_name: str) -> PolicyDecision:
        if cfg.allow_tools is not None and tool_name not in cfg.allow_tools:
            return PolicyDecision(allowed=False, reason="tool_not_allowlisted")
        if tool_name in cfg.deny_tools:
            return PolicyDecision(allowed=False, reason="tool_denylisted")
        return PolicyDecision(allowed=True)
