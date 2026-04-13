from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_RE = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}")
_DEFAULT_CLUSTER_STOPWORDS = (
    "and",
    "for",
    "file",
    "files",
    "from",
    "input",
    "json",
    "object",
    "output",
    "path",
    "the",
    "text",
    "tool",
    "type",
    "value",
    "values",
    "with",
    "content",
    "contents",
)


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
    def _validate(self) -> ServerConfig:
        if not self.name.strip():
            raise ValueError("ServerConfig.name must not be blank")
        if self.connect_timeout_ms <= 0 or self.call_timeout_ms <= 0:
            raise ValueError(f"{self.name}: timeouts must be positive")
        if self.max_concurrency is not None and self.max_concurrency < 1:
            raise ValueError(f"{self.name}: max_concurrency must be >= 1")
        if self.max_inline_bytes < 0:
            raise ValueError(f"{self.name}: max_inline_bytes must be non-negative")
        if self.command is not None and not self.command.strip():
            raise ValueError(f"{self.name}: command must not be blank")
        if self.transport in {"streamable_http", "sse_legacy"} and self.url is not None:
            parsed = urlparse(self.url)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError(f"{self.name}: url must include a scheme and host")
        return self


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    bm25_k1: float = 1.6
    bm25_b: float = 0.75
    reciprocal_rank_k: int = 60
    semantic_rerank_limit: int = 12
    semantic_fallback_threshold: float = 0.15
    semantic_name_fallback_threshold: float = 0.72
    leader_cluster_threshold: float = 0.35
    min_cluster_token_length: int = 3
    name_term_weight: int = 3
    schema_property_weight: int = 2
    required_field_weight: int = 2
    trigram_size: int = 3
    trigram_weight: float = 0.35
    cluster_stopwords: tuple[str, ...] = _DEFAULT_CLUSTER_STOPWORDS

    @field_validator("cluster_stopwords", mode="before")
    @classmethod
    def _parse_cluster_stopwords(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        if not stripped:
            return ()
        if stripped.startswith("["):
            return json.loads(stripped)
        return tuple(part.strip() for part in stripped.split(","))

    @field_validator("cluster_stopwords")
    @classmethod
    def _normalize_cluster_stopwords(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = [
            item.strip().lower() for item in value if isinstance(item, str) and item.strip()
        ]
        return tuple(dict.fromkeys(normalized))

    @model_validator(mode="after")
    def _validate(self) -> RetrievalConfig:
        if self.bm25_k1 <= 0:
            raise ValueError("bm25_k1 must be positive")
        if not 0 <= self.bm25_b <= 1:
            raise ValueError("bm25_b must be between 0 and 1")
        if self.reciprocal_rank_k < 0:
            raise ValueError("reciprocal_rank_k must be non-negative")
        if self.semantic_rerank_limit < 1:
            raise ValueError("semantic_rerank_limit must be >= 1")
        if not 0 <= self.semantic_fallback_threshold <= 1:
            raise ValueError("semantic_fallback_threshold must be between 0 and 1")
        if not 0 <= self.semantic_name_fallback_threshold <= 1:
            raise ValueError("semantic_name_fallback_threshold must be between 0 and 1")
        if not 0 <= self.leader_cluster_threshold <= 1:
            raise ValueError("leader_cluster_threshold must be between 0 and 1")
        if self.min_cluster_token_length < 1:
            raise ValueError("min_cluster_token_length must be >= 1")
        if self.name_term_weight < 1:
            raise ValueError("name_term_weight must be >= 1")
        if self.schema_property_weight < 1:
            raise ValueError("schema_property_weight must be >= 1")
        if self.required_field_weight < 1:
            raise ValueError("required_field_weight must be >= 1")
        if self.trigram_size < 1:
            raise ValueError("trigram_size must be >= 1")
        if self.trigram_weight <= 0:
            raise ValueError("trigram_weight must be positive")
        return self


class RegistryConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ADK_LAZY_MCP_",
        env_nested_delimiter="__",
        extra="ignore",
        frozen=True,
    )

    warm_mode: Literal["background", "eager", "on_demand"] = "background"
    summary_ttl_s: int = 300
    max_discover_results: int = 20
    hard_discover_cap: int = 100
    enable_client_validation: bool = True
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)

    @model_validator(mode="after")
    def _validate(self) -> RegistryConfig:
        if self.summary_ttl_s <= 0:
            raise ValueError("summary_ttl_s must be positive")
        if self.max_discover_results < 1:
            raise ValueError("max_discover_results must be >= 1")
        if self.hard_discover_cap < 1:
            raise ValueError("hard_discover_cap must be >= 1")
        return self


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
