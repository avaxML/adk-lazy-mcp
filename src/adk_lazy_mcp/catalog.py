from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .config import RegistryConfig, ServerConfig

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CAMEL_RE = re.compile(r"(?<!^)(?=[A-Z])")
_BM25_K1 = 1.6
_BM25_B = 0.75
_RRF_K = 60
_SEMANTIC_RERANK_LIMIT = 12
_SEMANTIC_FALLBACK_THRESHOLD = 0.15
_LEADER_CLUSTER_THRESHOLD = 0.35
_NAME_TERM_WEIGHT = 3
_SCHEMA_PROPERTY_WEIGHT = 2
_REQUIRED_FIELD_WEIGHT = 2
_TRIGRAM_SIZE = 3
_TRIGRAM_WEIGHT = 0.35
_CLUSTER_STOPWORDS = frozenset(
    {
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
    }
)


class ServerState(str, Enum):
    UNSEEN = "unseen"
    READY = "ready"
    DEGRADED = "degraded"
    CLOSED = "closed"


class ToolSchema(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]
    schema_hash: str = ""
    name_lower: str = ""
    description_lower: str = ""
    search_terms: tuple[str, ...] = ()

    def model_post_init(self, __context: Any) -> None:
        if not self.schema_hash:
            self.schema_hash = (
                "sha256:"
                + hashlib.sha256(_canonical_json(self.input_schema).encode("utf-8")).hexdigest()
            )
        self.name_lower = self.name.lower()
        self.description_lower = self.description.lower()
        self.search_terms = tuple(
            _build_document_terms(self.name, self.description, self.input_schema)
        )


class CatalogEntry(BaseModel):
    config: ServerConfig
    state: ServerState = ServerState.UNSEEN
    tools: dict[str, ToolSchema] = Field(default_factory=dict)
    refreshed_at: float = 0.0
    last_error: str | None = None
    version: str = ""


class CatalogManager:
    def __init__(self, registry_config: RegistryConfig) -> None:
        self._cfg = registry_config
        self._entries: dict[str, CatalogEntry] = {}
        self._lexical_indexes: dict[str, _ServerLexicalIndex] = {}
        self._semantic_vectors: dict[str, dict[str, dict[str, float]]] = {}
        self._tool_families: dict[str, dict[str, str]] = {}
        self._lock = asyncio.Lock()

    def register_server(self, server: ServerConfig) -> None:
        self._entries[server.name] = CatalogEntry(config=server)
        self._lexical_indexes[server.name] = _ServerLexicalIndex.empty()
        self._semantic_vectors[server.name] = {}
        self._tool_families[server.name] = {}

    def get_entry(self, server: str) -> CatalogEntry:
        return self._entries[server]

    def iter_entries(self) -> Iterator[tuple[str, CatalogEntry]]:
        return iter(self._entries.items())

    def server_names(self) -> list[str]:
        return sorted(self._entries.keys())

    async def hydrate_server(self, server: str, tools: list[dict[str, Any]]) -> None:
        mapped: dict[str, ToolSchema] = {}
        for t in tools:
            mapped[t["name"]] = ToolSchema(
                name=t["name"],
                description=t.get("description", ""),
                input_schema=t.get("inputSchema", {"type": "object", "properties": {}}),
            )
        version = self._catalog_hash(mapped)
        lexical_index = _ServerLexicalIndex.build(server, mapped)
        semantic_vectors = {
            name: _build_semantic_vector((*_tokenize_identifier(server), *tool.search_terms))
            for name, tool in mapped.items()
        }
        tool_families = _build_tool_families(mapped)
        async with self._lock:
            entry = self._entries[server]
            entry.tools = mapped
            entry.refreshed_at = time.time()
            entry.version = version
            entry.last_error = None
            entry.state = ServerState.READY
            self._lexical_indexes[server] = lexical_index
            self._semantic_vectors[server] = semantic_vectors
            self._tool_families[server] = tool_families

    def mark_error(self, server: str, error: str) -> None:
        e = self._entries[server]
        e.last_error = error
        e.state = ServerState.DEGRADED

    def is_stale(self, server: str) -> bool:
        e = self._entries[server]
        if e.refreshed_at == 0.0:
            return True
        return (time.time() - e.refreshed_at) > self._cfg.summary_ttl_s

    def discover(self, query: str, server: str | None = None) -> list[dict[str, str]]:
        q = query.strip().lower()
        entries = ((server, self._entries[server]),) if server else self._entries.items()
        if not q:
            return self._list_all_tools(entries)

        matches: list[_SearchMatch] = []
        for name, e in entries:
            if e.state != ServerState.READY:
                continue
            matches.extend(self._rank_server_matches(name, e, q))
        matches.sort(key=lambda item: (-item.score, item.tool))
        return [self._serialize_match(match) for match in matches]

    @staticmethod
    def _catalog_hash(tools: dict[str, ToolSchema]) -> str:
        payload = "|".join(sorted(f"{k}:{v.description}" for k, v in tools.items()))
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _list_all_tools(
        self, entries: Iterator[tuple[str, CatalogEntry]] | tuple[tuple[str, CatalogEntry], ...]
    ) -> list[dict[str, str]]:
        results: list[dict[str, str]] = []
        for name, entry in entries:
            if entry.state != ServerState.READY:
                continue
            for tool_name in sorted(entry.tools):
                results.append(
                    {
                        "server": name,
                        "tool": tool_name,
                        "description": entry.tools[tool_name].description,
                        "family": self._tool_families.get(name, {}).get(tool_name, tool_name),
                    }
                )
        return results

    def _rank_server_matches(
        self, server: str, entry: CatalogEntry, query: str
    ) -> list[_SearchMatch]:
        lexical_index = self._lexical_indexes[server]
        lexical_matches = lexical_index.search(query, entry.tools)
        if not lexical_matches:
            return self._semantic_fallback_matches(server, entry, query)

        candidate_names = [tool_name for tool_name, _ in lexical_matches[:_SEMANTIC_RERANK_LIMIT]]
        semantic_scores = self._semantic_scores(server, candidate_names, query)
        semantic_ranks = {
            tool_name: rank
            for rank, (tool_name, _) in enumerate(
                sorted(semantic_scores.items(), key=lambda item: (-item[1], item[0])),
                start=1,
            )
        }
        families = self._tool_families.get(server, {})
        ranked: list[_SearchMatch] = []
        for lexical_rank, (tool_name, _) in enumerate(lexical_matches, start=1):
            score = _reciprocal_rank_fusion_score(lexical_rank)
            if tool_name in semantic_ranks:
                score += _reciprocal_rank_fusion_score(semantic_ranks[tool_name])
            ranked.append(
                _SearchMatch(
                    server=server,
                    tool=tool_name,
                    description=entry.tools[tool_name].description,
                    family=families.get(tool_name, tool_name),
                    score=score,
                )
            )
        return ranked

    def _semantic_fallback_matches(
        self, server: str, entry: CatalogEntry, query: str
    ) -> list[_SearchMatch]:
        semantic_scores = self._semantic_scores(server, entry.tools.keys(), query)
        families = self._tool_families.get(server, {})
        matches: list[_SearchMatch] = []
        for rank, (tool_name, score) in enumerate(
            sorted(semantic_scores.items(), key=lambda item: (-item[1], item[0])),
            start=1,
        ):
            if score < _SEMANTIC_FALLBACK_THRESHOLD:
                continue
            matches.append(
                _SearchMatch(
                    server=server,
                    tool=tool_name,
                    description=entry.tools[tool_name].description,
                    family=families.get(tool_name, tool_name),
                    score=_reciprocal_rank_fusion_score(rank),
                )
            )
        return matches

    def _semantic_scores(
        self,
        server: str,
        tool_names: Iterable[str],
        query: str,
    ) -> dict[str, float]:
        query_vector = _build_semantic_vector(_extract_alphanumeric_tokens(query))
        if not query_vector:
            return {}
        return {
            tool_name: _cosine_similarity(
                query_vector, self._semantic_vectors[server].get(tool_name, {})
            )
            for tool_name in tool_names
        }

    @staticmethod
    def _serialize_match(match: _SearchMatch) -> dict[str, str]:
        return {
            "server": match.server,
            "tool": match.tool,
            "description": match.description,
            "family": match.family,
        }


@dataclass(frozen=True, slots=True)
class _SearchMatch:
    server: str
    tool: str
    description: str
    family: str
    score: float


@dataclass(frozen=True, slots=True)
class _ServerLexicalIndex:
    doc_terms: dict[str, Counter[str]]
    doc_lengths: dict[str, int]
    doc_freq: Counter[str]
    avg_doc_len: float

    @classmethod
    def empty(cls) -> _ServerLexicalIndex:
        return cls(doc_terms={}, doc_lengths={}, doc_freq=Counter(), avg_doc_len=0.0)

    @classmethod
    def build(cls, server: str, tools: dict[str, ToolSchema]) -> _ServerLexicalIndex:
        server_terms = tuple(_tokenize_identifier(server))
        doc_terms: dict[str, Counter[str]] = {}
        doc_lengths: dict[str, int] = {}
        doc_freq: Counter[str] = Counter()
        total_terms = 0
        for name, tool in tools.items():
            terms = Counter((*server_terms, *tool.search_terms))
            doc_terms[name] = terms
            doc_len = sum(terms.values())
            doc_lengths[name] = doc_len
            total_terms += doc_len
            doc_freq.update(terms.keys())
        avg_doc_len = total_terms / len(doc_terms) if doc_terms else 0.0
        return cls(
            doc_terms=doc_terms,
            doc_lengths=doc_lengths,
            doc_freq=doc_freq,
            avg_doc_len=avg_doc_len,
        )

    def search(self, query: str, tools: dict[str, ToolSchema]) -> list[tuple[str, float]]:
        if not query:
            return [(name, 1.0) for name in sorted(tools)]

        query_terms = _extract_alphanumeric_tokens(query)
        if not query_terms:
            return []

        corpus_size = len(self.doc_terms)
        results: list[tuple[str, float]] = []
        for tool_name, term_freqs in self.doc_terms.items():
            score = 0.0
            for term in query_terms:
                freq = term_freqs.get(term, 0)
                if freq == 0:
                    continue
                doc_freq = self.doc_freq.get(term, 0)
                if doc_freq == 0:
                    continue
                idf = math.log(1.0 + ((corpus_size - doc_freq + 0.5) / (doc_freq + 0.5)))
                doc_len = self.doc_lengths[tool_name]
                norm = 1.0 - _BM25_B + _BM25_B * (doc_len / max(self.avg_doc_len, 1.0))
                score += idf * ((freq * (_BM25_K1 + 1.0)) / (freq + _BM25_K1 * norm))

            boost = _name_match_boost(query, tools[tool_name])
            if score <= 0 and boost <= 0:
                continue
            results.append((tool_name, score + boost))

        results.sort(key=lambda item: (-item[1], item[0]))
        return results


def _canonical_json(value: Any) -> str:
    """Return a stable JSON string so semantically equal schemas hash the same."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _build_document_terms(name: str, description: str, schema: dict[str, Any]) -> list[str]:
    name_terms = _tokenize_identifier(name)
    description_terms = _extract_alphanumeric_tokens(description)
    schema_terms = _schema_terms(schema)
    return [*(name_terms * _NAME_TERM_WEIGHT), *description_terms, *schema_terms]


def _schema_terms(schema: Any) -> list[str]:
    tokens: list[str] = []
    if not isinstance(schema, dict):
        return tokens

    properties = schema.get("properties")
    if isinstance(properties, dict):
        for prop_name, prop_schema in properties.items():
            prop_terms = _tokenize_identifier(prop_name)
            tokens.extend(prop_terms * _SCHEMA_PROPERTY_WEIGHT)
            tokens.extend(_schema_terms(prop_schema))

    required = schema.get("required")
    if isinstance(required, list):
        for item in required:
            if isinstance(item, str):
                required_terms = _tokenize_identifier(item)
                tokens.extend(required_terms * _REQUIRED_FIELD_WEIGHT)

    items = schema.get("items")
    if isinstance(items, dict):
        tokens.extend(_schema_terms(items))

    for key in ("title", "description"):
        value = schema.get(key)
        if isinstance(value, str):
            tokens.extend(_extract_alphanumeric_tokens(value))

    for key in ("allOf", "anyOf", "oneOf"):
        value = schema.get(key)
        if isinstance(value, list):
            for item in value:
                tokens.extend(_schema_terms(item))
    return tokens


def _tokenize_identifier(value: str) -> list[str]:
    expanded = _CAMEL_RE.sub(" ", value.replace("-", " ").replace("_", " "))
    return _extract_alphanumeric_tokens(expanded)


def _extract_alphanumeric_tokens(value: str) -> list[str]:
    return _TOKEN_RE.findall(value.lower())


def _name_match_boost(query: str, tool: ToolSchema) -> float:
    name = tool.name_lower
    if name == query:
        return 3.0
    if name.startswith(query):
        return 2.0
    if query in name:
        return 1.0
    if query in tool.description_lower:
        return 0.3
    return 0.0


def _build_semantic_vector(token_sequence: tuple[str, ...] | list[str]) -> dict[str, float]:
    features: Counter[str] = Counter()
    for token in token_sequence:
        if not token:
            continue
        features[f"tok:{token}"] += 1.0
        if len(token) < _TRIGRAM_SIZE:
            continue
        padded = f"^{token}$"
        for idx in range(len(padded) - (_TRIGRAM_SIZE - 1)):
            features[f"tri:{padded[idx : idx + _TRIGRAM_SIZE]}"] += _TRIGRAM_WEIGHT
    norm = math.sqrt(sum(weight * weight for weight in features.values()))
    if norm == 0.0:
        return {}
    return {feature: weight / norm for feature, weight in features.items()}


def _cosine_similarity(vector_a: dict[str, float], vector_b: dict[str, float]) -> float:
    if not vector_a or not vector_b:
        return 0.0
    if len(vector_a) > len(vector_b):
        vector_a, vector_b = vector_b, vector_a
    return sum(weight * vector_b.get(feature, 0.0) for feature, weight in vector_a.items())


def _build_tool_families(tools: dict[str, ToolSchema]) -> dict[str, str]:
    leaders: list[tuple[str, set[str]]] = []
    assignments: dict[str, str] = {}
    for tool in sorted(tools.values(), key=lambda item: item.name):
        signature = _cluster_signature(tool)
        best_leader = tool.name
        best_similarity = 0.0
        for leader_name, leader_signature in leaders:
            similarity = _jaccard(signature, leader_signature)
            if similarity > best_similarity:
                best_similarity = similarity
                best_leader = leader_name
        if best_similarity >= _LEADER_CLUSTER_THRESHOLD:
            assignments[tool.name] = best_leader
            continue
        leaders.append((tool.name, signature))
        assignments[tool.name] = tool.name
    return assignments


def _cluster_signature(tool: ToolSchema) -> set[str]:
    return {
        token for token in tool.search_terms if len(token) > 2 and token not in _CLUSTER_STOPWORDS
    }


def _jaccard(signature_a: set[str], signature_b: set[str]) -> float:
    if not signature_a or not signature_b:
        return 0.0
    intersection = len(signature_a & signature_b)
    if intersection == 0:
        return 0.0
    return intersection / len(signature_a | signature_b)


def _reciprocal_rank_fusion_score(rank: int) -> float:
    return 1.0 / (_RRF_K + rank)
