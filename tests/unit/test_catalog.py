"""Unit tests for :mod:`adk_lazy_mcp.catalog`."""

from __future__ import annotations

from typing import Any

import pytest

from adk_lazy_mcp.catalog import (
    CatalogManager,
    ServerState,
    ToolSchema,
    _reciprocal_rank_fusion_score,
)
from adk_lazy_mcp.config import RegistryConfig, RetrievalConfig, ServerConfig


def _tool(name: str, description: str = "", schema: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "inputSchema": schema or {"type": "object", "properties": {}},
    }


class TestToolSchema:
    def test_hash_is_stable_for_same_schema(self) -> None:
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        a = ToolSchema(name="read", description="", input_schema=schema)
        b = ToolSchema(name="read", description="", input_schema=schema)
        assert a.schema_hash == b.schema_hash
        assert a.schema_hash.startswith("sha256:")

    def test_hash_changes_when_schema_changes(self) -> None:
        a = ToolSchema(
            name="read",
            description="",
            input_schema={"type": "object", "properties": {"x": {"type": "string"}}},
        )
        b = ToolSchema(
            name="read",
            description="",
            input_schema={"type": "object", "properties": {"y": {"type": "string"}}},
        )
        assert a.schema_hash != b.schema_hash

    def test_hash_is_stable_for_equivalent_nested_schema_order(self) -> None:
        a = ToolSchema(
            name="read",
            description="",
            input_schema={
                "type": "object",
                "properties": {
                    "x": {
                        "type": "object",
                        "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
                    }
                },
            },
        )
        b = ToolSchema(
            name="read",
            description="",
            input_schema={
                "properties": {
                    "x": {
                        "properties": {"b": {"type": "integer"}, "a": {"type": "string"}},
                        "type": "object",
                    }
                },
                "type": "object",
            },
        )
        assert a.schema_hash == b.schema_hash

    def test_lowercased_fields_populated(self) -> None:
        ts = ToolSchema(name="ReadFile", description="Reads FILES", input_schema={})
        assert ts.name_lower == "readfile"
        assert ts.description_lower == "reads files"


class TestCatalogManager:
    @pytest.fixture
    def manager(self) -> CatalogManager:
        cm = CatalogManager(RegistryConfig(summary_ttl_s=300))
        cm.register_server(ServerConfig(name="filesystem"))
        cm.register_server(ServerConfig(name="search"))
        return cm

    def test_initial_state_is_unseen(self, manager: CatalogManager) -> None:
        entry = manager.get_entry("filesystem")
        assert entry.state is ServerState.UNSEEN
        assert entry.tools == {}
        assert manager.server_names() == ["filesystem", "search"]

    async def test_hydrate_server_marks_ready(self, manager: CatalogManager) -> None:
        await manager.hydrate_server("filesystem", [_tool("read"), _tool("write")])
        entry = manager.get_entry("filesystem")
        assert entry.state is ServerState.READY
        assert set(entry.tools) == {"read", "write"}
        assert entry.refreshed_at > 0
        assert entry.last_error is None

    def test_mark_error_sets_degraded(self, manager: CatalogManager) -> None:
        manager.mark_error("filesystem", "boom")
        entry = manager.get_entry("filesystem")
        assert entry.state is ServerState.DEGRADED
        assert entry.last_error == "boom"

    def test_is_stale_when_never_hydrated(self, manager: CatalogManager) -> None:
        assert manager.is_stale("filesystem") is True

    async def test_is_stale_after_ttl(
        self, manager: CatalogManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        base = 1_000_000.0
        monkeypatch.setattr("adk_lazy_mcp.catalog.time.time", lambda: base)
        await manager.hydrate_server("filesystem", [_tool("read")])
        assert manager.is_stale("filesystem") is False
        monkeypatch.setattr("adk_lazy_mcp.catalog.time.time", lambda: base + 301)
        assert manager.is_stale("filesystem") is True

    async def test_discover_scoring_order(self, manager: CatalogManager) -> None:
        await manager.hydrate_server(
            "filesystem",
            [
                _tool("read", "Reads a file"),
                _tool("read_file", "Reads a file"),
                _tool("quickread", "Reads a file"),
                _tool("unrelated", "Contains the word read in the description"),
                _tool("nomatch", "Totally unrelated"),
            ],
        )
        results = manager.discover("read")
        names = [r["tool"] for r in results]
        # Exact match first, then prefix (ties broken alphabetically),
        # then substring match on name, then description match.
        assert names == ["read", "read_file", "quickread", "unrelated"]
        assert "nomatch" not in names

    async def test_discover_empty_query_returns_everything(self, manager: CatalogManager) -> None:
        await manager.hydrate_server("filesystem", [_tool("a"), _tool("b")])
        results = manager.discover("")
        assert {r["tool"] for r in results} == {"a", "b"}

    async def test_discover_filters_by_server(self, manager: CatalogManager) -> None:
        await manager.hydrate_server("filesystem", [_tool("read")])
        await manager.hydrate_server("search", [_tool("grep")])
        results = manager.discover("", server="search")
        assert [r["tool"] for r in results] == ["grep"]

    async def test_discover_skips_non_ready_servers(self, manager: CatalogManager) -> None:
        await manager.hydrate_server("filesystem", [_tool("read")])
        manager.mark_error("search", "boom")
        results = manager.discover("")
        assert {r["server"] for r in results} == {"filesystem"}

    async def test_discover_indexes_schema_properties(self, manager: CatalogManager) -> None:
        await manager.hydrate_server(
            "filesystem",
            [
                _tool(
                    "write_file",
                    "Writes file contents",
                    {
                        "type": "object",
                        "properties": {"path": {"type": "string"}, "text": {"type": "string"}},
                        "required": ["path", "text"],
                    },
                )
            ],
        )

        results = manager.discover("text")

        assert [r["tool"] for r in results] == ["write_file"]

    async def test_discover_returns_tool_family_metadata(self, manager: CatalogManager) -> None:
        await manager.hydrate_server(
            "filesystem",
            [
                _tool("read_file", "Read file contents"),
                _tool("read_text", "Read text contents"),
                _tool("write_file", "Write file contents"),
            ],
        )

        results = manager.discover("")

        families = {item["tool"]: item["family"] for item in results}
        assert families["read_file"] == families["read_text"]
        assert families["write_file"] != families["read_file"]

    async def test_discover_semantic_fallback_handles_close_variants(
        self, manager: CatalogManager
    ) -> None:
        await manager.hydrate_server("filesystem", [_tool("write_file", "Write file contents")])

        results = manager.discover("writer")

        assert [r["tool"] for r in results] == ["write_file"]

    async def test_rank_server_matches_ignores_zero_semantic_scores(
        self, manager: CatalogManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        await manager.hydrate_server("filesystem", [_tool("alpha"), _tool("beta")])
        entry = manager.get_entry("filesystem")

        class _StubLexicalIndex:
            @staticmethod
            def search(query: str, tools: dict[str, ToolSchema]) -> list[tuple[str, float]]:
                return [("alpha", 10.0), ("beta", 9.0)]

        monkeypatch.setitem(
            manager._lexical_indexes,
            "filesystem",
            _StubLexicalIndex(),
        )
        monkeypatch.setattr(
            manager,
            "_semantic_scores",
            lambda server, tool_names, query: {"alpha": 0.0, "beta": 0.0},
        )

        matches = manager._rank_server_matches("filesystem", entry, "query")

        assert [match.tool for match in matches] == ["alpha", "beta"]
        assert [match.score for match in matches] == [
            _reciprocal_rank_fusion_score(1, manager._cfg.retrieval),
            _reciprocal_rank_fusion_score(2, manager._cfg.retrieval),
        ]

    async def test_discover_respects_semantic_fallback_threshold_override(self) -> None:
        manager = CatalogManager(
            RegistryConfig(
                summary_ttl_s=300,
                retrieval=RetrievalConfig(semantic_fallback_threshold=0.3),
            )
        )
        manager.register_server(ServerConfig(name="filesystem"))

        await manager.hydrate_server("filesystem", [_tool("write_file", "Write file contents")])

        results = manager.discover("writer")

        assert results == []

    async def test_discover_respects_family_clustering_override(self) -> None:
        manager = CatalogManager(
            RegistryConfig(
                summary_ttl_s=300,
                retrieval=RetrievalConfig(
                    leader_cluster_threshold=0.35,
                    cluster_stopwords=("read",),
                ),
            )
        )
        manager.register_server(ServerConfig(name="filesystem"))

        await manager.hydrate_server(
            "filesystem",
            [
                _tool("read_file", "Read file contents"),
                _tool("read_text", "Read text contents"),
            ],
        )

        results = manager.discover("")

        families = {item["tool"]: item["family"] for item in results}
        assert families["read_file"] == "read_file"
        assert families["read_text"] == "read_text"
