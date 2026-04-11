# adk-lazy-mcp

`adk-lazy-mcp` v2 is a policy-aware lazy MCP toolset for ADK-style agents.

## Highlights
- Constant model-facing tool surface: `discover_mcp_tools`, `inspect_mcp_tool`, `execute_mcp_tool`.
- Configurable warm modes (`background`, `eager`, `on_demand`).
- Bounded discovery with per-server BM25 ranking, lightweight vector reranking, derived tool families, pagination hints, and unavailable-server reporting.
- Client-side schema validation before remote execution.
- Transport-aware per-server concurrency defaults and retry boundaries.
- Typed result normalization with truncation and artifact references.
- Health snapshot and lightweight telemetry primitives.

## Package layout
See `src/adk_lazy_mcp/` for runtime modules and `tests/` for baseline behavior checks.

## Development

Install the package in editable mode with the `dev` extras to pull in `ruff`,
`pytest`, `pytest-asyncio` and `pytest-cov`:

```bash
pip install -e .[dev]
```

Common tasks:

```bash
ruff check .             # lint
ruff format --check .    # formatting check
pytest                   # unit tests (default, skips integration)
pytest --cov=adk_lazy_mcp --cov-report=term-missing
```

### Environment-backed registry and retrieval settings

`RegistryConfig()` now reads environment variables via Pydantic Settings with
the `ADK_LAZY_MCP_` prefix. Existing explicit constructor arguments still work
and override environment values.

Examples:

```bash
export ADK_LAZY_MCP_WARM_MODE=on_demand
export ADK_LAZY_MCP_MAX_DISCOVER_RESULTS=50
export ADK_LAZY_MCP_RETRIEVAL__BM25_K1=1.8
export ADK_LAZY_MCP_RETRIEVAL__BM25_B=0.7
export ADK_LAZY_MCP_RETRIEVAL__RECIPROCAL_RANK_K=40
export ADK_LAZY_MCP_RETRIEVAL__SEMANTIC_RERANK_LIMIT=8
export ADK_LAZY_MCP_RETRIEVAL__SEMANTIC_FALLBACK_THRESHOLD=0.2
export ADK_LAZY_MCP_RETRIEVAL__SEMANTIC_NAME_FALLBACK_THRESHOLD=0.8
export ADK_LAZY_MCP_RETRIEVAL__LEADER_CLUSTER_THRESHOLD=0.4
export ADK_LAZY_MCP_RETRIEVAL__MIN_CLUSTER_TOKEN_LENGTH=4
export ADK_LAZY_MCP_RETRIEVAL__NAME_TERM_WEIGHT=4
export ADK_LAZY_MCP_RETRIEVAL__SCHEMA_PROPERTY_WEIGHT=3
export ADK_LAZY_MCP_RETRIEVAL__REQUIRED_FIELD_WEIGHT=3
export ADK_LAZY_MCP_RETRIEVAL__TRIGRAM_SIZE=4
export ADK_LAZY_MCP_RETRIEVAL__TRIGRAM_WEIGHT=0.25
export ADK_LAZY_MCP_RETRIEVAL__CLUSTER_STOPWORDS='["and","for","file","tool"]'
```

### Integration tests against Smithery

The `tests/integration/` suite exercises `LazyMCPToolset` against real
Smithery-hosted MCP servers and prints an efficiency report comparing the
three meta-tool surface to the naive "dump every MCP tool" approach. It is
gated on a `SMITHERY_API_KEY` environment variable and the `integration`
extra:

```bash
pip install -e .[dev,integration]
export SMITHERY_API_KEY=...       # from https://smithery.ai/account/api-keys
pytest -m integration -s tests/integration/
```

The tests are skipped cleanly when the API key or the `mcp` package is
missing, so regular unit runs stay offline.

### Continuous integration

GitHub Actions runs `ruff` and the unit-test matrix on every push/PR against
every Python version supported by both this package and Google ADK
(3.11 – 3.14). A separate integration job runs against the live Smithery
servers on every push and pull request when the `SMITHERY_API_KEY`
repository secret is available; it is `continue-on-error: true` so network
flakes and fork PRs (which cannot see the secret) never block merges.
