# adk-lazy-mcp

`adk-lazy-mcp` v2 is a policy-aware lazy MCP toolset for ADK-style agents.

## Highlights
- Constant model-facing tool surface: `discover_mcp_tools`, `inspect_mcp_tool`, `execute_mcp_tool`.
- Configurable warm modes (`background`, `eager`, `on_demand`).
- Bounded discovery with ranking, limits, pagination hints, and unavailable-server reporting.
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
