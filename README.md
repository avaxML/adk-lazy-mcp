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
