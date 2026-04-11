# adk-lazy-mcp

`adk-lazy-mcp` is a lazy, policy-aware MCP toolset designed for **Google ADK agents**.

Instead of exposing hundreds of raw MCP tools directly to the model, it exposes only three stable meta-tools:

1. `discover_mcp_tools`
2. `inspect_mcp_tool`
3. `execute_mcp_tool`

That keeps tool selection accurate, token usage predictable, and execution safer.

---

## The problem this library solves

When you connect many MCP servers to an LLM agent, common issues appear quickly:

- **Tool explosion**: the model sees too many tools and picks the wrong one.
- **Large prompts**: shipping every tool schema on every turn increases cost/latency.
- **Schema drift and argument mistakes**: tools fail because arguments do not match expected JSON schema.
- **Inconsistent reliability**: one flaky server can degrade end-user experience.
- **Weak guardrails**: allow/deny lists and transport policy are often ad hoc.

This is especially painful for new joiners, because they must understand ADK + MCP + several servers before they can safely ship anything.

## The solution in one paragraph

`adk-lazy-mcp` gives ADK a small and repeatable MCP workflow:

- discover only relevant tools at runtime,
- inspect one tool's schema before first execution,
- execute with client-side validation and normalized output.

Internally it adds bounded discovery, ranking, caching, per-server concurrency/timeouts, circuit-breaker style session handling, and policy checks.

---

## Architecture and workflow

### Model-facing contract (always 3 tools)

The model only ever sees:

- `discover_mcp_tools(query, server, limit)`
- `inspect_mcp_tool(server, tool, refresh)`
- `execute_mcp_tool(server, tool, arguments, timeout_ms)`

### Runtime flow

1. Your ADK agent asks `discover_mcp_tools`.
2. Agent calls `inspect_mcp_tool` for chosen tool.
3. Agent calls `execute_mcp_tool` with validated arguments.
4. Result is normalized into a stable envelope (`content`, `structured_data`, `artifact_refs`, etc.).

### Warm modes

`RegistryConfig.warm_mode` controls catalog hydration:

- `background` (default): start fast, hydrate asynchronously.
- `eager`: hydrate all servers before first use.
- `on_demand`: hydrate only when a server is first needed.

---

## Install

```bash
pip install adk-lazy-mcp
```

For local development:

```bash
pip install -e .[dev]
```

For Smithery-backed integration tests:

```bash
pip install -e .[dev,integration]
```

---

## Quick start with Google ADK

`LazyMCPToolset` is ADK-native: it follows the `BaseToolset.get_tools(readonly_context=...)` shape and returns async callables that ADK can expose to model tool-calling.

### 1) Define server config

```python
from adk_lazy_mcp import RegistryConfig, ServerConfig

servers = [
    ServerConfig(
        name="filesystem",
        transport="stdio",
        command="python",
        args=("-m", "your_mcp_server"),
        trusted=False,
        call_timeout_ms=30_000,
        allow_tools=None,
        deny_tools=("dangerous_write",),
    )
]

registry_cfg = RegistryConfig(
    warm_mode="background",
    summary_ttl_s=300,
    max_discover_results=20,
    hard_discover_cap=100,
    enable_client_validation=True,
)
```

### 2) Provide MCP adapter callbacks

You provide two async callbacks:

- `list_tools(server_name) -> list[dict]`
- `execute_tool(server_name, tool_name, arguments) -> dict`

```python
from typing import Any

async def list_tools(server: str) -> list[dict[str, Any]]:
    # bridge to your MCP client/session implementation
    ...

async def execute_tool(server: str, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    # should return MCP CallToolResult-like payload
    # e.g. {"isError": False, "content": [...], "structuredContent": {...}}
    ...
```

### 3) Build and attach toolset

```python
from adk_lazy_mcp import LazyMCPToolset

lazy_toolset = LazyMCPToolset(
    server_configs=servers,
    registry_config=registry_cfg,
    list_tools=list_tools,
    execute_tool=execute_tool,
)

# ADK will call this and expose returned functions as model tools.
tools = await lazy_toolset.get_tools(readonly_context=None)
```

### 4) Prime the model instruction

Use the built-in instruction so the model follows the 3-step workflow:

```python
system_instruction = LazyMCPToolset.default_instruction()
```

---

## What can be configured

## 1) `ServerConfig` (per MCP server)

- `name`: logical server name used by the toolset.
- `transport`: `stdio` | `streamable_http` | `sse_legacy`.
- `command`, `args`, `env`, `cwd`: process launch fields for stdio servers.
- `url`, `headers`: endpoint/auth fields for HTTP/SSE servers.
- `trusted`: if `True`, allows retry path in session execution.
- `connect_timeout_ms`: connection timeout.
- `call_timeout_ms`: default per-tool execution timeout.
- `max_concurrency`: optional per-server concurrency override.
- `allow_tools`: optional allowlist (if set, only these tools can run).
- `deny_tools`: denylist (always blocked).
- `max_inline_bytes`: truncate oversized text payloads in normalized result.

## 2) `RegistryConfig` (global behavior)

- `warm_mode`: `background` | `eager` | `on_demand`.
- `summary_ttl_s`: tool-catalog freshness window.
- `max_discover_results`: soft cap returned to model.
- `hard_discover_cap`: hard cap safety ceiling.
- `enable_client_validation`: schema-check arguments before remote call.

## 3) Policy engine

You can pass a custom `PolicyEngine` to enforce:

- HTTPS-only policy for remote transports.
- host allowlist checks.
- tool allow/deny decisions.

## 4) Telemetry sink

You can pass `Telemetry` (or your own compatible sink) to track counters and execution timings.

## 5) Environment variable interpolation helper

`resolve_env_vars` resolves strings like `${API_KEY}` or `${API_KEY:-fallback}` when building server configs from env-driven templates.

---

## Result envelope shape (execute)

`execute_mcp_tool` returns normalized output:

- `is_error`
- `content` (typed content items)
- `structured_data`
- `artifact_refs` (for image/audio or offloaded payloads)
- `truncated` (whether text was byte-truncated)

This keeps downstream ADK agent logic predictable across heterogeneous MCP servers.

---

## Development and checks

```bash
ruff check .
ruff format --check .
pytest
pytest --cov=adk_lazy_mcp --cov-report=term-missing
```

### Integration tests (Smithery)

```bash
export SMITHERY_API_KEY=...
pytest -m integration -s tests/integration/
```

Integration tests skip cleanly if `SMITHERY_API_KEY` or optional integration deps are missing.

---

## Why this helps new joiners

A new engineer only needs to learn one repeatable pattern:

1. Discover
2. Inspect
3. Execute

They do **not** need to manually wire every MCP tool into ADK prompts, and they get policy + validation + normalized results by default.

