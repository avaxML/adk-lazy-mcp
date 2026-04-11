"""Integration-test fixtures for running against live Smithery-hosted MCP servers.

These tests require:

* ``SMITHERY_API_KEY`` in the environment (obtain one at
  https://smithery.ai/account/api-keys).
* The optional ``mcp`` extra installed: ``pip install -e .[dev,integration]``.

If either prerequisite is missing the whole ``tests/integration`` tree is
skipped cleanly so local + CI runs without the secret stay green.
"""

from __future__ import annotations

import base64
import contextlib
import json
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import pytest
from pydantic import BaseModel

# Skip this whole folder unless both prerequisites are met.
mcp = pytest.importorskip("mcp", reason="the 'mcp' package is required for integration tests")

if not os.environ.get("SMITHERY_API_KEY"):
    pytest.skip(
        "SMITHERY_API_KEY is not set - skipping Smithery integration tests",
        allow_module_level=True,
    )

# ``mcp`` is available: import the bits we need. Done lazily so the top-level
# skip above runs before we touch any optional submodules.
from mcp import ClientSession  # noqa: E402
from mcp.client.streamable_http import streamablehttp_client  # noqa: E402

SMITHERY_BASE = "https://server.smithery.ai"

# Two free, no-config, remotely-deployed Smithery servers chosen to make
# the efficiency benchmark meaningful and CI-safe:
#
# * ``@EthanHenrickson/math-mcp`` — 22 pure-function arithmetic / stats /
#   trig tools. No I/O, no state, no API keys. Gives a large schema-size
#   delta against the three lazy meta-tools.
# * ``@smithery-ai/server-sequential-thinking`` — single-tool first-party
#   server (``sequentialthinking``). Stable baseline, zero-config, side-
#   effect-free.
#
# Neither server requires a ``config=<base64>`` query parameter — only the
# ``SMITHERY_API_KEY`` registry key is needed.
DEFAULT_SERVERS: tuple[tuple[str, str], ...] = (
    ("math", "@EthanHenrickson/math-mcp"),
    ("sequential_thinking", "@smithery-ai/server-sequential-thinking"),
)


def _server_url(reference: str, config: dict[str, Any] | None = None) -> str:
    api_key = os.environ["SMITHERY_API_KEY"]
    params = [f"api_key={api_key}"]
    if config:
        cfg_b64 = base64.b64encode(json.dumps(config).encode("utf-8")).decode("ascii")
        params.append(f"config={cfg_b64}")
    return f"{SMITHERY_BASE}/{reference}/mcp?{'&'.join(params)}"


@contextlib.asynccontextmanager
async def _open_session(reference: str) -> AsyncIterator[ClientSession]:
    url = _server_url(reference)
    async with streamablehttp_client(url) as (read, write, _close):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


class SmitheryClient(BaseModel):
    """Adapter exposing the callback shape ``LazyMCPToolset`` expects.

    Each call to ``list_tools`` / ``execute_tool`` opens a fresh MCP session
    against the Smithery-hosted server. Opening per call is a little slower
    than holding a session across the whole test, but it keeps every anyio
    cancel scope entered and exited in the same task — which is the only
    reliable way to use ``streamablehttp_client`` under ``pytest-asyncio``.
    That plugin runs fixture setup and finalization in *different* tasks
    (``runner.run(async_finalizer(), ...)``), so a long-lived session opened
    in a fixture would raise ``RuntimeError("Attempted to exit cancel scope
    in a different task than it was entered in")`` during teardown.
    """

    references: dict[str, str]

    async def list_tools(self, server: str) -> list[dict[str, Any]]:
        async with _open_session(self.references[server]) as session:
            response = await session.list_tools()
            out: list[dict[str, Any]] = []
            for tool in response.tools:
                out.append(
                    {
                        "name": tool.name,
                        "description": tool.description or "",
                        "inputSchema": tool.inputSchema or {"type": "object"},
                    }
                )
            return out

    async def execute_tool(
        self, server: str, tool: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        async with _open_session(self.references[server]) as session:
            result = await session.call_tool(tool, arguments=arguments or {})
            content: list[dict[str, Any]] = []
            for item in result.content or []:
                item_type = getattr(item, "type", "text")
                if item_type == "text":
                    content.append({"type": "text", "text": getattr(item, "text", "")})
                elif item_type == "image":
                    content.append(
                        {
                            "type": "image",
                            "mimeType": getattr(item, "mimeType", None),
                            "uri": getattr(item, "uri", None),
                        }
                    )
                else:
                    content.append({"type": item_type})
            return {
                "isError": bool(getattr(result, "isError", False)),
                "content": content,
                "structuredContent": getattr(result, "structuredContent", None),
            }


@pytest.fixture
def smithery_client() -> SmitheryClient:
    """Return a session-opening adapter for the default Smithery MCPs.

    The fixture itself is synchronous: all network I/O happens inside the
    tests' own task when they invoke the callbacks. That is deliberate — see
    the ``SmitheryClient`` docstring for the pytest-asyncio task-boundary
    rationale.
    """
    return SmitheryClient(references=dict(DEFAULT_SERVERS))


@pytest.fixture
def adk_callbacks(
    smithery_client: SmitheryClient,
) -> tuple[
    Callable[[str], Awaitable[list[dict[str, Any]]]],
    Callable[[str, str, dict[str, Any]], Awaitable[dict[str, Any]]],
]:
    """Return the ``(list_tools, execute_tool)`` pair wired to Smithery."""
    return smithery_client.list_tools, smithery_client.execute_tool
