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
from dataclasses import dataclass
from typing import Any

import pytest

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

# Small, free, widely-available servers chosen so the test suite is cheap
# and deterministic. Both are well-maintained by Smithery and require no
# per-user auth beyond the registry API key.
DEFAULT_SERVERS: tuple[tuple[str, str], ...] = (
    ("sequential_thinking", "@smithery-ai/server-sequential-thinking"),
    ("fetch", "@smithery-ai/fetch"),
)


def _server_url(reference: str, config: dict[str, Any] | None = None) -> str:
    api_key = os.environ["SMITHERY_API_KEY"]
    params = [f"api_key={api_key}"]
    if config:
        cfg_b64 = base64.b64encode(json.dumps(config).encode("utf-8")).decode("ascii")
        params.append(f"config={cfg_b64}")
    return f"{SMITHERY_BASE}/{reference}/mcp?{'&'.join(params)}"


@dataclass
class SmitheryClient:
    """Adapter exposing the callback shape ``LazyMCPToolset`` expects.

    The real MCP SDK speaks in terms of ``ClientSession`` objects. We keep one
    open per logical "server name" and wrap the SDK calls so they look like
    ``async (server) -> list[tool]`` / ``async (server, tool, args) -> result``
    to the lazy toolset.
    """

    sessions: dict[str, ClientSession]
    references: dict[str, str]

    async def list_tools(self, server: str) -> list[dict[str, Any]]:
        response = await self.sessions[server].list_tools()
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
        result = await self.sessions[server].call_tool(tool, arguments=arguments or {})
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


@contextlib.asynccontextmanager
async def _open_session(reference: str) -> AsyncIterator[ClientSession]:
    url = _server_url(reference)
    async with streamablehttp_client(url) as (read, write, _close):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


@pytest.fixture
async def smithery_client() -> AsyncIterator[SmitheryClient]:
    """Open live sessions to the default Smithery MCPs used by integration tests."""
    stack = contextlib.AsyncExitStack()
    sessions: dict[str, ClientSession] = {}
    references: dict[str, str] = {}
    try:
        for name, reference in DEFAULT_SERVERS:
            session = await stack.enter_async_context(_open_session(reference))
            sessions[name] = session
            references[name] = reference
        yield SmitheryClient(sessions=sessions, references=references)
    finally:
        await stack.aclose()


@pytest.fixture
async def adk_callbacks(
    smithery_client: SmitheryClient,
) -> tuple[
    Callable[[str], Awaitable[list[dict[str, Any]]]],
    Callable[[str, str, dict[str, Any]], Awaitable[dict[str, Any]]],
]:
    """Return the ``(list_tools, execute_tool)`` pair wired to Smithery."""
    return smithery_client.list_tools, smithery_client.execute_tool
