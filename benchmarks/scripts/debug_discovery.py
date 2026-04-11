"""Quick debug harness for discovery queries. Not part of the benchmark output."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from adk_lazy_mcp import LazyMCPToolset, RegistryConfig, ServerConfig
from adk_lazy_mcp.config import RetrievalConfig

ROOT = Path(__file__).resolve().parents[1]


def _sanitize(raw: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in raw.lower()).strip("_") or "server"


async def main() -> None:
    catalog = json.loads((ROOT / "data" / "mcp_catalog.json").read_text())
    golden = json.loads((ROOT / "data" / "golden_queries.json").read_text())["queries"]
    sample = catalog["servers"][:60]
    names = {s["qualified_name"] for s in sample}
    gold = [g for g in golden if g["server"] in names][:10]

    backend_servers = {_sanitize(s["qualified_name"]): s for s in sample}

    async def list_tools(server: str) -> list[dict]:
        r = backend_servers.get(server)
        if not r:
            return []
        return [
            {
                "name": t["name"],
                "description": t.get("description", ""),
                "inputSchema": t.get("inputSchema", {"type": "object"}),
            }
            for t in r["tools"]
        ]

    async def execute_tool(*args, **kwargs):
        return {"isError": False, "content": [], "structuredContent": None}

    configs = [
        ServerConfig(name=_sanitize(s["qualified_name"]), transport="streamable_http")
        for s in sample
    ]
    toolset = LazyMCPToolset(
        configs,
        RegistryConfig(
            warm_mode="eager",
            hard_discover_cap=200,
            max_discover_results=100,
            retrieval=RetrievalConfig(global_score_weight=0.05),
        ),
        list_tools=list_tools,
        execute_tool=execute_tool,
    )
    await toolset.get_tools()
    try:
        for entry in gold:
            expected_server = _sanitize(entry["server"])
            print(f"\nquery={entry['query']!r}")
            print(f"expect: {expected_server}/{entry['tool']}")
            print(f"descr : {entry['description'][:100]}")
            result = await toolset.discover_mcp_tools(query=entry["query"], limit=50)
            for i, t in enumerate(result.get("tools") or [], start=1):
                marker = (
                    " <-- HIT"
                    if t.get("server") == expected_server and t.get("tool") == entry["tool"]
                    else ""
                )
                print(f"  {i:>2}. {t.get('server')}/{t.get('tool')}{marker}")
    finally:
        await toolset.close()


if __name__ == "__main__":
    asyncio.run(main())
