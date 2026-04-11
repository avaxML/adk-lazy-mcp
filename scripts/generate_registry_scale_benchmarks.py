from __future__ import annotations

import asyncio
import inspect
import json
import statistics
import time
from pathlib import Path
from typing import Any

from adk_lazy_mcp import LazyMCPToolset, RegistryConfig, ServerConfig

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "docs" / "benchmarks"
DATA_PATH = BENCHMARK_DIR / "registry_corpus.json"
RESULTS_PATH = BENCHMARK_DIR / "registry_scale_results.json"
TOKENS_SVG_PATH = BENCHMARK_DIR / "registry_scale_tokens.svg"
LATENCY_SVG_PATH = BENCHMARK_DIR / "registry_scale_latency.svg"

LEVELS = (10, 20, 40, 50, 100)
CHARS_PER_TOKEN = 4
ITERATIONS = 400
SVG_WIDTH = 900
SVG_HEIGHT = 460
PLOT_LEFT = 80
PLOT_RIGHT = 30
PLOT_TOP = 50
PLOT_BOTTOM = 70
SERIES_COLORS = ("#2563eb", "#16a34a", "#f59e0b")


class BenchmarkError(RuntimeError):
    """Raised when the benchmark corpus is malformed."""


def _approx_tokens(payload: str) -> int:
    return (len(payload) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN


def _slug(value: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "_" for char in value)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def _load_corpus() -> list[dict[str, Any]]:
    data = json.loads(DATA_PATH.read_text())
    if not isinstance(data, list) or len(data) < max(LEVELS):
        raise BenchmarkError(
            f"expected at least {max(LEVELS)} benchmark entries in {DATA_PATH}, got {len(data)}"
        )
    return sorted(data, key=lambda item: item["uses"], reverse=True)


def _proxy_tool(entry: dict[str, Any]) -> dict[str, Any]:
    server_slug = _slug(entry["name"])
    return {
        "name": f"{server_slug}_query",
        "description": entry["desc"],
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": f"Query or instruction for {entry['name']}.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 5,
                    "description": "Maximum number of results to return.",
                },
            },
            "required": ["query"],
        },
    }


def _make_catalog(entries: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {_slug(entry["name"]): [_proxy_tool(entry)] for entry in entries}


def _lazy_tool_surface(toolset: LazyMCPToolset) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for fn in (
        toolset.discover_mcp_tools,
        toolset.inspect_mcp_tool,
        toolset.execute_mcp_tool,
    ):
        signature = inspect.signature(fn)
        out.append(
            {
                "name": fn.__name__,
                "description": (fn.__doc__ or "").strip(),
                "parameters": {
                    name: str(param.annotation) for name, param in signature.parameters.items()
                },
            }
        )
    return out


async def _benchmark_level(entries: list[dict[str, Any]]) -> dict[str, Any]:
    catalog = _make_catalog(entries)
    server_configs = [
        ServerConfig(name=server_name, transport="streamable_http") for server_name in catalog
    ]

    async def list_tools(server: str) -> list[dict[str, Any]]:
        return catalog[server]

    async def execute_tool(
        server: str, tool: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "isError": False,
            "content": [{"type": "text", "text": f"{server}:{tool}"}],
            "structuredContent": arguments,
        }

    toolset = LazyMCPToolset(
        server_configs,
        RegistryConfig(warm_mode="eager", max_discover_results=20),
        list_tools=list_tools,
        execute_tool=execute_tool,
    )
    try:
        await toolset.get_tools()
        lazy_surface = _lazy_tool_surface(toolset)
        naive_surface = [
            {
                "server": server_name,
                "name": tool["name"],
                "description": tool.get("description", ""),
                "inputSchema": tool.get("inputSchema", {}),
            }
            for server_name, tools in catalog.items()
            for tool in tools
        ]
        discover_result = await toolset.discover_mcp_tools(query="search", limit=20)

        lazy_json = json.dumps(lazy_surface, sort_keys=True)
        naive_json = json.dumps(naive_surface, sort_keys=True)
        discover_json = json.dumps(discover_result, sort_keys=True)

        naive_prompt_prep_ms: list[float] = []
        lazy_prompt_prep_ms: list[float] = []
        lazy_discover_ms: list[float] = []
        for _ in range(ITERATIONS):
            start = time.perf_counter_ns()
            json.dumps(naive_surface, sort_keys=True)
            naive_prompt_prep_ms.append((time.perf_counter_ns() - start) / 1_000_000)

            start = time.perf_counter_ns()
            json.dumps(lazy_surface, sort_keys=True)
            lazy_prompt_prep_ms.append((time.perf_counter_ns() - start) / 1_000_000)

            start = time.perf_counter_ns()
            await toolset.discover_mcp_tools(query="search", limit=20)
            lazy_discover_ms.append((time.perf_counter_ns() - start) / 1_000_000)

        return {
            "mcp_count": len(entries),
            "naive_tokens": _approx_tokens(naive_json),
            "lazy_tokens": _approx_tokens(lazy_json),
            "discover_tokens": _approx_tokens(discover_json),
            "naive_prompt_prep_ms_p50": round(statistics.median(naive_prompt_prep_ms), 4),
            "lazy_prompt_prep_ms_p50": round(statistics.median(lazy_prompt_prep_ms), 4),
            "lazy_discover_ms_p50": round(statistics.median(lazy_discover_ms), 4),
        }
    finally:
        await toolset.close()


async def _run_benchmarks() -> dict[str, Any]:
    corpus = _load_corpus()
    benchmark_rows = []
    for level in LEVELS:
        benchmark_rows.append(await _benchmark_level(corpus[:level]))
    return {
        "methodology": {
            "levels": list(LEVELS),
            "chars_per_token": CHARS_PER_TOKEN,
            "iterations": ITERATIONS,
            "corpus_size": len(corpus),
            "verified_entries": sum(1 for entry in corpus if entry.get("verified")),
            "community_entries": sum(1 for entry in corpus if not entry.get("verified")),
            "notes": [
                "The corpus is the 100-server manifest supplied for this task.",
                "Each registry entry is normalized into one conservative proxy tool built from its public name and description.",
                "Token counts are approximate using the same 4-chars-per-token heuristic as the existing efficiency integration test.",
            ],
        },
        "results": benchmark_rows,
    }


def _svg_line_chart(
    *,
    title: str,
    subtitle: str,
    x_values: list[int],
    series: list[tuple[str, list[float]]],
    y_label: str,
    value_formatter: str,
) -> str:
    plot_width = SVG_WIDTH - PLOT_LEFT - PLOT_RIGHT
    plot_height = SVG_HEIGHT - PLOT_TOP - PLOT_BOTTOM
    y_max = max(max(values) for _, values in series)
    y_max *= 1.1 if y_max else 1.0

    def x_pos(index: int) -> float:
        if len(x_values) == 1:
            return PLOT_LEFT + plot_width / 2
        return PLOT_LEFT + (plot_width * index / (len(x_values) - 1))

    def y_pos(value: float) -> float:
        return PLOT_TOP + plot_height - ((value / y_max) * plot_height if y_max else 0.0)

    y_ticks = 5
    tick_values = [y_max * tick / y_ticks for tick in range(y_ticks + 1)]

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}" role="img" aria-labelledby="title desc">',
        '<style>text{font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;fill:#111827} .muted{fill:#6b7280} .grid{stroke:#e5e7eb;stroke-width:1} .axis{stroke:#9ca3af;stroke-width:1.2} .legend text{font-size:13px} .title{font-size:22px;font-weight:700} .subtitle{font-size:13px} .tick{font-size:12px}</style>',
        f'<title id="title">{title}</title>',
        f'<desc id="desc">{subtitle}</desc>',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{PLOT_LEFT}" y="28" class="title">{title}</text>',
        f'<text x="{PLOT_LEFT}" y="46" class="subtitle muted">{subtitle}</text>',
    ]

    for tick_value in tick_values:
        y = y_pos(tick_value)
        lines.append(
            f'<line x1="{PLOT_LEFT}" y1="{y:.2f}" x2="{SVG_WIDTH - PLOT_RIGHT}" y2="{y:.2f}" class="grid"/>'
        )
        lines.append(
            f'<text x="{PLOT_LEFT - 10}" y="{y + 4:.2f}" class="tick muted" text-anchor="end">{tick_value:{value_formatter}}</text>'
        )

    lines.append(
        f'<line x1="{PLOT_LEFT}" y1="{PLOT_TOP}" x2="{PLOT_LEFT}" y2="{SVG_HEIGHT - PLOT_BOTTOM}" class="axis"/>'
    )
    lines.append(
        f'<line x1="{PLOT_LEFT}" y1="{SVG_HEIGHT - PLOT_BOTTOM}" x2="{SVG_WIDTH - PLOT_RIGHT}" y2="{SVG_HEIGHT - PLOT_BOTTOM}" class="axis"/>'
    )

    for index, value in enumerate(x_values):
        x = x_pos(index)
        lines.append(
            f'<line x1="{x:.2f}" y1="{SVG_HEIGHT - PLOT_BOTTOM}" x2="{x:.2f}" y2="{SVG_HEIGHT - PLOT_BOTTOM + 6}" class="axis"/>'
        )
        lines.append(
            f'<text x="{x:.2f}" y="{SVG_HEIGHT - PLOT_BOTTOM + 24}" class="tick muted" text-anchor="middle">{value}</text>'
        )

    lines.append(
        f'<text x="{SVG_WIDTH / 2:.2f}" y="{SVG_HEIGHT - 18}" class="tick muted" text-anchor="middle">MCP servers in corpus</text>'
    )
    lines.append(
        f'<text x="24" y="{SVG_HEIGHT / 2:.2f}" class="tick muted" text-anchor="middle" transform="rotate(-90 24 {SVG_HEIGHT / 2:.2f})">{y_label}</text>'
    )

    legend_x = PLOT_LEFT
    legend_y = SVG_HEIGHT - PLOT_BOTTOM + 44
    for index, (label, values) in enumerate(series):
        color = SERIES_COLORS[index % len(SERIES_COLORS)]
        points = " ".join(
            f"{x_pos(i):.2f},{y_pos(value):.2f}" for i, value in enumerate(values)
        )
        lines.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{points}"/>'
        )
        for i, value in enumerate(values):
            lines.append(
                f'<circle cx="{x_pos(i):.2f}" cy="{y_pos(value):.2f}" r="4" fill="{color}"/>'
            )
        legend_item_x = legend_x + (index * 250)
        lines.append(f'<g class="legend">')
        lines.append(
            f'<line x1="{legend_item_x}" y1="{legend_y}" x2="{legend_item_x + 26}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>'
        )
        lines.append(
            f'<text x="{legend_item_x + 34}" y="{legend_y + 4}" class="muted">{label}</text>'
        )
        lines.append('</g>')

    lines.append('</svg>')
    return "\n".join(lines)


def _write_outputs(payload: dict[str, Any]) -> None:
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2) + "\n")

    results = payload["results"]
    x_values = [row["mcp_count"] for row in results]

    tokens_svg = _svg_line_chart(
        title="Token growth across 100 MCPs",
        subtitle="Conservative proxy benchmark built from the supplied Smithery + official registry corpus.",
        x_values=x_values,
        series=[
            ("Naive upfront tool surface", [row["naive_tokens"] for row in results]),
            ("Lazy 3-tool surface", [row["lazy_tokens"] for row in results]),
            ("Lazy discover payload", [row["discover_tokens"] for row in results]),
        ],
        y_label="Approximate prompt tokens",
        value_formatter=",.0f",
    )
    TOKENS_SVG_PATH.write_text(tokens_svg)

    latency_svg = _svg_line_chart(
        title="Client-side latency across 100 MCPs",
        subtitle="p50 over 400 iterations; lazy discovery is an extra runtime lookup, while the 3-tool prompt stays flat.",
        x_values=x_values,
        series=[
            ("Naive prompt prep", [row["naive_prompt_prep_ms_p50"] for row in results]),
            ("Lazy prompt prep", [row["lazy_prompt_prep_ms_p50"] for row in results]),
            ("Lazy discover", [row["lazy_discover_ms_p50"] for row in results]),
        ],
        y_label="Milliseconds (p50)",
        value_formatter=".2f",
    )
    LATENCY_SVG_PATH.write_text(latency_svg)


def main() -> None:
    payload = asyncio.run(_run_benchmarks())
    _write_outputs(payload)


if __name__ == "__main__":
    main()
