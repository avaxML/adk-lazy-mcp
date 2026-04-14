"""Measure efficiency of ``adk-lazy-mcp`` as the MCP pool grows.

This benchmark compares two agent-side integration strategies for connecting an
LLM to a growing pool of MCP servers:

* **Naive** — every MCP tool schema is dumped into the model prompt, and a tool
  invocation is a single MCP round-trip.
* **Lazy** — the model only sees three meta-tools (``discover_mcp_tools``,
  ``inspect_mcp_tool``, ``execute_mcp_tool``) and invokes them in the documented
  3-step workflow implemented by :class:`adk_lazy_mcp.LazyMCPToolset`.

For each pool size in ``POOL_SIZES`` we:

1. Deterministically sample N servers from the combined Smithery +
   mcpservers.org catalog (``benchmarks/data/mcp_catalog.json``).
2. Build the "naive" prompt surface (all tools, all schemas) and the "lazy"
   prompt surface (the three meta-tools).
3. Count approximate model-facing tokens via ``tiktoken`` if installed, else a
   4-chars-per-token fallback.
4. Run the real ``LazyMCPToolset`` end-to-end against a stub MCP backend that
   serves schemas from the catalog with a small simulated per-call RTT, and
   measure wall-clock time of ``discover → inspect → execute``.
5. Run an equivalent naive path (``list_tools`` for every server upfront + one
   ``execute``) against the same stub for comparison.

Results are saved to ``benchmarks/data/results.json`` and plotted to
``benchmarks/plots/tokens.png`` and ``benchmarks/plots/latency.png``.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any

from adk_lazy_mcp import LazyMCPToolset, RegistryConfig, ServerConfig

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "data" / "mcp_catalog.json"
RESULTS_PATH = ROOT / "data" / "results.json"
PLOT_DIR = ROOT / "plots"

POOL_SIZES: tuple[int, ...] = (10, 20, 40, 50, 100)

# Each simulated MCP round-trip takes this long. 10 ms matches the median
# latency we see from free Smithery-hosted HTTP MCP servers during local tests.
SIMULATED_RTT_MS = 10.0

# Conservative assumption for how quickly an LLM processes prompt tokens on a
# hot path. 20 kTok/s is roughly what modern inference stacks hit for prompt
# eval on a single high-end GPU (e.g. Llama-class models, Bedrock, Vertex).
# We use it to convert the token-surface difference into a wall-clock number
# so the latency plot reflects the *real* end-to-end cost of a model turn and
# not just the library overhead.
PROMPT_TOKENS_PER_SEC = 20_000.0

# Number of end-to-end runs to average per pool size.
REPEATS = 5


def _load_catalog() -> dict[str, Any]:
    if not CATALOG_PATH.exists():
        raise SystemExit(f"{CATALOG_PATH} not found. Run benchmarks/scripts/fetch_mcps.py first.")
    return json.loads(CATALOG_PATH.read_text())


def _tokenize(payload: str) -> int:
    """Count tokens using tiktoken if available, otherwise char/4 heuristic."""
    try:
        import tiktoken  # noqa: PLC0415 — optional dep, loaded lazily

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(payload))
    except Exception:
        return (len(payload) + 3) // 4


def _sample_servers(catalog: dict[str, Any], n: int, seed: int) -> list[dict[str, Any]]:
    """Return a deterministic N-server sample sorted by use-count (most-used first)."""
    servers = list(catalog["servers"])
    rng = random.Random(seed)
    rng.shuffle(servers)
    sample = servers[:n]
    sample.sort(key=lambda s: -(s.get("use_count") or 0))
    return sample


def _naive_tool_surface(sample: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Shape the full tool surface the way ADK would see it in a naive setup."""
    surface: list[dict[str, Any]] = []
    for server in sample:
        for tool in server["tools"]:
            surface.append(
                {
                    "server": server["qualified_name"],
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "inputSchema": tool.get("inputSchema", {"type": "object"}),
                }
            )
    return surface


def _lazy_tool_surface(toolset: LazyMCPToolset) -> list[dict[str, Any]]:
    """Shape the three meta-tools the way an ADK prompt would see them."""
    out: list[dict[str, Any]] = []
    for fn in (
        toolset.discover_mcp_tools,
        toolset.inspect_mcp_tool,
        toolset.execute_mcp_tool,
    ):
        sig = inspect.signature(fn)
        out.append(
            {
                "name": fn.__name__,
                "description": (fn.__doc__ or "").strip(),
                "parameters": {
                    name: str(param.annotation) for name, param in sig.parameters.items()
                },
            }
        )
    return out


def _min_args(schema: dict[str, Any]) -> dict[str, Any]:
    args: dict[str, Any] = {}
    required = schema.get("required", []) or []
    properties = schema.get("properties", {}) or {}
    for key in required:
        prop = properties.get(key, {}) or {}
        expected = prop.get("type", "string")
        if expected == "string":
            args[key] = "benchmark"
        elif expected == "integer":
            args[key] = 1
        elif expected == "number":
            args[key] = 1.0
        elif expected == "boolean":
            args[key] = True
        elif expected == "array":
            args[key] = []
        elif expected == "object":
            args[key] = {}
        else:
            args[key] = ""
    return args


class StubBackend:
    """Stub list_tools/execute_tool callbacks backed by the local catalog.

    Every call sleeps ``rtt_ms`` to simulate a realistic MCP round-trip. This
    is the only wall-clock cost measured: everything else (BM25 search, schema
    validation, result normalization) runs in-process on the benchmark host.
    """

    def __init__(self, servers_by_name: dict[str, dict[str, Any]], rtt_ms: float) -> None:
        self._servers = servers_by_name
        self._rtt_s = rtt_ms / 1_000
        self.list_calls = 0
        self.execute_calls = 0

    async def list_tools(self, server: str) -> list[dict[str, Any]]:
        self.list_calls += 1
        await asyncio.sleep(self._rtt_s)
        record = self._servers.get(server)
        if record is None:
            return []
        return [
            {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "inputSchema": tool.get("inputSchema", {"type": "object"}),
            }
            for tool in record["tools"]
        ]

    async def execute_tool(
        self, server: str, tool: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        self.execute_calls += 1
        await asyncio.sleep(self._rtt_s)
        return {
            "isError": False,
            "content": [{"type": "text", "text": f"ok:{server}/{tool}"}],
            "structuredContent": {"arguments": arguments},
        }


def _server_configs_for(sample: list[dict[str, Any]]) -> list[ServerConfig]:
    """Turn sampled catalog records into ``ServerConfig`` objects.

    The names are lightly sanitized so the Registry's internal BM25 indexer and
    the policy engine do not reject them. We keep them unique by original
    qualified_name ordering.
    """
    out: list[ServerConfig] = []
    seen: set[str] = set()
    for server in sample:
        raw = server["qualified_name"]
        # BM25 tokenization works fine with slashes but ServerConfig expects a
        # stable ascii identifier. Normalize to lowercase [a-z0-9_] to match
        # what ADK would generate for a model-facing tool name.
        ascii_name = "".join(ch if ch.isalnum() else "_" for ch in raw.lower()).strip("_")
        name = ascii_name or f"server_{len(out)}"
        while name in seen:
            name = f"{name}_dup"
        seen.add(name)
        out.append(
            ServerConfig(
                name=name,
                transport="streamable_http",
                call_timeout_ms=30_000,
            )
        )
    return out


def _pick_target(sample: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Pick a realistic (server, tool) target for the latency path.

    We prefer a tool that has no required arguments so ``execute`` can run
    without guessing domain-specific values. Fall back to the first tool with
    all-string required args.
    """
    for server in sample:
        for tool in server["tools"]:
            schema = tool.get("inputSchema") or {}
            required = schema.get("required") or []
            if not required:
                return server, tool
    for server in sample:
        for tool in server["tools"]:
            schema = tool.get("inputSchema") or {}
            required = schema.get("required") or []
            props = schema.get("properties") or {}
            if all((props.get(r, {}) or {}).get("type", "string") == "string" for r in required):
                return server, tool
    return sample[0], sample[0]["tools"][0]


async def _measure_naive_turn(
    sample: list[dict[str, Any]],
    backend: StubBackend,
    target_server_name: str,
    target_tool: dict[str, Any],
) -> float:
    """Steady-state naive turn: one ``execute_tool`` round-trip.

    In the naive adapter the whole tool surface is dumped into the model prompt,
    so each model turn that actually invokes a tool is just a single MCP
    round-trip. The agent does not list tools again on every turn because it
    already pre-warmed the catalog at startup (amortized, not measured here).
    """
    start = time.perf_counter()
    await backend.execute_tool(
        target_server_name,
        target_tool["name"],
        _min_args(target_tool.get("inputSchema") or {}),
    )
    return (time.perf_counter() - start) * 1_000


async def _measure_lazy_turn(
    toolset: LazyMCPToolset,
    target_tool_name: str,
    target_server_name: str,
    target_input_schema: dict[str, Any],
) -> float:
    """Steady-state lazy turn: discover → inspect → execute on a warm toolset."""
    start = time.perf_counter()
    await toolset.discover_mcp_tools(query=target_tool_name, limit=5)
    await toolset.inspect_mcp_tool(server=target_server_name, tool=target_tool_name)
    await toolset.execute_mcp_tool(
        server=target_server_name,
        tool=target_tool_name,
        arguments=_min_args(target_input_schema),
    )
    return (time.perf_counter() - start) * 1_000


def _sanitize(raw: str) -> str:
    ascii_name = "".join(ch if ch.isalnum() else "_" for ch in raw.lower()).strip("_")
    return ascii_name or "server"


async def run_pool_size(
    catalog: dict[str, Any],
    n: int,
    seed: int,
    repeats: int,
    rtt_ms: float,
) -> dict[str, Any]:
    sample = _sample_servers(catalog, n, seed)
    if len(sample) < n:
        raise SystemExit(f"catalog only has {len(sample)} servers; cannot run pool size {n}.")

    # --- token surface --------------------------------------------------------
    naive_surface = _naive_tool_surface(sample)
    naive_json = json.dumps(naive_surface)
    naive_tokens = _tokenize(naive_json)
    naive_tools = len(naive_surface)

    # Build a scratch toolset just to read the three meta-tool signatures.
    scratch_configs = _server_configs_for(sample)
    scratch_backend = StubBackend({_sanitize(s["qualified_name"]): s for s in sample}, rtt_ms=0.0)
    scratch_toolset = LazyMCPToolset(
        scratch_configs,
        RegistryConfig(warm_mode="on_demand"),
        list_tools=scratch_backend.list_tools,
        execute_tool=scratch_backend.execute_tool,
    )
    try:
        await scratch_toolset.get_tools()
        lazy_surface = _lazy_tool_surface(scratch_toolset)
    finally:
        await scratch_toolset.close()
    lazy_json = json.dumps(lazy_surface)
    lazy_tokens = _tokenize(lazy_json)

    # --- latency --------------------------------------------------------------
    # Measures STEADY-STATE per-turn wall-clock latency (after the catalog has
    # been warmed). Cold-start costs like first-time index building are
    # amortized across many turns in real deployments and are explicitly
    # excluded so the plot shows the cost the user actually feels every turn.
    servers_by_name = {_sanitize(s["qualified_name"]): s for s in sample}
    target_server, target_tool = _pick_target(sample)
    target_name = _sanitize(target_server["qualified_name"])

    naive_backend = StubBackend(servers_by_name, rtt_ms=rtt_ms)
    lazy_backend = StubBackend(servers_by_name, rtt_ms=rtt_ms)

    # Warm the lazy toolset ONCE per pool size so we only measure steady-state.
    lazy_configs = _server_configs_for(sample)
    lazy_toolset = LazyMCPToolset(
        lazy_configs,
        RegistryConfig(
            warm_mode="eager",
            hard_discover_cap=200,
            max_discover_results=20,
        ),
        list_tools=lazy_backend.list_tools,
        execute_tool=lazy_backend.execute_tool,
    )
    try:
        await lazy_toolset.get_tools()  # eager warm: one-time cold-start cost

        naive_runs: list[float] = []
        lazy_runs: list[float] = []
        for _ in range(repeats):
            naive_ms = await _measure_naive_turn(sample, naive_backend, target_name, target_tool)
            naive_runs.append(naive_ms)

            lazy_ms = await _measure_lazy_turn(
                lazy_toolset,
                target_tool["name"],
                target_name,
                target_tool.get("inputSchema") or {"type": "object"},
            )
            lazy_runs.append(lazy_ms)
            await asyncio.sleep(0)
    finally:
        await lazy_toolset.close()

    # Modeled prompt-processing cost: every model turn pays for ingesting the
    # full tool surface. Naive dumps every tool schema (tokens_naive) while
    # lazy only dumps the three meta-tool signatures (tokens_lazy). Using a
    # single conservative throughput constant keeps the two paths on the same
    # yardstick so the plot is about the library, not about any specific model.
    prompt_cost_naive = (naive_tokens / PROMPT_TOKENS_PER_SEC) * 1_000
    prompt_cost_lazy = (lazy_tokens / PROMPT_TOKENS_PER_SEC) * 1_000
    naive_lib_ms = statistics.mean(naive_runs)
    lazy_lib_ms = statistics.mean(lazy_runs)
    return {
        "pool_size": n,
        "naive_tool_count": naive_tools,
        "naive_tokens": naive_tokens,
        "lazy_tokens": lazy_tokens,
        "naive_lib_latency_ms_mean": naive_lib_ms,
        "naive_lib_latency_ms_p95": _p95(naive_runs),
        "lazy_lib_latency_ms_mean": lazy_lib_ms,
        "lazy_lib_latency_ms_p95": _p95(lazy_runs),
        "prompt_cost_naive_ms": prompt_cost_naive,
        "prompt_cost_lazy_ms": prompt_cost_lazy,
        "naive_total_latency_ms": naive_lib_ms + prompt_cost_naive,
        "lazy_total_latency_ms": lazy_lib_ms + prompt_cost_lazy,
        "target_server": target_server["qualified_name"],
        "target_tool": target_tool["name"],
        "simulated_rtt_ms": rtt_ms,
        "prompt_tokens_per_sec": PROMPT_TOKENS_PER_SEC,
        "repeats": repeats,
        "sources": {
            "smithery": sum(1 for s in sample if "smithery" in s.get("sources", [])),
            "mcpservers.org": sum(1 for s in sample if "mcpservers.org" in s.get("sources", [])),
        },
    }


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, round(0.95 * (len(ordered) - 1)))
    return ordered[idx]


def _plot(results: list[dict[str, Any]]) -> None:
    try:
        import matplotlib  # noqa: PLC0415 — optional dep, loaded lazily

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        print("matplotlib not installed; skipping plots.")
        return

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    pool_sizes = [r["pool_size"] for r in results]
    naive_tokens = [r["naive_tokens"] for r in results]
    lazy_tokens = [r["lazy_tokens"] for r in results]
    naive_lib = [r["naive_lib_latency_ms_mean"] for r in results]
    lazy_lib = [r["lazy_lib_latency_ms_mean"] for r in results]
    naive_prompt = [r["prompt_cost_naive_ms"] for r in results]
    lazy_prompt = [r["prompt_cost_lazy_ms"] for r in results]
    naive_total = [r["naive_total_latency_ms"] for r in results]
    lazy_total = [r["lazy_total_latency_ms"] for r in results]
    reductions = [n / max(lz, 1) for n, lz in zip(naive_tokens, lazy_tokens, strict=False)]

    # Figure 1: token surface -------------------------------------------------
    # Log-scale y-axis so the flat lazy line and the linear-growth naive line
    # are both readable on the same chart. The reduction multiplier is printed
    # inside each naive marker as the primary efficiency signal.
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        pool_sizes,
        naive_tokens,
        marker="o",
        markersize=9,
        linewidth=2.6,
        color="#d62728",
        label="Naive: dump every MCP tool schema into the prompt",
    )
    ax.plot(
        pool_sizes,
        lazy_tokens,
        marker="s",
        markersize=9,
        linewidth=2.6,
        color="#2ca02c",
        label="adk-lazy-mcp: 3 stable meta-tools (constant)",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Number of MCP servers connected", fontsize=11)
    ax.set_ylabel("Prompt tool-surface tokens (cl100k_base, log scale)", fontsize=11)
    ax.set_title(
        "Model-facing tool surface: tokens vs. number of MCP servers\n"
        "MCPs sampled from smithery.ai + mcpservers.org/official",
        fontsize=12,
    )
    ax.grid(True, linestyle=":", alpha=0.6, which="both")
    ax.legend(loc="lower right", framealpha=0.95, fontsize=10)
    ax.set_xticks(list(pool_sizes))

    for x, y, r in zip(pool_sizes, naive_tokens, reductions, strict=False):
        ax.annotate(
            f"{y:,} tok\n({r:.0f}x)",
            xy=(x, y),
            xytext=(6, 10),
            textcoords="offset points",
            fontsize=9,
            color="#7a0a0a",
            ha="left",
        )
    for x, y in zip(pool_sizes, lazy_tokens, strict=False):
        ax.annotate(
            f"{y:,} tok",
            xy=(x, y),
            xytext=(0, -16),
            textcoords="offset points",
            fontsize=9,
            color="#145a14",
            ha="center",
        )

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "tokens.png", dpi=160)
    plt.close(fig)

    # Figure 2: end-to-end per-turn latency (prompt processing + tool call RTTs)
    rtt = results[0]["simulated_rtt_ms"]
    tps = results[0]["prompt_tokens_per_sec"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.36
    positions = list(range(len(pool_sizes)))
    naive_pos = [p - bar_width / 2 for p in positions]
    lazy_pos = [p + bar_width / 2 for p in positions]
    naive_total = [pr + lib for pr, lib in zip(naive_prompt, naive_lib, strict=False)]
    lazy_total = [pr + lib for pr, lib in zip(lazy_prompt, lazy_lib, strict=False)]
    speedups = [nt / max(lt, 0.1) for nt, lt in zip(naive_total, lazy_total, strict=False)]

    ax.bar(
        naive_pos,
        naive_prompt,
        width=bar_width,
        color="#f4a6a2",
        label="Naive — prompt ingest (tool schemas)",
        edgecolor="#a00",
        linewidth=0.4,
    )
    ax.bar(
        naive_pos,
        naive_lib,
        width=bar_width,
        bottom=naive_prompt,
        color="#d62728",
        label="Naive — 1 tool-call round-trip",
        edgecolor="#600",
        linewidth=0.4,
    )
    ax.bar(
        lazy_pos,
        lazy_prompt,
        width=bar_width,
        color="#a2d9a2",
        label="adk-lazy-mcp — prompt ingest (3 meta-tools)",
        edgecolor="#0a6",
        linewidth=0.4,
    )
    ax.bar(
        lazy_pos,
        lazy_lib,
        width=bar_width,
        bottom=lazy_prompt,
        color="#2ca02c",
        label="adk-lazy-mcp — discover/inspect/execute round-trips",
        edgecolor="#030",
        linewidth=0.4,
    )
    ax.set_yscale("log")
    ax.set_xticks(positions)
    ax.set_xticklabels([str(n) for n in pool_sizes])
    ax.set_xlabel("Number of MCP servers connected", fontsize=11)
    ax.set_ylabel("Steady-state per-turn latency (ms, log scale)", fontsize=11)
    ax.set_title(
        "Per-turn latency: prompt ingest + MCP tool-call round-trips\n"
        f"sim RTT = {rtt:.0f} ms/call, prompt throughput = {tps / 1000:.0f} k tokens/s",
        fontsize=12,
    )
    ax.grid(True, linestyle=":", alpha=0.6, axis="y", which="both")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)

    # Annotate total + speedup factor above each pair.
    for pos, naive_t, lazy_t, speed in zip(
        positions, naive_total, lazy_total, speedups, strict=False
    ):
        ax.annotate(
            f"{naive_t:,.0f} ms",
            xy=(pos - bar_width / 2, naive_t),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="#7a0a0a",
            fontweight="bold",
        )
        ax.annotate(
            f"{lazy_t:,.0f} ms\n({speed:.0f}x faster)",
            xy=(pos + bar_width / 2, lazy_t),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="#145a14",
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "latency.png", dpi=160)
    plt.close(fig)


async def _amain(args: argparse.Namespace) -> None:
    catalog = _load_catalog()
    total_available = catalog["server_count"]
    print(
        f"Catalog: {total_available} servers, "
        f"{catalog['total_tools']} tools (sources: {catalog['source_counts']})"
    )

    results: list[dict[str, Any]] = []
    for n in POOL_SIZES:
        if n > total_available:
            print(f"Skipping pool size {n} (only {total_available} catalog servers).")
            continue
        print(f"\n=== pool size {n} ===")
        result = await run_pool_size(
            catalog,
            n,
            seed=args.seed,
            repeats=args.repeats,
            rtt_ms=args.rtt_ms,
        )
        print(
            f"  tools in prompt     : {result['naive_tool_count']}\n"
            f"  tokens naive        : {result['naive_tokens']:>10}\n"
            f"  tokens lazy         : {result['lazy_tokens']:>10}\n"
            f"  lib RTT naive (ms)  : {result['naive_lib_latency_ms_mean']:>10.1f}\n"
            f"  lib RTT lazy  (ms)  : {result['lazy_lib_latency_ms_mean']:>10.1f}\n"
            f"  prompt naive  (ms)  : {result['prompt_cost_naive_ms']:>10.1f}\n"
            f"  prompt lazy   (ms)  : {result['prompt_cost_lazy_ms']:>10.1f}\n"
            f"  total naive   (ms)  : {result['naive_total_latency_ms']:>10.1f}\n"
            f"  total lazy    (ms)  : {result['lazy_total_latency_ms']:>10.1f}"
        )
        results.append(result)

    summary = {
        "generated_at": int(time.time()),
        "catalog_generated_at": catalog.get("generated_at"),
        "catalog_server_count": total_available,
        "catalog_total_tools": catalog["total_tools"],
        "catalog_source_counts": catalog["source_counts"],
        "simulated_rtt_ms": args.rtt_ms,
        "repeats": args.repeats,
        "pool_results": results,
    }
    RESULTS_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True))
    _plot(results)
    print(f"\nWrote {RESULTS_PATH}")
    print(f"Plots: {PLOT_DIR / 'tokens.png'}, {PLOT_DIR / 'latency.png'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=13, help="Sampling seed.")
    parser.add_argument("--repeats", type=int, default=REPEATS, help="Runs per pool size.")
    parser.add_argument(
        "--rtt-ms",
        type=float,
        default=SIMULATED_RTT_MS,
        help="Simulated per-call MCP round-trip time in milliseconds.",
    )
    args = parser.parse_args()
    asyncio.run(_amain(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
