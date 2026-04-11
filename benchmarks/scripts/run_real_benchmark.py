"""Real-world benchmark: naive vs. lazy against a live Gemini model.

``run_benchmark.py`` isolates the library overhead by modelling prompt-ingest
latency with a ``tokens / throughput`` heuristic. This script replaces the
heuristic with actual calls to the Gemini API (``google-genai``) so the
numbers reflect what you would see in production:

* **Naive turn** — every MCP tool schema is declared as a Gemini
  ``FunctionDeclaration``, the model is asked to pick the right one, and we
  measure wall-clock + API-reported token counts for that single generation.
* **Lazy turn** — only the three meta-tools (``discover``, ``inspect``,
  ``execute``) are declared; the script runs the actual 3-step workflow
  (discover → inspect → execute), issuing one Gemini call per step and
  feeding the function responses back in. Wall-clock + token counts are
  summed across the three calls.

This script does **not** need to run inside ADK. ``LazyMCPToolset`` exposes
the same three meta-tool callables that ADK would give to the model, and
we drive them directly via Gemini's native function-calling API. The MCP
round-trips themselves still go to the stub backend (configurable RTT) so
the variable under test is "prompt + LLM inference cost" — which is exactly
what ``adk-lazy-mcp`` is built to shrink.

Outputs
-------
* ``benchmarks/data/results_real.json`` — per-pool-size metrics.
* ``benchmarks/plots/tokens_real.png`` — API-reported prompt tokens per turn.
* ``benchmarks/plots/latency_real.png`` — wall-clock per turn (naive vs lazy).

Usage
-----
    export GOOGLE_API_KEY=...                       # from Google AI Studio
    python benchmarks/scripts/run_real_benchmark.py --model gemini-2.0-flash

    # Narrower sweep if you want to save API budget:
    python benchmarks/scripts/run_real_benchmark.py \\
        --model gemini-3-flash-preview --pool-sizes 10 40 100 --repeats 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import time
from pathlib import Path
from typing import Any

from adk_lazy_mcp import LazyMCPToolset, RegistryConfig, ServerConfig
from adk_lazy_mcp.config import RetrievalConfig

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "data" / "mcp_catalog.json"
RESULTS_PATH = ROOT / "data" / "results_real.json"
PLOT_DIR = ROOT / "plots"

DEFAULT_POOL_SIZES: tuple[int, ...] = (10, 20, 40, 50, 100)
DEFAULT_RTT_MS = 10.0
DEFAULT_REPEATS = 3
DEFAULT_MODEL = "gemini-2.0-flash"


def _sample_servers(catalog: dict[str, Any], n: int, seed: int) -> list[dict[str, Any]]:
    servers = list(catalog["servers"])
    rng = random.Random(seed)
    rng.shuffle(servers)
    sample = servers[:n]
    sample.sort(key=lambda s: -(s.get("use_count") or 0))
    return sample


def _sanitize(raw: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in raw.lower()).strip("_") or "server"


def _gemini_safe_name(raw: str) -> str:
    """Gemini function names must match ``[a-zA-Z0-9_.-]{1,64}``."""
    cleaned = "".join(ch if (ch.isalnum() or ch in "_.-") else "_" for ch in raw)
    if not cleaned:
        cleaned = "tool"
    return cleaned[:63]


def _min_args(schema: dict[str, Any]) -> dict[str, Any]:
    args: dict[str, Any] = {}
    for key in schema.get("required") or []:
        prop = (schema.get("properties") or {}).get(key, {}) or {}
        ty = prop.get("type", "string")
        if ty == "integer":
            args[key] = 1
        elif ty == "number":
            args[key] = 1.0
        elif ty == "boolean":
            args[key] = True
        elif ty == "array":
            args[key] = []
        elif ty == "object":
            args[key] = {}
        else:
            args[key] = "benchmark"
    return args


def _clean_schema(schema: dict[str, Any] | None) -> dict[str, Any]:
    """Drop keys Gemini's JSON-Schema parser chokes on (``$schema``, ``$id``)."""
    if not isinstance(schema, dict):
        return {"type": "object"}
    out = {
        k: v
        for k, v in schema.items()
        if k not in {"$schema", "$id", "$ref", "$defs", "definitions"}
    }
    out.setdefault("type", "object")
    # Recursively scrub nested properties/items.
    if isinstance(out.get("properties"), dict):
        out["properties"] = {k: _clean_schema(v) for k, v in out["properties"].items()}
    if isinstance(out.get("items"), dict):
        out["items"] = _clean_schema(out["items"])
    return out


# ---------------------------------------------------------------------------
# Stub MCP backend (same simulated RTT model as run_benchmark.py)
# ---------------------------------------------------------------------------


class StubBackend:
    def __init__(self, servers_by_name: dict[str, dict[str, Any]], rtt_ms: float) -> None:
        self._servers = servers_by_name
        self._rtt_s = rtt_ms / 1000.0

    async def list_tools(self, server: str) -> list[dict[str, Any]]:
        await asyncio.sleep(self._rtt_s)
        record = self._servers.get(server)
        if record is None:
            return []
        return [
            {
                "name": t["name"],
                "description": t.get("description", ""),
                "inputSchema": t.get("inputSchema", {"type": "object"}),
            }
            for t in record["tools"]
        ]

    async def execute_tool(
        self, server: str, tool: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        await asyncio.sleep(self._rtt_s)
        return {
            "isError": False,
            "content": [{"type": "text", "text": f"ok:{server}/{tool}"}],
            "structuredContent": {"arguments": arguments},
        }


# ---------------------------------------------------------------------------
# Gemini function-declaration builders
# ---------------------------------------------------------------------------


def _naive_function_declarations(sample: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One FunctionDeclaration per MCP tool. Name-collisions are rare across
    servers for small pools but we disambiguate by prefixing the server slug.
    """
    from google.genai import types  # noqa: PLC0415

    decls: list[types.FunctionDeclaration] = []
    seen: set[str] = set()
    for server in sample:
        server_slug = _sanitize(server["qualified_name"])
        for tool in server["tools"]:
            # Prefix with the server so discovery is deterministic.
            raw_name = f"{server_slug}__{tool['name']}"
            name = _gemini_safe_name(raw_name)
            if name in seen:
                continue
            seen.add(name)
            decls.append(
                types.FunctionDeclaration(
                    name=name,
                    description=(tool.get("description") or "")[:1000],
                    parameters_json_schema=_clean_schema(tool.get("inputSchema")),
                )
            )
    return decls


def _lazy_function_declarations() -> list[Any]:
    """The three meta-tools, declared so Gemini can call them."""
    from google.genai import types  # noqa: PLC0415

    return [
        types.FunctionDeclaration(
            name="discover_mcp_tools",
            description=(
                "Search the MCP catalog for tools whose name or description match a natural "
                "language query. Returns a ranked list of `{server, tool, description}` "
                "entries. Always call this first when you need to find a tool."
            ),
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language description of what you want to do.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results to return (default 5).",
                    },
                },
                "required": ["query"],
            },
        ),
        types.FunctionDeclaration(
            name="inspect_mcp_tool",
            description=(
                "Fetch the full JSON Schema and description for one tool. Call this after "
                "`discover_mcp_tools` and before `execute_mcp_tool` to learn the argument shape."
            ),
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "server": {"type": "string"},
                    "tool": {"type": "string"},
                },
                "required": ["server", "tool"],
            },
        ),
        types.FunctionDeclaration(
            name="execute_mcp_tool",
            description=(
                "Execute an MCP tool with validated arguments. Only call this after you have "
                "inspected the tool's input schema via `inspect_mcp_tool`."
            ),
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "server": {"type": "string"},
                    "tool": {"type": "string"},
                    "arguments": {"type": "object"},
                },
                "required": ["server", "tool", "arguments"],
            },
        ),
    ]


# ---------------------------------------------------------------------------
# Gemini turn measurement
# ---------------------------------------------------------------------------


def _usage_tokens(response: Any) -> tuple[int, int]:
    meta = getattr(response, "usage_metadata", None)
    if not meta:
        return 0, 0
    return (
        getattr(meta, "prompt_token_count", 0) or 0,
        getattr(meta, "candidates_token_count", 0) or 0,
    )


def _find_function_call(response: Any) -> Any | None:
    for cand in getattr(response, "candidates", []) or []:
        content = getattr(cand, "content", None)
        if content is None:
            continue
        for part in getattr(content, "parts", []) or []:
            fc = getattr(part, "function_call", None)
            if fc is not None and getattr(fc, "name", None):
                return fc
    return None


async def _gemini_naive_turn(
    client: Any,
    model: str,
    sample: list[dict[str, Any]],
    prompt: str,
    naive_decls: list[Any],
) -> dict[str, Any]:
    """Single naive turn: give Gemini all tool schemas, measure one generation."""
    from google.genai import types  # noqa: PLC0415

    config = types.GenerateContentConfig(
        tools=[types.Tool(function_declarations=naive_decls)],
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode="ANY")
        ),
        temperature=0.0,
    )
    start = time.perf_counter()
    response = await asyncio.to_thread(
        client.models.generate_content,
        model=model,
        contents=prompt,
        config=config,
    )
    wall_ms = (time.perf_counter() - start) * 1000.0
    prompt_tokens, output_tokens = _usage_tokens(response)
    fc = _find_function_call(response)
    return {
        "wall_ms": wall_ms,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "picked_tool": getattr(fc, "name", None) if fc else None,
        "api_calls": 1,
    }


async def _gemini_lazy_turn(
    client: Any,
    model: str,
    toolset: LazyMCPToolset,
    prompt: str,
    lazy_decls: list[Any],
    max_steps: int = 6,
) -> dict[str, Any]:
    """Full discover → inspect → execute loop, measured end-to-end."""
    from google.genai import types  # noqa: PLC0415

    config = types.GenerateContentConfig(
        tools=[types.Tool(function_declarations=lazy_decls)],
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode="ANY")
        ),
        temperature=0.0,
    )
    history: list[types.Content] = [
        types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
    ]
    total_wall_ms = 0.0
    prompt_tokens_total = 0
    output_tokens_total = 0
    api_calls = 0
    executed_tool: str | None = None
    for _ in range(max_steps):
        start = time.perf_counter()
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=model,
            contents=history,
            config=config,
        )
        total_wall_ms += (time.perf_counter() - start) * 1000.0
        pt, ot = _usage_tokens(response)
        prompt_tokens_total += pt
        output_tokens_total += ot
        api_calls += 1

        fc = _find_function_call(response)
        if fc is None:
            break
        args = dict(fc.args or {})
        name = fc.name
        if name == "discover_mcp_tools":
            result = await toolset.discover_mcp_tools(
                query=args.get("query", ""),
                limit=int(args.get("limit") or 5),
            )
        elif name == "inspect_mcp_tool":
            result = await toolset.inspect_mcp_tool(server=args["server"], tool=args["tool"])
        elif name == "execute_mcp_tool":
            result = await toolset.execute_mcp_tool(
                server=args["server"],
                tool=args["tool"],
                arguments=args.get("arguments") or {},
            )
            executed_tool = f"{args['server']}/{args['tool']}"
        else:
            result = {"error": f"unknown tool {name}"}

        history.append(types.Content(role="model", parts=[types.Part(function_call=fc)]))
        history.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_function_response(
                        name=name,
                        response={"result": result},
                    )
                ],
            )
        )
        if executed_tool is not None:
            break
    return {
        "wall_ms": total_wall_ms,
        "prompt_tokens": prompt_tokens_total,
        "output_tokens": output_tokens_total,
        "picked_tool": executed_tool,
        "api_calls": api_calls,
    }


# ---------------------------------------------------------------------------
# Pool-size runner
# ---------------------------------------------------------------------------


def _pick_target(sample: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Pick a realistic target tool + build a natural-language prompt for it."""
    for server in sample:
        for tool in server["tools"]:
            descr = (tool.get("description") or "").strip()
            if descr and len(descr) >= 30:
                name_words = tool["name"].replace("_", " ").replace("-", " ")
                prompt = f"I need to {descr.split('.')[0].lower()}. Use the {name_words} tool."
                return server, tool, prompt
    server = sample[0]
    tool = server["tools"][0]
    return server, tool, f"Call the {tool['name']} tool."


async def run_pool_size(
    client: Any,
    model: str,
    catalog: dict[str, Any],
    n: int,
    seed: int,
    repeats: int,
    rtt_ms: float,
) -> dict[str, Any]:
    sample = _sample_servers(catalog, n, seed)
    if len(sample) < n:
        raise SystemExit(f"catalog only has {len(sample)} servers; cannot run pool size {n}.")
    target_server, target_tool, prompt = _pick_target(sample)
    naive_decls = _naive_function_declarations(sample)
    lazy_decls = _lazy_function_declarations()

    # --- lazy toolset (warm once) -------------------------------------------
    configs = [
        ServerConfig(name=_sanitize(s["qualified_name"]), transport="streamable_http")
        for s in sample
    ]
    backend = StubBackend({_sanitize(s["qualified_name"]): s for s in sample}, rtt_ms=rtt_ms)
    toolset = LazyMCPToolset(
        configs,
        RegistryConfig(
            warm_mode="eager",
            hard_discover_cap=50,
            max_discover_results=10,
            retrieval=RetrievalConfig(global_score_weight=0.02, name_term_weight=5),
        ),
        list_tools=backend.list_tools,
        execute_tool=backend.execute_tool,
    )
    await toolset.get_tools()

    naive_runs: list[dict[str, Any]] = []
    lazy_runs: list[dict[str, Any]] = []
    try:
        for rep in range(repeats):
            print(f"  pool={n} repeat={rep + 1}/{repeats}: naive...", flush=True)
            naive_runs.append(await _gemini_naive_turn(client, model, sample, prompt, naive_decls))
            print(f"  pool={n} repeat={rep + 1}/{repeats}: lazy...", flush=True)
            lazy_runs.append(await _gemini_lazy_turn(client, model, toolset, prompt, lazy_decls))
    finally:
        await toolset.close()

    def _agg(values: list[float]) -> dict[str, float]:
        if not values:
            return {"mean": 0.0, "p50": 0.0, "p95": 0.0}
        ordered = sorted(values)
        return {
            "mean": statistics.fmean(values),
            "p50": statistics.median(values),
            "p95": ordered[max(0, round(0.95 * (len(ordered) - 1)))],
        }

    return {
        "pool_size": n,
        "target": f"{target_server['qualified_name']}/{target_tool['name']}",
        "prompt": prompt,
        "naive_tool_count": len(naive_decls),
        "naive_wall_ms": _agg([r["wall_ms"] for r in naive_runs]),
        "lazy_wall_ms": _agg([r["wall_ms"] for r in lazy_runs]),
        "naive_prompt_tokens": _agg([float(r["prompt_tokens"]) for r in naive_runs]),
        "lazy_prompt_tokens": _agg([float(r["prompt_tokens"]) for r in lazy_runs]),
        "naive_output_tokens": _agg([float(r["output_tokens"]) for r in naive_runs]),
        "lazy_output_tokens": _agg([float(r["output_tokens"]) for r in lazy_runs]),
        "lazy_api_calls": _agg([float(r["api_calls"]) for r in lazy_runs]),
        "naive_picked_tools": [r["picked_tool"] for r in naive_runs],
        "lazy_picked_tools": [r["picked_tool"] for r in lazy_runs],
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def _plot(results: list[dict[str, Any]], model: str) -> None:
    try:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        print("matplotlib not installed; skipping plots.")
        return

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    pool = [r["pool_size"] for r in results]
    naive_p = [r["naive_prompt_tokens"]["mean"] for r in results]
    lazy_p = [r["lazy_prompt_tokens"]["mean"] for r in results]
    naive_ms = [r["naive_wall_ms"]["mean"] for r in results]
    lazy_ms = [r["lazy_wall_ms"]["mean"] for r in results]

    # Tokens figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(pool, naive_p, marker="o", markersize=9, color="#d1495b", label="Naive (all schemas)")
    ax.plot(
        pool, lazy_p, marker="s", markersize=9, color="#2a9d8f", label="adk-lazy-mcp (3 meta-tools)"
    )
    ax.set_yscale("log")
    ax.set_xlabel("Number of MCP servers connected", fontsize=11)
    ax.set_ylabel(f"API-reported prompt tokens / turn ({model})", fontsize=11)
    ax.set_title(
        f"Real Gemini prompt tokens per turn — {model}\nmeasured via usage_metadata.prompt_token_count",
        fontsize=12,
    )
    for x, n, lz in zip(pool, naive_p, lazy_p, strict=False):
        ax.annotate(
            f"{n:,.0f}",
            (x, n),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=9,
            color="#7a1c2b",
            fontweight="bold",
        )
        ax.annotate(
            f"{lz:,.0f}",
            (x, lz),
            textcoords="offset points",
            xytext=(0, -14),
            ha="center",
            fontsize=9,
            color="#155a53",
            fontweight="bold",
        )
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "tokens_real.png", dpi=160)
    plt.close(fig)

    # Latency figure
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(pool))
    bar_width = 0.36
    ax.bar(
        [i - bar_width / 2 for i in x],
        naive_ms,
        width=bar_width,
        color="#d1495b",
        edgecolor="#6a0f1a",
        label="Naive — 1 Gemini call",
    )
    ax.bar(
        [i + bar_width / 2 for i in x],
        lazy_ms,
        width=bar_width,
        color="#2a9d8f",
        edgecolor="#0c4942",
        label="adk-lazy-mcp — discover→inspect→execute",
    )
    ax.set_yscale("log")
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(p) for p in pool])
    ax.set_xlabel("Number of MCP servers connected", fontsize=11)
    ax.set_ylabel("End-to-end wall-clock per turn (ms, log)", fontsize=11)
    ax.set_title(
        f"Real Gemini wall-clock per turn — {model}\n"
        f"includes prompt ingest + inference + 10 ms stub MCP RTT",
        fontsize=12,
    )
    for i, (nm, lm) in enumerate(zip(naive_ms, lazy_ms, strict=False)):
        speedup = nm / max(lm, 0.1)
        ax.annotate(
            f"{nm:,.0f} ms",
            xy=(i - bar_width / 2, nm),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="#6a0f1a",
            fontweight="bold",
        )
        ax.annotate(
            f"{lm:,.0f} ms\n({speedup:.1f}x faster)",
            xy=(i + bar_width / 2, lm),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="#0c4942",
            fontweight="bold",
        )
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "latency_real.png", dpi=160)
    plt.close(fig)
    print(f"Wrote {PLOT_DIR / 'tokens_real.png'}")
    print(f"Wrote {PLOT_DIR / 'latency_real.png'}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


async def _amain(args: argparse.Namespace) -> None:
    if "GOOGLE_API_KEY" not in os.environ and "GEMINI_API_KEY" not in os.environ:
        raise SystemExit("Set GOOGLE_API_KEY (or GEMINI_API_KEY) before running this benchmark.")
    if not CATALOG_PATH.exists():
        raise SystemExit(f"{CATALOG_PATH} missing — run benchmarks/scripts/fetch_mcps.py first.")

    from google import genai  # noqa: PLC0415 — imported lazily so the rest of
    # the repo still imports without google-genai installed.

    client = genai.Client()
    catalog = json.loads(CATALOG_PATH.read_text())

    results: list[dict[str, Any]] = []
    for n in args.pool_sizes:
        print(f"\n=== pool size {n} ===")
        try:
            result = await run_pool_size(
                client,
                model=args.model,
                catalog=catalog,
                n=n,
                seed=args.seed,
                repeats=args.repeats,
                rtt_ms=args.rtt_ms,
            )
        except Exception as exc:
            print(f"pool {n} failed: {exc}")
            continue
        print(
            f"  naive: {result['naive_wall_ms']['mean']:.0f} ms, "
            f"{result['naive_prompt_tokens']['mean']:.0f} prompt tok, "
            f"tool={result['naive_picked_tools']}"
        )
        print(
            f"  lazy : {result['lazy_wall_ms']['mean']:.0f} ms, "
            f"{result['lazy_prompt_tokens']['mean']:.0f} prompt tok "
            f"({int(result['lazy_api_calls']['mean'])} API calls), "
            f"tool={result['lazy_picked_tools']}"
        )
        results.append(result)

    doc = {
        "generated_at": int(time.time()),
        "model": args.model,
        "repeats": args.repeats,
        "simulated_rtt_ms": args.rtt_ms,
        "pool_results": results,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(doc, indent=2, sort_keys=True))
    print(f"\nWrote {RESULTS_PATH}")
    if results:
        _plot(results, args.model)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Gemini model ID (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--pool-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_POOL_SIZES),
        help="Pool sizes to sweep (default: 10 20 40 50 100).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help="Runs per pool size to average (default: 3).",
    )
    parser.add_argument(
        "--rtt-ms",
        type=float,
        default=DEFAULT_RTT_MS,
        help="Simulated MCP RTT for the stub backend (default: 10 ms).",
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    asyncio.run(_amain(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
