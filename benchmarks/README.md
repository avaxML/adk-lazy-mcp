# benchmarks

Reproducible benchmarks for `adk-lazy-mcp`, measuring three axes on a real MCP
catalog:

1. **Tokens** — model-facing prompt surface as the number of connected MCP
   servers grows (10 → 100).
2. **Latency** — steady-state per-turn wall-clock time in the naive vs. lazy
   integration.
3. **Discovery accuracy** — `recall@1`, `recall@5`, `recall@10`, and `MRR@10`
   of `discover_mcp_tools` against an IDF-weighted golden query set, plus a
   small grid search that tunes `RetrievalConfig` for the catalog.

Plots are in [`plots/`](./plots/) and are referenced from the top-level
[README.md](../README.md#benchmarks).

## Catalog sources

All measurements run against a 300-server / 4,314-tool catalog pulled from:

- [smithery.ai registry](https://smithery.ai/docs/concepts/registry_search_servers)
  — the `/servers` endpoint (paginated), plus a per-server `/servers/{qualified_name}`
  fetch for tool schemas. Requires a Smithery bearer token.
- [mcpservers.org/official](https://mcpservers.org/official) — scraped as HTML
  for server slugs, then cross-referenced against Smithery entries so the
  attribution (`sources`) column in the catalog reflects the union.

The raw catalog dump lives at [`data/mcp_catalog.json`](./data/mcp_catalog.json)
so benchmarks are deterministic across re-runs (Smithery drift can't change
historical numbers).

## Running everything end-to-end

```bash
pip install -e .[dev]
pip install tiktoken matplotlib httpx  # for tokenizing + plotting

# 1. Pull the full catalog from Smithery (writes data/mcp_catalog.json).
#    You need a Smithery bearer token — see https://smithery.ai/account/api-keys
export SMITHERY_API_KEY=...
python benchmarks/scripts/fetch_mcps.py --token "$SMITHERY_API_KEY"

# 2. Generate the IDF-weighted golden query set (writes data/golden_queries.json).
python benchmarks/scripts/generate_golden_set.py --max 500

# 3. Measure tokens + latency at pool sizes 10, 20, 40, 50, 100.
#    Writes data/results.json, plots/tokens.png, plots/latency.png.
python benchmarks/scripts/run_benchmark.py

# 4. Measure discovery accuracy and grid-search RetrievalConfig.
#    Writes data/accuracy.json, data/tuned_retrieval_config.env, plots/accuracy.png.
python benchmarks/scripts/tune_discovery.py --catalog-size 60 --max-queries 150
```

All scripts are idempotent and keep their output under `benchmarks/data/` and
`benchmarks/plots/`.

## Scripts

| Script | Purpose |
|--------|---------|
| [`scripts/fetch_mcps.py`](./scripts/fetch_mcps.py) | Pulls MCP servers + tool schemas from Smithery, scrapes mcpservers.org for cross-attribution, writes `data/mcp_catalog.json`. |
| [`scripts/reattribute_sources.py`](./scripts/reattribute_sources.py) | One-off post-processor that re-runs the normalized cross-source matcher if you want to bump mcpservers.org overlap on an existing dump. |
| [`scripts/generate_golden_set.py`](./scripts/generate_golden_set.py) | Builds `data/golden_queries.json` — one natural-language query per tool, using IDF-weighted distinctive terms from the tool name + description. |
| [`scripts/run_benchmark.py`](./scripts/run_benchmark.py) | Main efficiency benchmark. Counts prompt tokens (`cl100k_base`) and measures steady-state per-turn latency for naive vs. lazy integration over a stub MCP backend with simulated RTT. Writes tokens/latency plots. |
| [`scripts/tune_discovery.py`](./scripts/tune_discovery.py) | Discovery accuracy grid search over `RetrievalConfig`. Computes `recall@k`, `MRR@10`, and writes `data/tuned_retrieval_config.env` with drop-in env vars. |
| [`scripts/debug_discovery.py`](./scripts/debug_discovery.py) | Quick debug harness — prints the top-N discovery results for a handful of golden queries, marking the expected hit. Not part of the benchmark output. |

## Tuned retrieval config

The tuner picks the winning config and writes it to
[`data/tuned_retrieval_config.env`](./data/tuned_retrieval_config.env). Drop it
into your environment to reproduce the accuracy gain without changing code:

```bash
source benchmarks/data/tuned_retrieval_config.env
```

The single biggest win comes from a non-zero `GLOBAL_SCORE_WEIGHT`, which acts
as a cross-server tie-breaker on top of per-server Reciprocal Rank Fusion.
Without it, every server's top match shares the same fused score and the final
merge collapses to alphabetical order — which is why stock `RetrievalConfig`
was stuck at ~10% recall@1 on the 60-server benchmark before this knob was
added.
