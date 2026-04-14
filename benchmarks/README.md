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

## Real-world Gemini benchmark (optional)

`scripts/run_benchmark.py` isolates the library overhead with a
`tokens / throughput` heuristic so it's fully reproducible without any
network calls. If you want end-to-end numbers against a live model,
`scripts/run_real_benchmark.py` replaces that heuristic with actual calls to
the Gemini API via `google-genai`:

* **Naive turn** — every MCP tool schema in the sampled pool is declared as
  a Gemini `FunctionDeclaration`, the model is asked to pick the right one,
  and we record wall-clock + `usage_metadata.prompt_token_count` for that
  single generation.
* **Lazy turn** — only the three meta-tools (`discover_mcp_tools`,
  `inspect_mcp_tool`, `execute_mcp_tool`) are declared; the script drives the
  real 3-step `discover → inspect → execute` loop, making one Gemini call
  per step and feeding the function responses back in. Wall-clock + tokens
  are summed across the three calls. The MCP round-trips go to the same
  stub backend as `run_benchmark.py` with configurable RTT so the variable
  under test is "prompt surface + LLM inference cost" — exactly what
  `adk-lazy-mcp` is built to shrink.

```bash
pip install google-genai                             # extra dep, not in [dev]
export GOOGLE_API_KEY=...                            # or GEMINI_API_KEY
python benchmarks/scripts/run_real_benchmark.py \
    --model gemini-2.0-flash \
    --pool-sizes 10 40 100 \
    --repeats 3
```

Outputs:

* `data/results_real.json` — per-pool-size metrics (naive + lazy wall-clock,
  prompt/output token counts, sampled tool used as the target).
* `plots/tokens_real.png` — API-reported prompt tokens per turn.
* `plots/latency_real.png` — wall-clock per turn (naive vs. lazy).

Notes:

* `--model` accepts any Gemini model with function calling (e.g.
  `gemini-2.0-flash`, `gemini-2.5-flash`, `gemini-3-flash-preview`).
* `--repeats` controls how many turns get averaged per pool size — bump it
  for tighter error bars, lower it to save API budget.
* The script is deterministic for a given `--seed`: the same tool is sampled
  as the target across the naive and lazy paths so both integrations are
  measured on the same work.

## Scripts

| Script | Purpose |
|--------|---------|
| [`scripts/fetch_mcps.py`](./scripts/fetch_mcps.py) | Pulls MCP servers + tool schemas from Smithery, scrapes mcpservers.org for cross-attribution, writes `data/mcp_catalog.json`. |
| [`scripts/reattribute_sources.py`](./scripts/reattribute_sources.py) | One-off post-processor that re-runs the normalized cross-source matcher if you want to bump mcpservers.org overlap on an existing dump. |
| [`scripts/generate_golden_set.py`](./scripts/generate_golden_set.py) | Builds `data/golden_queries.json` — one natural-language query per tool, using IDF-weighted distinctive terms from the tool name + description. |
| [`scripts/run_benchmark.py`](./scripts/run_benchmark.py) | Main efficiency benchmark. Counts prompt tokens (`cl100k_base`) and measures steady-state per-turn latency for naive vs. lazy integration over a stub MCP backend with simulated RTT. Writes tokens/latency plots. |
| [`scripts/run_real_benchmark.py`](./scripts/run_real_benchmark.py) | Real-world variant of `run_benchmark.py`. Drives `google-genai` against a live Gemini model, measures API-reported prompt tokens + wall-clock for naive vs. lazy turns. Requires `GOOGLE_API_KEY`. |
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
