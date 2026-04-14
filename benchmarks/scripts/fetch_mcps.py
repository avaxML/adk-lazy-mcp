"""Fetch MCP server metadata + tool schemas from Smithery and mcpservers.org.

Primary source is the Smithery registry (https://api.smithery.ai), which exposes
both a paginated search endpoint and a per-server details endpoint that returns
the full tool list with JSON schemas. We grab as many public deployed servers as
we reasonably can. Secondary source is https://mcpservers.org/official which we
scrape only for attribution purposes — it does not expose tool schemas, so any
overlap with Smithery is credited to both sources and any non-Smithery entries
are recorded as "name-only" references.

Output: ``benchmarks/data/mcp_catalog.json`` — a single document containing every
server we could fetch a tool schema for, plus a ``sources`` list noting where each
server was discovered.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

SMITHERY_API = "https://api.smithery.ai"
MCPSERVERS_ORG = "https://mcpservers.org"
DEFAULT_TOKEN = "f51ff318-34b5-44e6-8abd-581cf4c494f8"
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "mcp_catalog.json"
USER_AGENT = "adk-lazy-mcp-benchmarks/1.0 (+https://github.com/avaxML/adk-lazy-mcp)"


def _http_get(url: str, headers: dict[str, str] | None = None, retries: int = 3) -> bytes:
    """GET ``url`` with simple retry/backoff for transient failures."""
    hdrs = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if headers:
        hdrs.update(headers)
    request = Request(url, headers=hdrs)
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urlopen(request, timeout=30) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            time.sleep(1.5**attempt)
    raise RuntimeError(f"failed to fetch {url}: {last_error}")


def _smithery_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def list_smithery_servers(
    token: str,
    *,
    max_pages: int,
    page_size: int = 100,
) -> list[dict[str, Any]]:
    """Return a flat list of Smithery server summaries (most-used first)."""
    out: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    headers = _smithery_headers(token)
    for page in range(1, max_pages + 1):
        query = urlencode({"page": page, "pageSize": page_size})
        url = f"{SMITHERY_API}/servers?{query}"
        try:
            payload = json.loads(_http_get(url, headers=headers))
        except Exception as exc:
            print(f"[smithery] page {page} failed: {exc}", file=sys.stderr)
            break
        servers = payload.get("servers", []) or []
        if not servers:
            break
        for server in servers:
            sid = server.get("id")
            if sid and sid not in seen_ids:
                seen_ids.add(sid)
                out.append(server)
        total_pages = (payload.get("pagination") or {}).get("totalPages")
        print(
            f"[smithery] page {page}/{total_pages}: +{len(servers)} servers "
            f"(running total {len(out)})"
        )
        if total_pages and page >= total_pages:
            break
    return out


def fetch_smithery_server_details(qualified_name: str, token: str) -> dict[str, Any] | None:
    """Fetch full server details including tools. Returns None on failure."""
    url = f"{SMITHERY_API}/servers/{qualified_name}"
    try:
        payload = json.loads(_http_get(url, headers=_smithery_headers(token)))
    except Exception as exc:
        print(f"[smithery] details {qualified_name} failed: {exc}", file=sys.stderr)
        return None
    return payload


def fetch_mcpservers_org_official(max_pages: int = 14) -> list[dict[str, Any]]:
    """Scrape mcpservers.org/official for server names + repo links.

    The site is a React SPA that server-renders a list of cards; we pull the
    plain HTML fallback and do a permissive regex-free walk. We only care about
    the set of server slugs so we can cross-reference with Smithery later.
    """
    import re  # noqa: PLC0415 — scoped to the scraping helper

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for page in range(1, max_pages + 1):
        url = f"{MCPSERVERS_ORG}/official?page={page}&sort=name"
        try:
            html = _http_get(url).decode("utf-8", errors="replace")
        except Exception as exc:
            print(f"[mcpservers.org] page {page} failed: {exc}", file=sys.stderr)
            continue
        # Match both github.com/owner/repo and /servers/<slug> style references.
        gh_matches = re.findall(r'"(https?://github\.com/[^"]+)"', html)
        slug_matches = re.findall(r'"/servers/([a-zA-Z0-9_\-./]+)"', html)
        discovered = 0
        for gh_url in gh_matches:
            key = gh_url.lower()
            if key in seen:
                continue
            seen.add(key)
            parts = gh_url.rstrip("/").split("/")
            if len(parts) >= 5:
                owner, repo = parts[-2], parts[-1]
                out.append(
                    {
                        "source": "mcpservers.org",
                        "owner": owner,
                        "repo": repo,
                        "url": gh_url,
                        "qualifiedName_guess": f"{owner}/{repo}",
                    }
                )
                discovered += 1
        for slug in slug_matches:
            key = f"slug:{slug.lower()}"
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "source": "mcpservers.org",
                    "slug": slug,
                    "url": f"{MCPSERVERS_ORG}/servers/{slug}",
                    "qualifiedName_guess": slug,
                }
            )
            discovered += 1
        print(f"[mcpservers.org] page {page}: +{discovered} entries")
        if discovered == 0 and page > 1:
            # Likely past the last populated page.
            break
    return out


def _coerce_tools(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for tool in raw:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        if not isinstance(name, str):
            continue
        schema = tool.get("inputSchema")
        if not isinstance(schema, dict):
            schema = {"type": "object"}
        out.append(
            {
                "name": name,
                "description": tool.get("description") or "",
                "inputSchema": schema,
            }
        )
    return out


def build_catalog(
    *,
    token: str,
    max_smithery_pages: int,
    max_detail_fetches: int,
    min_tools: int = 1,
) -> dict[str, Any]:
    print(f"Fetching up to {max_smithery_pages} pages of Smithery servers...")
    summaries = list_smithery_servers(token, max_pages=max_smithery_pages)
    print(f"Smithery summaries fetched: {len(summaries)}")

    # Prefer deployed remote servers first (they have schemas populated server-side).
    summaries.sort(
        key=lambda s: (
            0 if s.get("isDeployed") else 1,
            0 if s.get("remote") else 1,
            -(s.get("useCount") or 0),
        )
    )

    print("Fetching mcpservers.org/official server listing for attribution...")
    try:
        mcpservers_entries = fetch_mcpservers_org_official()
    except Exception as exc:
        print(f"[mcpservers.org] failed: {exc}", file=sys.stderr)
        mcpservers_entries = []
    mcpservers_lookup = {
        entry.get("qualifiedName_guess", "").lower(): entry for entry in mcpservers_entries
    }
    mcpservers_repo_lookup = {
        f"{entry.get('owner', '').lower()}/{entry.get('repo', '').lower()}": entry
        for entry in mcpservers_entries
        if entry.get("owner") and entry.get("repo")
    }

    servers: list[dict[str, Any]] = []
    source_index: dict[str, list[str]] = {"smithery": [], "mcpservers.org": []}
    attempts = 0
    for summary in summaries:
        if len(servers) >= max_detail_fetches:
            break
        attempts += 1
        qualified = summary.get("qualifiedName") or ""
        if not qualified:
            continue
        details = fetch_smithery_server_details(qualified, token)
        if not details:
            continue
        tools = _coerce_tools(details.get("tools"))
        if len(tools) < min_tools:
            continue
        # Attribution: always Smithery; also add mcpservers.org if we matched it.
        sources = ["smithery"]
        lookup_keys = {
            qualified.lower(),
            (summary.get("slug") or "").lower(),
        }
        repo_key = qualified.lower()
        if repo_key in mcpservers_repo_lookup or any(
            k and k in mcpservers_lookup for k in lookup_keys
        ):
            sources.append("mcpservers.org")
        server_record = {
            "qualified_name": qualified,
            "display_name": summary.get("displayName") or qualified,
            "description": summary.get("description") or details.get("description") or "",
            "use_count": summary.get("useCount"),
            "verified": summary.get("verified"),
            "is_deployed": summary.get("isDeployed"),
            "remote": summary.get("remote"),
            "tool_count": len(tools),
            "tools": tools,
            "sources": sources,
        }
        servers.append(server_record)
        for src in sources:
            source_index[src].append(qualified)
        if len(servers) % 25 == 0:
            print(f"[details] progress {len(servers)}/{max_detail_fetches} (attempts={attempts})")

    catalog = {
        "generated_at": int(time.time()),
        "smithery_api": SMITHERY_API,
        "mcpservers_org": MCPSERVERS_ORG,
        "server_count": len(servers),
        "total_tools": sum(s["tool_count"] for s in servers),
        "source_counts": {k: len(v) for k, v in source_index.items()},
        "source_index": source_index,
        "mcpservers_org_entries": mcpservers_entries,
        "servers": servers,
    }
    return catalog


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--token",
        default=os.environ.get("SMITHERY_API_KEY", DEFAULT_TOKEN),
        help="Smithery bearer token.",
    )
    parser.add_argument(
        "--max-smithery-pages",
        type=int,
        default=8,
        help="Number of Smithery search pages (pageSize=100) to enumerate.",
    )
    parser.add_argument(
        "--max-detail-fetches",
        type=int,
        default=200,
        help="Maximum server detail documents to retain (after filtering).",
    )
    parser.add_argument("--output", default=str(OUTPUT_PATH), help="Catalog output path.")
    args = parser.parse_args()

    catalog = build_catalog(
        token=args.token,
        max_smithery_pages=args.max_smithery_pages,
        max_detail_fetches=args.max_detail_fetches,
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(catalog, indent=2, sort_keys=True))
    print(f"Wrote {len(catalog['servers'])} servers ({catalog['total_tools']} tools) to {out_path}")
    print(f"Source counts: {catalog['source_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
