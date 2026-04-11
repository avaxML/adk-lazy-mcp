"""Cross-reference Smithery servers with mcpservers.org attribution.

``fetch_mcps.py`` does a strict exact match when attributing servers. The reality
is that Smithery uses namespaces like ``EthanHenrickson/math-mcp`` while
mcpservers.org uses slugs like ``ethanhenrickson/math-mcp`` or only a repo name.
This script rewrites the catalog with a more lenient match (case-insensitive,
owner/repo and display-name token overlap) so that the ``sources`` field reflects
which servers also appear on mcpservers.org/official.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

CATALOG_PATH = Path(__file__).resolve().parents[1] / "data" / "mcp_catalog.json"


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def main() -> int:
    catalog = json.loads(CATALOG_PATH.read_text())
    entries = catalog.get("mcpservers_org_entries", [])

    repo_keys: set[str] = set()
    norm_keys: set[str] = set()
    for entry in entries:
        owner = (entry.get("owner") or "").lower()
        repo = (entry.get("repo") or "").lower()
        if owner and repo:
            repo_keys.add(f"{owner}/{repo}")
            norm_keys.add(_normalize(f"{owner}-{repo}"))
            norm_keys.add(_normalize(repo))
        slug = (entry.get("slug") or "").lower()
        if slug:
            repo_keys.add(slug)
            norm_keys.add(_normalize(slug))
        guess = (entry.get("qualifiedName_guess") or "").lower()
        if guess:
            repo_keys.add(guess)
            norm_keys.add(_normalize(guess))

    index = {"smithery": [], "mcpservers.org": []}
    for server in catalog.get("servers", []):
        qualified = (server.get("qualified_name") or "").lower()
        display = (server.get("display_name") or "").lower()
        candidates = {qualified, display}
        if "/" in qualified:
            candidates.add(qualified.split("/", 1)[1])
        hit = False
        if qualified in repo_keys:
            hit = True
        else:
            for candidate in candidates:
                if not candidate:
                    continue
                normalized = _normalize(candidate)
                if normalized and normalized in norm_keys:
                    hit = True
                    break
        sources = ["smithery"]
        if hit:
            sources.append("mcpservers.org")
        server["sources"] = sources
        for src in sources:
            index[src].append(server["qualified_name"])

    catalog["source_index"] = index
    catalog["source_counts"] = {k: len(v) for k, v in index.items()}
    CATALOG_PATH.write_text(json.dumps(catalog, indent=2, sort_keys=True))
    print("Updated source counts:", catalog["source_counts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
