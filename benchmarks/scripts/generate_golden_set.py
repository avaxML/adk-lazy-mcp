"""Generate a golden (query, expected tool) set from the fetched MCP catalog.

Every entry captures: a natural-language ``query`` a developer would plausibly
type, the ``server`` and ``tool`` pair that should surface for it, and the
raw tool ``description`` for reference.

The query generation is intentionally simple and deterministic so the benchmark
is reproducible: we strip the tool name, drop stopwords, and keep the first few
content words from the description. When the description is too short we fall
back to the human-friendly form of the tool name itself.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "data" / "mcp_catalog.json"
GOLDEN_PATH = ROOT / "data" / "golden_queries.json"

_STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "for",
    "to",
    "from",
    "in",
    "on",
    "by",
    "with",
    "without",
    "this",
    "that",
    "these",
    "those",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "it",
    "its",
    "as",
    "at",
    "into",
    "about",
    "your",
    "you",
    "any",
    "all",
    "some",
    "each",
    "one",
    "two",
    "can",
    "will",
    "use",
    "uses",
    "used",
    "using",
    "via",
    "over",
    "up",
    "down",
    "out",
    "off",
    "if",
    "then",
    "else",
    "so",
    "such",
    "also",
    "only",
    "just",
    "more",
    "less",
    "than",
    "when",
    "while",
    "per",
    "given",
    "their",
    "them",
    "they",
}

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]+")


def _humanize(name: str) -> list[str]:
    """Split ``listBucketObjects`` / ``LIST_BUCKET_OBJECTS`` into content words.

    Handles three naming styles:
    * ``snake_case`` and ``UPPER_SNAKE_CASE`` - split on underscores.
    * ``camelCase`` and ``PascalCase`` - split on transitions into an uppercase
      letter followed by a lowercase letter.
    * Mixed identifiers joined by ``/`` or ``-``.

    Returns a list of lowercase word tokens, dropping anything shorter than two
    characters so junk like ``s l`` from ``SLACK`` never leaks into a query.
    """
    parts = re.split(r"[_\-./]+", name)
    words: list[str] = []
    for part in parts:
        if not part:
            continue
        # If the part is all upper or all lower, it is already a single token.
        if part.isupper() or part.islower():
            words.append(part.lower())
            continue
        # Otherwise treat it as camel/PascalCase: split on lower→upper and
        # upper→upperLower boundaries.
        chunks = re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+", part)
        for chunk in chunks:
            if chunk:
                words.append(chunk.lower())
    return [w for w in words if len(w) >= 2]


def _content_words(text: str, *, skip: set[str]) -> list[str]:
    words: list[str] = []
    for token in _TOKEN_RE.findall(text.lower()):
        if token in _STOPWORDS:
            continue
        if token in skip:
            continue
        if len(token) < 3:
            continue
        words.append(token)
    return words


def _build_query(
    tool_name: str,
    description: str,
    *,
    idf: dict[str, float],
) -> str:
    """Build an IDF-weighted query from a tool's name and description.

    Real users search with *distinctive* nouns - not ``get``, ``list``, or
    ``search``. We rank candidate tokens by IDF (rarer = more distinctive) and
    pick the top four. A deterministic action verb pulled from the tool name
    is kept as a lightweight prefix so the query still looks like a natural
    intent (``download medrxiv paper`` instead of just ``medrxiv paper``).
    """
    name_words = _humanize(tool_name)
    name_words = [w for w in name_words if w not in _STOPWORDS and len(w) >= 3]
    # Candidate pool: all content words from both name and description.
    description_words = _content_words(description, skip=set())
    candidates = list(dict.fromkeys([*name_words, *description_words]))
    candidates = [w for w in candidates if len(w) >= 3 and w not in _STOPWORDS]
    if not candidates:
        return " ".join(name_words[:3]).strip()
    # Sort by IDF descending (rarer first).
    candidates.sort(key=lambda w: -idf.get(w, 0.0))
    distinctive = candidates[:4]
    # Prefix with the first action verb from the tool name if we have one, so
    # the query still reads like a natural intent.
    action = next(
        (
            w
            for w in name_words
            if w
            in {
                "list",
                "get",
                "create",
                "delete",
                "update",
                "search",
                "fetch",
                "download",
                "upload",
                "send",
                "add",
                "remove",
                "run",
                "execute",
                "query",
                "read",
                "write",
            }
        ),
        "",
    )
    tokens: list[str] = []
    if action:
        tokens.append(action)
    for word in distinctive:
        if word == action:
            continue
        if word not in tokens:
            tokens.append(word)
        if len(tokens) >= 5:
            break
    return " ".join(tokens).strip()


def _build_idf(catalog: dict[str, Any]) -> dict[str, float]:
    """Document-frequency IDF over all tool name+description tokens.

    A 'document' is one tool. The resulting IDF scores are used to weight
    words by their rarity in the catalog so that we can pick distinctive
    tokens as query terms.
    """
    df: Counter[str] = Counter()
    total = 0
    for server in catalog["servers"]:
        for tool in server["tools"]:
            total += 1
            tokens = set(_humanize(tool["name"]))
            for w in _content_words(tool.get("description", ""), skip=set()):
                tokens.add(w)
            for token in tokens:
                if len(token) >= 3 and token not in _STOPWORDS:
                    df[token] += 1
    idf: dict[str, float] = {}
    for token, freq in df.items():
        idf[token] = math.log((total + 1) / (freq + 1)) + 1.0
    return idf


def generate_golden(
    catalog: dict[str, Any],
    *,
    max_tools: int,
    seed: int,
) -> list[dict[str, Any]]:
    idf = _build_idf(catalog)
    rng = random.Random(seed)
    entries: list[dict[str, Any]] = []
    all_tools: list[tuple[str, dict[str, Any]]] = []
    for server in catalog["servers"]:
        for tool in server["tools"]:
            if not tool.get("description"):
                continue
            all_tools.append((server["qualified_name"], tool))
    rng.shuffle(all_tools)
    for server_name, tool in all_tools:
        if len(entries) >= max_tools:
            break
        query = _build_query(tool["name"], tool.get("description", ""), idf=idf)
        if len(query.split()) < 2:
            continue
        entries.append(
            {
                "query": query,
                "server": server_name,
                "tool": tool["name"],
                "description": tool.get("description", ""),
            }
        )
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max", type=int, default=500, help="Max golden queries.")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if not CATALOG_PATH.exists():
        raise SystemExit(f"{CATALOG_PATH} not found. Run fetch_mcps.py first.")
    catalog = json.loads(CATALOG_PATH.read_text())
    golden = generate_golden(catalog, max_tools=args.max, seed=args.seed)
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps({"queries": golden}, indent=2))
    print(f"Wrote {len(golden)} golden queries to {GOLDEN_PATH}")
    for entry in golden[:5]:
        print(f"  {entry['query']!r} -> {entry['server']}/{entry['tool']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
