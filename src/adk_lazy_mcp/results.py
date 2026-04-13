from __future__ import annotations

from typing import Any

_PREVIEW_CHARS = 200


def _truncate_utf8(text: str, max_bytes: int) -> tuple[str, bool]:
    """Return text truncated to at most ``max_bytes`` UTF-8 bytes, plus a truncated flag."""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text, False
    # Decode with ``ignore`` to drop the partial multi-byte sequence at the boundary.
    return encoded[:max_bytes].decode("utf-8", errors="ignore"), True


def normalize_result(raw: dict[str, Any], *, max_inline_bytes: int = 16_384) -> dict[str, Any]:
    """Normalize MCP CallToolResult-like payload to stable typed envelope."""
    content: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    truncated = False
    raw_content = raw.get("content")
    items = raw_content if isinstance(raw_content, list) else []

    for item in items:
        if not isinstance(item, dict):
            content.append({"type": "unknown", "preview": str(item)[:_PREVIEW_CHARS]})
            continue
        item_type = item.get("type", "text")
        if item_type == "text":
            text = item.get("text", "")
            trimmed, was_truncated = _truncate_utf8(text, max_inline_bytes)
            if was_truncated:
                truncated = True
            content.append({"type": "text", "text": trimmed})
        elif item_type in {"image", "audio"}:
            artifacts.append(
                {
                    "kind": item_type,
                    "mime_type": item.get("mimeType", "application/octet-stream"),
                    "uri": item.get("uri", "artifact://inline-offload"),
                }
            )
        else:
            content.append(
                {
                    "type": item_type,
                    "uri": item.get("uri"),
                    "mime_type": item.get("mimeType"),
                    "preview": str(item)[:_PREVIEW_CHARS],
                }
            )

    return {
        "is_error": bool(raw.get("isError", False)),
        "content": content,
        "structured_data": raw.get("structuredContent"),
        "artifact_refs": artifacts,
        "truncated": truncated,
    }
