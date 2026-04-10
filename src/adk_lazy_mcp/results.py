from __future__ import annotations

from typing import Any


def normalize_result(raw: dict[str, Any], *, max_inline_bytes: int = 16_384) -> dict[str, Any]:
    """Normalize MCP CallToolResult-like payload to stable typed envelope."""
    content: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    truncated = False

    for item in raw.get("content", []):
        item_type = item.get("type", "text")
        if item_type == "text":
            text = item.get("text", "")
            if len(text.encode("utf-8")) > max_inline_bytes:
                text = text[: max_inline_bytes // 2]
                truncated = True
            content.append({"type": "text", "text": text, "preview": text[:200]})
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
                    "preview": str(item)[:200],
                }
            )

    return {
        "is_error": bool(raw.get("isError", False)),
        "content": content,
        "structured_data": raw.get("structuredContent"),
        "artifact_refs": artifacts,
        "truncated": truncated,
    }
