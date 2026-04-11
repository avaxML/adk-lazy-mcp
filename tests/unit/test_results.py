"""Unit tests for :mod:`adk_lazy_mcp.results`."""

from __future__ import annotations

from adk_lazy_mcp.results import normalize_result


class TestTextContent:
    def test_text_under_limit_is_passed_through(self) -> None:
        raw = {"content": [{"type": "text", "text": "hello"}]}
        result = normalize_result(raw, max_inline_bytes=100)
        assert result["content"] == [{"type": "text", "text": "hello"}]
        assert result["truncated"] is False
        assert result["artifact_refs"] == []

    def test_text_over_limit_is_truncated_and_flagged(self) -> None:
        raw = {"content": [{"type": "text", "text": "a" * 50}]}
        result = normalize_result(raw, max_inline_bytes=10)
        assert result["truncated"] is True
        assert result["content"][0]["text"] == "a" * 10

    def test_utf8_multibyte_boundary_is_safe(self) -> None:
        # "ä" is 2 bytes in UTF-8. Slicing at an odd boundary must not
        # produce a UnicodeDecodeError, just drop the partial codepoint.
        text = "ä" * 10  # 20 bytes
        result = normalize_result(
            {"content": [{"type": "text", "text": text}]},
            max_inline_bytes=5,
        )
        assert result["truncated"] is True
        trimmed = result["content"][0]["text"]
        # 5 bytes can fit 2 whole "ä" chars (4 bytes); the partial 5th byte is dropped.
        assert trimmed == "ää"

    def test_emoji_boundary_is_safe(self) -> None:
        # emoji takes 4 bytes in UTF-8
        raw = {"content": [{"type": "text", "text": "😀" * 3}]}  # 12 bytes
        result = normalize_result(raw, max_inline_bytes=6)
        assert result["truncated"] is True
        # Only one whole emoji fits in 6 bytes; 2 bytes of the next one get dropped.
        assert result["content"][0]["text"] == "😀"

    def test_empty_content_yields_empty_content(self) -> None:
        result = normalize_result({"content": []})
        assert result["content"] == []
        assert result["truncated"] is False

    def test_none_content_is_treated_as_empty(self) -> None:
        result = normalize_result({"content": None})
        assert result["content"] == []
        assert result["artifact_refs"] == []


class TestBinaryContent:
    def test_image_becomes_artifact_ref(self) -> None:
        raw = {
            "content": [
                {
                    "type": "image",
                    "mimeType": "image/png",
                    "uri": "https://example.com/img.png",
                }
            ]
        }
        result = normalize_result(raw)
        assert result["content"] == []
        assert result["artifact_refs"] == [
            {
                "kind": "image",
                "mime_type": "image/png",
                "uri": "https://example.com/img.png",
            }
        ]

    def test_audio_becomes_artifact_ref(self) -> None:
        raw = {"content": [{"type": "audio", "mimeType": "audio/mp3", "uri": "u"}]}
        result = normalize_result(raw)
        assert result["artifact_refs"][0]["kind"] == "audio"
        assert result["artifact_refs"][0]["mime_type"] == "audio/mp3"

    def test_image_without_uri_uses_placeholder(self) -> None:
        raw = {"content": [{"type": "image"}]}
        result = normalize_result(raw)
        assert result["artifact_refs"][0]["uri"] == "artifact://inline-offload"
        assert result["artifact_refs"][0]["mime_type"] == "application/octet-stream"


class TestMisc:
    def test_is_error_flag_propagates(self) -> None:
        result = normalize_result({"isError": True, "content": []})
        assert result["is_error"] is True

    def test_structured_data_passes_through(self) -> None:
        raw = {"content": [], "structuredContent": {"rows": 3}}
        result = normalize_result(raw)
        assert result["structured_data"] == {"rows": 3}

    def test_unknown_type_gets_preview(self) -> None:
        raw = {"content": [{"type": "resource", "uri": "file://x", "mimeType": "text/plain"}]}
        result = normalize_result(raw)
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "resource"
        assert result["content"][0]["uri"] == "file://x"
        assert result["content"][0]["mime_type"] == "text/plain"
        assert "preview" in result["content"][0]

    def test_mixed_content_types(self) -> None:
        raw = {
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "image", "uri": "u"},
                {"type": "text", "text": "world"},
            ]
        }
        result = normalize_result(raw)
        assert len(result["content"]) == 2
        assert len(result["artifact_refs"]) == 1

    def test_non_mapping_content_item_gets_preview(self) -> None:
        result = normalize_result({"content": ["unexpected"]})
        assert result["content"] == [{"type": "unknown", "preview": "unexpected"}]
