"""Tests for classifier.py — categorization, sanitization, and fallback."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from classifier import (
    VALID_CATEGORIES,
    Proposal,
    classify_files,
    fallback_classify,
    sanitize_filename,
    _parse_model_response,
)
from scanner import FileInfo


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_file_info(
    name: str,
    extension: str = "",
    size: int = 100,
    snippet: str = "",
    tmp_path: Path | None = None,
) -> FileInfo:
    path = (tmp_path or Path("/fake")) / name
    ext = extension or Path(name).suffix.lower()
    return FileInfo(path=path, name=name, extension=ext, size=size, snippet=snippet)


# ── Fallback mapping tests ────────────────────────────────────────────────────


class TestFallbackMapping:
    """Extension-based classification without AI."""

    @pytest.mark.parametrize(
        "ext, expected",
        [
            (".pdf", "Documents"),
            (".docx", "Documents"),
            (".csv", "Documents"),
            (".png", "Images"),
            (".jpg", "Images"),
            (".py", "Code"),
            (".js", "Code"),
            (".yaml", "Code"),
            (".md", "Study"),
            (".epub", "Study"),
            (".exe", "Installers"),
            (".msi", "Installers"),
            (".zip", "Archives"),
            (".tar", "Archives"),
        ],
    )
    def test_known_extensions(self, ext: str, expected: str) -> None:
        fi = _make_file_info(f"file{ext}", extension=ext)
        category, _ = fallback_classify(fi)
        assert category == expected

    def test_unknown_extension_falls_back_to_other(self) -> None:
        fi = _make_file_info("mystery.xyz", extension=".xyz")
        category, _ = fallback_classify(fi)
        assert category == "Other"


# ── Filename sanitization tests ───────────────────────────────────────────────


class TestSanitizeFilename:
    """Ensure filenames are cleaned and safe."""

    def test_strips_forbidden_chars(self) -> None:
        result = sanitize_filename('my<>:"/\\|?*file', ".txt")
        assert "<" not in result
        assert ">" not in result
        assert ":" not in result
        assert result.endswith(".txt")

    def test_preserves_original_extension(self) -> None:
        result = sanitize_filename("report.docx", ".pdf")
        assert result.endswith(".pdf")
        assert not result.endswith(".docx")

    def test_name_length_capped(self) -> None:
        long_name = "a" * 500
        result = sanitize_filename(long_name, ".txt")
        assert len(result) <= 200

    def test_empty_name_becomes_unnamed(self) -> None:
        result = sanitize_filename("", ".txt")
        assert result == "unnamed.txt"

    def test_only_dots_becomes_unnamed(self) -> None:
        result = sanitize_filename("...", ".py")
        assert result == "unnamed.py"

    def test_collapses_spaces_and_underscores(self) -> None:
        result = sanitize_filename("my   file__name", ".txt")
        assert "  " not in result
        assert "__" not in result


# ── Model response parsing tests ──────────────────────────────────────────────


class TestParseModelResponse:
    """Validate JSON parsing of model output."""

    def test_valid_response_parsed(self) -> None:
        files = [_make_file_info("report.pdf", extension=".pdf")]
        response = json.dumps([
            {"original": "report.pdf", "category": "Documents", "new_name": "quarterly_report"},
        ])
        result = _parse_model_response(response, files)
        assert "report.pdf" in result
        assert result["report.pdf"] == ("Documents", "quarterly_report")

    def test_invalid_json_returns_empty(self) -> None:
        files = [_make_file_info("file.txt", extension=".txt")]
        result = _parse_model_response("not json at all {{{", files)
        assert result == {}

    def test_invalid_category_rejected(self) -> None:
        files = [_make_file_info("file.txt", extension=".txt")]
        response = json.dumps([
            {"original": "file.txt", "category": "INVALID", "new_name": "clean"},
        ])
        result = _parse_model_response(response, files)
        assert "file.txt" not in result

    def test_missing_new_name_rejected(self) -> None:
        files = [_make_file_info("file.txt", extension=".txt")]
        response = json.dumps([
            {"original": "file.txt", "category": "Documents", "new_name": ""},
        ])
        result = _parse_model_response(response, files)
        assert "file.txt" not in result

    def test_unknown_original_rejected(self) -> None:
        files = [_make_file_info("file.txt", extension=".txt")]
        response = json.dumps([
            {"original": "nope.txt", "category": "Documents", "new_name": "clean"},
        ])
        result = _parse_model_response(response, files)
        assert result == {}

    def test_json_embedded_in_text(self) -> None:
        """Model sometimes wraps JSON in markdown fences or prose."""
        files = [_make_file_info("file.py", extension=".py")]
        response = (
            'Sure! Here is the result:\n```json\n'
            '[{"original": "file.py", "category": "Code", "new_name": "main_script"}]\n'
            '```'
        )
        result = _parse_model_response(response, files)
        assert "file.py" in result


# ── End-to-end classify_files (with mocked Ollama) ────────────────────────────


class TestClassifyFiles:
    """Full classify_files pipeline with mocked HTTP."""

    @patch("classifier._call_ollama")
    def test_ai_result_used_when_valid(
        self, mock_ollama: MagicMock, tmp_path: Path
    ) -> None:
        fi = _make_file_info("notes.txt", extension=".txt", tmp_path=tmp_path)
        mock_ollama.return_value = json.dumps([
            {"original": "notes.txt", "category": "Study", "new_name": "lecture_notes"},
        ])

        proposals = classify_files([fi], tmp_path)
        assert len(proposals) == 1
        assert proposals[0].category == "Study"
        assert proposals[0].new_name == "lecture_notes.txt"
        assert proposals[0].ai_generated is True

    @patch("classifier._call_ollama")
    def test_fallback_when_ollama_fails(
        self, mock_ollama: MagicMock, tmp_path: Path
    ) -> None:
        fi = _make_file_info("image.png", extension=".png", tmp_path=tmp_path)
        mock_ollama.return_value = None  # simulate failure

        proposals = classify_files([fi], tmp_path)
        assert len(proposals) == 1
        assert proposals[0].category == "Images"
        assert proposals[0].ai_generated is False

    @patch("classifier._call_ollama")
    def test_fallback_when_json_invalid(
        self, mock_ollama: MagicMock, tmp_path: Path
    ) -> None:
        fi = _make_file_info("data.csv", extension=".csv", tmp_path=tmp_path)
        mock_ollama.return_value = "I can't help with that"

        proposals = classify_files([fi], tmp_path)
        assert proposals[0].category == "Documents"
        assert proposals[0].ai_generated is False
