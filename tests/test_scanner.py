"""Tests for scanner.py — file discovery and metadata extraction."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scanner import FileInfo, scan_folder, SNIPPET_MAX_BYTES


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture()
def sample_folder(tmp_path: Path) -> Path:
    """Create a folder with a mix of sample files."""
    (tmp_path / "notes.txt").write_text("Hello, world!", encoding="utf-8")
    (tmp_path / "script.py").write_text("print('hi')", encoding="utf-8")
    (tmp_path / "photo.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)
    return tmp_path


# ── Tests ──────────────────────────────────────────────────────────────────────


class TestScanBasic:
    """Core scanning behaviour."""

    def test_scan_returns_file_infos(self, sample_folder: Path) -> None:
        results = scan_folder(sample_folder)
        assert len(results) == 3
        names = {fi.name for fi in results}
        assert names == {"notes.txt", "script.py", "photo.png"}

    def test_file_info_fields(self, sample_folder: Path) -> None:
        results = scan_folder(sample_folder)
        txt = next(fi for fi in results if fi.name == "notes.txt")
        assert txt.extension == ".txt"
        assert txt.size == len("Hello, world!")
        assert txt.snippet == "Hello, world!"
        assert txt.path == (sample_folder / "notes.txt").resolve()


class TestSkipHidden:
    """Hidden files must be excluded."""

    def test_dot_prefixed_files_skipped(self, tmp_path: Path) -> None:
        (tmp_path / ".hidden").write_text("secret", encoding="utf-8")
        (tmp_path / "visible.txt").write_text("ok", encoding="utf-8")
        results = scan_folder(tmp_path)
        names = {fi.name for fi in results}
        assert ".hidden" not in names
        assert "visible.txt" in names

    def test_dot_prefixed_dirs_not_entered(self, tmp_path: Path) -> None:
        hidden_dir = tmp_path / ".config"
        hidden_dir.mkdir()
        (hidden_dir / "file.txt").write_text("inside", encoding="utf-8")
        results = scan_folder(tmp_path)
        assert len(results) == 0  # no files in the top-level


class TestNoRecursion:
    """Scanner must NOT recurse into subdirectories."""

    def test_subdir_files_excluded(self, tmp_path: Path) -> None:
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "deep.txt").write_text("deep content", encoding="utf-8")
        (tmp_path / "top.txt").write_text("top content", encoding="utf-8")

        results = scan_folder(tmp_path)
        names = {fi.name for fi in results}
        assert "top.txt" in names
        assert "deep.txt" not in names


class TestEmptyFile:
    """Empty files should be reported with size 0 and empty snippet."""

    def test_empty_file_handled(self, tmp_path: Path) -> None:
        (tmp_path / "empty.txt").write_text("", encoding="utf-8")
        results = scan_folder(tmp_path)
        assert len(results) == 1
        assert results[0].size == 0
        assert results[0].snippet == ""


class TestSnippetLength:
    """Snippet must be capped at SNIPPET_MAX_BYTES characters."""

    def test_snippet_capped(self, tmp_path: Path) -> None:
        long_text = "A" * (SNIPPET_MAX_BYTES * 3)
        (tmp_path / "long.txt").write_text(long_text, encoding="utf-8")
        results = scan_folder(tmp_path)
        assert len(results[0].snippet) <= SNIPPET_MAX_BYTES


class TestBinaryFile:
    """Binary files should return an empty snippet."""

    def test_binary_snippet_empty(self, tmp_path: Path) -> None:
        (tmp_path / "image.png").write_bytes(b"\x89PNG" + os.urandom(100))
        results = scan_folder(tmp_path)
        assert len(results) == 1
        assert results[0].snippet == ""


class TestInvalidFolder:
    """Scanning a non-directory should raise."""

    def test_not_a_directory(self, tmp_path: Path) -> None:
        f = tmp_path / "afile.txt"
        f.write_text("x", encoding="utf-8")
        with pytest.raises(NotADirectoryError):
            scan_folder(f)
