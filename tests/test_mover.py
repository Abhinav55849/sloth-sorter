"""Tests for mover.py — safe file moves, collision handling, and undo."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from classifier import Proposal
from mover import LOG_FILENAME, apply_moves, read_log, undo_last_run


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_proposal(
    root: Path,
    original_name: str,
    category: str,
    new_name: str,
) -> Proposal:
    """Create a Proposal with the file physically present in *root*."""
    src = root / original_name
    dest = root / category / new_name
    return Proposal(
        original_path=src,
        original_name=original_name,
        category=category,
        new_name=new_name,
        dest_path=dest,
        ai_generated=False,
    )


# ── Tests ──────────────────────────────────────────────────────────────────────


class TestMoveBasic:
    """Basic file move operations."""

    def test_file_moves_to_category_dir(self, tmp_path: Path) -> None:
        (tmp_path / "report.pdf").write_text("content", encoding="utf-8")
        proposal = _make_proposal(tmp_path, "report.pdf", "Documents", "report.pdf")

        result = apply_moves([proposal], tmp_path)

        assert len(result.records) == 1
        assert not (tmp_path / "report.pdf").exists()
        assert (tmp_path / "Documents" / "report.pdf").exists()

    def test_category_dir_created(self, tmp_path: Path) -> None:
        (tmp_path / "pic.png").write_text("img", encoding="utf-8")
        proposal = _make_proposal(tmp_path, "pic.png", "Images", "pic.png")

        apply_moves([proposal], tmp_path)

        assert (tmp_path / "Images").is_dir()


class TestCollisionAvoidance:
    """Duplicate filenames must get numeric suffixes."""

    def test_collision_appends_suffix(self, tmp_path: Path) -> None:
        (tmp_path / "file.txt").write_text("original", encoding="utf-8")
        dest_dir = tmp_path / "Documents"
        dest_dir.mkdir()
        (dest_dir / "file.txt").write_text("existing", encoding="utf-8")

        proposal = _make_proposal(tmp_path, "file.txt", "Documents", "file.txt")
        result = apply_moves([proposal], tmp_path)

        assert len(result.records) == 1
        assert (dest_dir / "file_1.txt").exists()
        # Original at destination is untouched
        assert (dest_dir / "file.txt").read_text(encoding="utf-8") == "existing"


class TestSelfMoveSkip:
    """Moving a file onto itself should be silently skipped."""

    def test_self_move_skipped(self, tmp_path: Path) -> None:
        cat_dir = tmp_path / "Code"
        cat_dir.mkdir()
        (cat_dir / "main.py").write_text("code", encoding="utf-8")

        # Proposal where src == dest
        proposal = Proposal(
            original_path=cat_dir / "main.py",
            original_name="main.py",
            category="Code",
            new_name="main.py",
            dest_path=cat_dir / "main.py",
            ai_generated=False,
        )

        result = apply_moves([proposal], tmp_path)
        assert len(result.records) == 0
        assert len(result.errors) == 0
        assert (cat_dir / "main.py").exists()


class TestPathEscapeRejected:
    """Destinations outside the root folder must be rejected."""

    def test_escape_blocked(self, tmp_path: Path) -> None:
        (tmp_path / "secret.txt").write_text("data", encoding="utf-8")
        evil_dest = tmp_path.parent / "escaped" / "secret.txt"

        proposal = Proposal(
            original_path=tmp_path / "secret.txt",
            original_name="secret.txt",
            category="Other",
            new_name="secret.txt",
            dest_path=evil_dest,
            ai_generated=False,
        )

        result = apply_moves([proposal], tmp_path)
        assert len(result.records) == 0
        assert any("outside root" in e.lower() for e in result.errors)
        # File still in original location
        assert (tmp_path / "secret.txt").exists()


class TestLogWritten:
    """A JSON log must be written after successful moves."""

    def test_log_created(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("aaa", encoding="utf-8")
        proposal = _make_proposal(tmp_path, "a.txt", "Documents", "a.txt")

        apply_moves([proposal], tmp_path)

        log_path = tmp_path / LOG_FILENAME
        assert log_path.exists()
        data = json.loads(log_path.read_text(encoding="utf-8"))
        assert len(data) == 1
        assert "src" in data[0]
        assert "dest" in data[0]


class TestUndo:
    """Undo should reverse all moves from the last run."""

    def test_undo_reverses_moves(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("aaa", encoding="utf-8")
        (tmp_path / "b.py").write_text("bbb", encoding="utf-8")
        proposals = [
            _make_proposal(tmp_path, "a.txt", "Documents", "a.txt"),
            _make_proposal(tmp_path, "b.py", "Code", "b.py"),
        ]

        apply_moves(proposals, tmp_path)
        assert not (tmp_path / "a.txt").exists()
        assert not (tmp_path / "b.py").exists()

        undo_result = undo_last_run(tmp_path)
        assert len(undo_result.records) == 2
        assert (tmp_path / "a.txt").exists()
        assert (tmp_path / "b.py").exists()
        # Category dirs should be cleaned up
        assert not (tmp_path / "Documents").exists()
        assert not (tmp_path / "Code").exists()

    def test_undo_cleans_up_log(self, tmp_path: Path) -> None:
        (tmp_path / "x.txt").write_text("x", encoding="utf-8")
        proposal = _make_proposal(tmp_path, "x.txt", "Documents", "x.txt")
        apply_moves([proposal], tmp_path)

        undo_last_run(tmp_path)

        assert not (tmp_path / LOG_FILENAME).exists()


class TestUndoMissingFile:
    """Undo should handle gracefully when a file has been manually moved/deleted."""

    def test_undo_missing_file_reports_error(self, tmp_path: Path) -> None:
        (tmp_path / "gone.txt").write_text("data", encoding="utf-8")
        proposal = _make_proposal(tmp_path, "gone.txt", "Documents", "gone.txt")
        apply_moves([proposal], tmp_path)

        # Simulate someone manually deleting the moved file
        (tmp_path / "Documents" / "gone.txt").unlink()

        result = undo_last_run(tmp_path)
        assert any("not found" in e.lower() for e in result.errors)


class TestNoLogUndo:
    """Undo with no log should return an error, not crash."""

    def test_undo_no_log(self, tmp_path: Path) -> None:
        result = undo_last_run(tmp_path)
        assert len(result.records) == 0
        assert any("no move log" in e.lower() for e in result.errors)
