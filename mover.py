"""mover.py — Safe file operations, collision handling, and undo for Sloth Sorter."""

from __future__ import annotations

import json
import shutil
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from classifier import Proposal

LOG_FILENAME = ".sloth_sorter_log.json"


@dataclass
class MoveRecord:
    """A single recorded file move."""

    src: str        # original absolute path
    dest: str       # destination absolute path
    timestamp: str  # ISO 8601


@dataclass
class MoveLog:
    """Result of an apply operation."""

    records: list[MoveRecord]
    errors: list[str]


# ── Internal helpers ───────────────────────────────────────────────────────────


def _resolve_collision(dest_path: Path) -> Path:
    """If *dest_path* already exists, append ``_1``, ``_2``, … before the
    extension until a free name is found.
    """
    if not dest_path.exists():
        return dest_path

    stem = dest_path.stem
    suffix = dest_path.suffix
    parent = dest_path.parent
    counter = 1
    while True:
        candidate = parent / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1
        if counter > 10_000:  # safety valve
            raise RuntimeError(f"Too many collisions for {dest_path}")


def _is_within(child: Path, parent: Path) -> bool:
    """Return ``True`` if *child* is strictly within *parent*."""
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _write_log(root_folder: Path, records: list[MoveRecord]) -> None:
    """Persist move records to the JSON log file."""
    log_path = root_folder / LOG_FILENAME
    log_data = [asdict(r) for r in records]
    try:
        log_path.write_text(json.dumps(log_data, indent=2), encoding="utf-8")
    except OSError as e:
        warnings.warn(f"Failed to write move log: {e}")


def _cleanup_empty_dirs(root_folder: Path) -> None:
    """Remove empty subdirectories inside *root_folder*."""
    for child in root_folder.iterdir():
        if child.is_dir():
            try:
                if not any(child.iterdir()):
                    child.rmdir()
            except OSError:
                pass


# ── Public API ─────────────────────────────────────────────────────────────────


def apply_moves(proposals: list[Proposal], root_folder: Path) -> MoveLog:
    """Move files according to the selected proposals.

    Safety guarantees
    -----------------
    - Both *src* and *dest* must be within *root_folder*.
    - Self-moves (src == dest) are silently skipped.
    - Collisions are resolved with numeric suffixes — never overwrites.
    - A JSON log is written for later undo.
    """
    root_folder = Path(root_folder).resolve()
    records: list[MoveRecord] = []
    errors: list[str] = []
    for proposal in proposals:
        src = proposal.original_path.resolve()
        requested_dest = proposal.dest_path.resolve()

        # Safety: path containment
        if not _is_within(src, root_folder):
            errors.append(f"Source outside root folder: {src}")
            continue
        if not _is_within(requested_dest, root_folder):
            errors.append(f"Destination outside root folder: {requested_dest}")
            continue

        # Skip files that are already at their intended destination
        if src == requested_dest:
            continue

        # Resolve filename collisions only after checking for self-moves
        dest = _resolve_collision(requested_dest)

        # Safety: path containment
        if not _is_within(src, root_folder):
            errors.append(f"Source outside root folder: {src}")
            continue
        if not _is_within(dest, root_folder):
            errors.append(f"Destination outside root folder: {dest}")
            continue

        # Self-move guard
        if src == dest:
            continue

        # Verify source still exists
        if not src.exists():
            errors.append(f"Source file not found: {src}")
            continue

        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))
            records.append(
                MoveRecord(
                    src=str(src),
                    dest=str(dest),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
            )
        except (OSError, shutil.Error) as e:
            errors.append(f"Failed to move {src.name}: {e}")

    log = MoveLog(records=records, errors=errors)
    if records:
        _write_log(root_folder, records)

    return log


def read_log(root_folder: Path) -> Optional[list[MoveRecord]]:
    """Read the move log from *root_folder*.

    Returns ``None`` if no log exists or parsing fails.
    """
    log_path = Path(root_folder).resolve() / LOG_FILENAME
    if not log_path.exists():
        return None

    try:
        data = json.loads(log_path.read_text(encoding="utf-8"))
        return [
            MoveRecord(src=r["src"], dest=r["dest"], timestamp=r["timestamp"])
            for r in data
        ]
    except (json.JSONDecodeError, KeyError, OSError) as e:
        warnings.warn(f"Failed to read move log: {e}")
        return None


def undo_last_run(root_folder: Path) -> MoveLog:
    """Reverse all moves from the last run.

    - Reads ``.sloth_sorter_log.json``
    - Moves each file from *dest* back to *src*
    - Removes empty category directories
    - Deletes the log file on success
    """
    root_folder = Path(root_folder).resolve()
    records = read_log(root_folder)

    if not records:
        return MoveLog(records=[], errors=["No move log found to undo."])

    undo_records: list[MoveRecord] = []
    errors: list[str] = []

    for record in reversed(records):
        src = Path(record.src)    # original location
        dest = Path(record.dest)  # current (moved-to) location

        if not dest.exists():
            errors.append(f"Cannot undo: file not found at {dest}")
            continue

        if src.exists():
            errors.append(
                f"Cannot undo: original location already occupied: {src}"
            )
            continue

        try:
            src.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(dest), str(src))
            undo_records.append(
                MoveRecord(
                    src=str(dest),
                    dest=str(src),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
            )
        except (OSError, shutil.Error) as e:
            errors.append(f"Failed to undo {dest.name}: {e}")

    # Clean up empty category directories
    _cleanup_empty_dirs(root_folder)

    # Delete the log on full success
    if not errors:
        log_path = root_folder / LOG_FILENAME
        try:
            log_path.unlink(missing_ok=True)
        except OSError:
            pass

    return MoveLog(records=undo_records, errors=errors)
