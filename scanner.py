"""scanner.py — File discovery and metadata extraction for Sloth Sorter."""

from __future__ import annotations

import os
import stat
import warnings
from dataclasses import dataclass
from pathlib import Path

# Extensions considered as text files for snippet extraction
TEXT_EXTENSIONS: set[str] = {
    ".txt", ".md", ".py", ".csv", ".json", ".log", ".html", ".css",
    ".js", ".ts", ".java", ".c", ".cpp", ".h", ".xml", ".yaml",
    ".yml", ".toml", ".ini", ".cfg", ".sh", ".bat", ".ps1", ".rb",
    ".rs", ".go", ".sql",
}

SNIPPET_MAX_BYTES: int = 512


@dataclass
class FileInfo:
    """Metadata for a single scanned file."""

    path: Path       # absolute path
    name: str        # original filename (stem + suffix)
    extension: str   # lowercase, e.g. ".py"
    size: int        # bytes
    snippet: str     # first 512 bytes of text content, or ""


def _is_hidden(entry: os.DirEntry[str]) -> bool:
    """Check if a directory entry is hidden (dot-prefixed or Windows hidden attribute)."""
    if entry.name.startswith("."):
        return True
    try:
        if os.name == "nt":
            return bool(entry.stat().st_file_attributes & stat.FILE_ATTRIBUTE_HIDDEN)  # type: ignore[attr-defined]
    except (OSError, AttributeError):
        pass
    return False


def _read_snippet(path: Path, extension: str) -> str:
    """Read up to SNIPPET_MAX_BYTES from a text file.

    Returns an empty string for binary or unreadable files.
    """
    if extension not in TEXT_EXTENSIONS:
        return ""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read(SNIPPET_MAX_BYTES)
    except (OSError, PermissionError):
        return ""


def scan_folder(folder: Path) -> list[FileInfo]:
    """Scan a single directory (non-recursive) and return metadata for each
    visible file.

    Skips hidden files/directories, subdirectories, and unreadable entries.
    **Never** modifies, deletes, or executes any file.
    """
    folder = Path(folder).resolve()
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")

    results: list[FileInfo] = []

    try:
        entries = os.scandir(folder)
    except PermissionError:
        warnings.warn(f"Permission denied: {folder}")
        return results

    with entries:
        for entry in entries:
            try:
                if not entry.is_file(follow_symlinks=False):
                    continue
                if _is_hidden(entry):
                    continue

                path = Path(entry.path).resolve()
                name = entry.name
                extension = Path(name).suffix.lower()

                try:
                    size = entry.stat().st_size
                except OSError:
                    size = 0

                snippet = _read_snippet(path, extension)

                results.append(
                    FileInfo(
                        path=path,
                        name=name,
                        extension=extension,
                        size=size,
                        snippet=snippet,
                    )
                )
            except (OSError, PermissionError) as e:
                warnings.warn(f"Skipping {entry.name}: {e}")
                continue

    return results
