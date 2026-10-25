"""classifier.py — AI categorization with Ollama + extension-based fallback."""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

from scanner import FileInfo

# ── Constants ──────────────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:4b"
REQUEST_TIMEOUT = 120  # seconds

VALID_CATEGORIES: set[str] = {
    "Documents",
    "Images",
    "Code",
    "Study",
    "Installers",
    "Archives",
    "Other",
}

MAX_FILENAME_LENGTH = 200
BATCH_SIZE = 20

# Characters forbidden in filenames (Windows + Unix)
FORBIDDEN_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

# ── Extension → Category fallback map ─────────────────────────────────────────

CATEGORY_MAP: dict[str, str] = {
    # Documents
    ".pdf": "Documents",
    ".doc": "Documents",
    ".docx": "Documents",
    ".txt": "Documents",
    ".odt": "Documents",
    ".rtf": "Documents",
    ".xls": "Documents",
    ".xlsx": "Documents",
    ".ppt": "Documents",
    ".pptx": "Documents",
    ".csv": "Documents",
    # Images
    ".png": "Images",
    ".jpg": "Images",
    ".jpeg": "Images",
    ".gif": "Images",
    ".bmp": "Images",
    ".svg": "Images",
    ".webp": "Images",
    ".ico": "Images",
    ".tiff": "Images",
    # Code
    ".py": "Code",
    ".js": "Code",
    ".ts": "Code",
    ".java": "Code",
    ".c": "Code",
    ".cpp": "Code",
    ".h": "Code",
    ".cs": "Code",
    ".go": "Code",
    ".rs": "Code",
    ".rb": "Code",
    ".php": "Code",
    ".html": "Code",
    ".css": "Code",
    ".sql": "Code",
    ".sh": "Code",
    ".bat": "Code",
    ".ps1": "Code",
    ".json": "Code",
    ".xml": "Code",
    ".yaml": "Code",
    ".yml": "Code",
    ".toml": "Code",
    # Study
    ".md": "Study",
    ".tex": "Study",
    ".bib": "Study",
    ".epub": "Study",
    ".mobi": "Study",
    # Installers
    ".exe": "Installers",
    ".msi": "Installers",
    ".dmg": "Installers",
    ".deb": "Installers",
    ".rpm": "Installers",
    ".appimage": "Installers",
    # Archives
    ".zip": "Archives",
    ".tar": "Archives",
    ".gz": "Archives",
    ".bz2": "Archives",
    ".7z": "Archives",
    ".rar": "Archives",
    ".xz": "Archives",
}


# ── Data classes ───────────────────────────────────────────────────────────────


@dataclass
class Proposal:
    """A proposed file move: category + suggested new name."""

    original_path: Path
    original_name: str
    category: str       # one of VALID_CATEGORIES
    new_name: str       # sanitized, with original extension preserved
    dest_path: Path     # root_folder / category / new_name
    ai_generated: bool  # True if from model, False if fallback


# ── Filename sanitization ─────────────────────────────────────────────────────


def sanitize_filename(suggested_name: str, original_extension: str) -> str:
    """Clean up a model-suggested filename.

    - Strip forbidden characters.
    - Remove leading/trailing whitespace and dots.
    - Cap length.
    - Preserve the original file extension.
    """
    # Remove any extension the model may have added
    stem = Path(suggested_name).stem
    # Strip forbidden chars
    stem = FORBIDDEN_CHARS.sub("", stem)
    # Strip leading/trailing whitespace and dots
    stem = stem.strip().strip(".")
    # Replace multiple spaces / underscores with a single underscore
    stem = re.sub(r"[\s_]+", "_", stem)
    # Cap length (leaving room for extension)
    max_stem = MAX_FILENAME_LENGTH - len(original_extension)
    if len(stem) > max_stem:
        stem = stem[:max_stem]
    # If nothing is left, use "unnamed"
    if not stem:
        stem = "unnamed"
    return stem + original_extension


# ── Fallback classification ───────────────────────────────────────────────────


def fallback_classify(file_info: FileInfo) -> tuple[str, str]:
    """Return ``(category, new_name)`` based on the extension mapping."""
    category = CATEGORY_MAP.get(file_info.extension, "Other")
    new_name = sanitize_filename(file_info.name, file_info.extension)
    return category, new_name


# ── Ollama interaction ────────────────────────────────────────────────────────


def _build_prompt(files: list[FileInfo]) -> str:
    """Build the prompt for the AI model."""
    file_descriptions: list[str] = []
    for i, f in enumerate(files, 1):
        desc = (
            f'  {i}. Filename: "{f.name}", Extension: "{f.extension}", '
            f"Size: {f.size} bytes"
        )
        if f.snippet:
            snippet_preview = f.snippet[:200].replace('"', '\\"')
            desc += f', Content preview: "{snippet_preview}"'
        file_descriptions.append(desc)

    files_block = "\n".join(file_descriptions)

    return (
        "You are a file organizer assistant. Analyze these files and suggest "
        "a category and a cleaner filename for each.\n\n"
        "ALLOWED CATEGORIES (use exactly one per file):\n"
        "- Documents (PDFs, Word docs, spreadsheets, text files)\n"
        "- Images (photos, graphics, icons)\n"
        "- Code (source code, scripts, config files)\n"
        "- Study (notes, research papers, ebooks)\n"
        "- Installers (executable installers, packages)\n"
        "- Archives (compressed files)\n"
        "- Other (anything that doesn't fit above)\n\n"
        f"FILES TO CLASSIFY:\n{files_block}\n\n"
        "Respond with ONLY a valid JSON array. Each element must have these "
        "exact keys:\n"
        '- "original": the original filename (exactly as given)\n'
        '- "category": one of the allowed categories above\n'
        '- "new_name": a cleaner, human-readable filename WITHOUT the file '
        "extension\n\n"
        "Example response format:\n"
        "[\n"
        '  {"original": "IMG_20230415_143022.jpg", "category": "Images", '
        '"new_name": "photo_2023_04_15"},\n'
        '  {"original": "asdf123.py", "category": "Code", '
        '"new_name": "untitled_script"}\n'
        "]\n\n"
        "Respond with ONLY the JSON array, no other text."
    )


def _parse_model_response(
    response_text: str,
    files: list[FileInfo],
) -> dict[str, tuple[str, str]]:
    """Parse the model's JSON response.

    Returns a dict mapping ``original_name -> (category, suggested_stem)``.
    Only entries that pass validation are included.
    """
    results: dict[str, tuple[str, str]] = {}

    text = response_text.strip()

    # Locate the JSON array in the response
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return results

    json_str = text[start : end + 1]

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return results

    if not isinstance(data, list):
        return results

    valid_names = {f.name for f in files}

    for entry in data:
        if not isinstance(entry, dict):
            continue

        original = entry.get("original", "")
        category = entry.get("category", "")
        new_name = entry.get("new_name", "")

        if original not in valid_names:
            continue
        if category not in VALID_CATEGORIES:
            continue
        if not new_name or not isinstance(new_name, str):
            continue

        results[original] = (category, new_name)

    return results


def _call_ollama(prompt: str) -> Optional[str]:
    """Call the local Ollama API.

    Returns the response text, or ``None`` on failure.
    """
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
            },
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "")
    except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
        warnings.warn(f"Ollama call failed: {e}")
        return None


def check_ollama_connection() -> bool:
    """Check if Ollama is reachable and the model is available."""
    try:
        resp = requests.get(
            "http://localhost:11434/api/tags",
            timeout=5,
        )
        resp.raise_for_status()
        models = resp.json().get("models", [])
        return any(MODEL_NAME in m.get("name", "") for m in models)
    except (requests.RequestException, json.JSONDecodeError):
        return False


# ── Main classification entry point ───────────────────────────────────────────


def classify_files(
    files: list[FileInfo],
    root_folder: Path,
) -> list[Proposal]:
    """Classify a list of files using the AI model with extension-based
    fallback.

    Returns a :class:`Proposal` for each file.
    """
    root_folder = Path(root_folder).resolve()
    proposals: list[Proposal] = []

    # Process files in batches to stay within context limits
    for batch_start in range(0, len(files), BATCH_SIZE):
        batch = files[batch_start : batch_start + BATCH_SIZE]

        # Attempt AI classification
        ai_results: dict[str, tuple[str, str]] = {}
        prompt = _build_prompt(batch)
        response_text = _call_ollama(prompt)
        if response_text:
            ai_results = _parse_model_response(response_text, batch)

        # Build proposals — AI result when available, otherwise fallback
        for file_info in batch:
            if file_info.name in ai_results:
                category, raw_new_name = ai_results[file_info.name]
                new_name = sanitize_filename(raw_new_name, file_info.extension)
                ai_generated = True
            else:
                category, new_name = fallback_classify(file_info)
                ai_generated = False

            dest_path = root_folder / category / new_name

            proposals.append(
                Proposal(
                    original_path=file_info.path,
                    original_name=file_info.name,
                    category=category,
                    new_name=new_name,
                    dest_path=dest_path,
                    ai_generated=ai_generated,
                )
            )

    return proposals
