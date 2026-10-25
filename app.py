"""app.py — Streamlit UI for Sloth Sorter."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from classifier import Proposal, check_ollama_connection, classify_files
from mover import apply_moves, read_log, undo_last_run
from scanner import scan_folder

# ── Page configuration ─────────────────────────────────────────────────────────

st.set_page_config(
    page_title="🦥 Sloth Sorter",
    page_icon="🦥",
    layout="wide",
)

# ── Session state defaults ─────────────────────────────────────────────────────

_DEFAULTS: dict[str, object] = {
    "stage": "select",        # select → preview → done
    "folder_path": "",
    "proposals": [],          # list[Proposal]
    "selected": [],           # list[bool]
    "move_result": None,
    "undo_result": None,
}

for key, default in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default


def _reset() -> None:
    """Reset the session to the initial state."""
    for key, default in _DEFAULTS.items():
        st.session_state[key] = default


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🦥 Sloth Sorter")
    st.caption("AI-powered local file organizer")
    st.divider()

    # Ollama connection status
    ollama_ok = check_ollama_connection()
    if ollama_ok:
        st.success("✅ Ollama connected (gemma3:4b)")
    else:
        st.warning(
            "⚠️ Ollama not reachable — extension-based fallback will be used."
        )

    st.divider()
    if st.button("🔄 Start Over"):
        _reset()
        st.rerun()

    st.divider()
    st.markdown(
        "Made for [Hacktoberfest AI Challenge](https://hacktoberfest.com) · "
        "[GitHub](https://github.com)"
    )

# ── Main content ───────────────────────────────────────────────────────────────

st.header("🦥 Sloth Sorter")
st.markdown("Organize messy folders with local AI — no cloud, no API keys.")

# ── Stage 1: Select folder ─────────────────────────────────────────────────────

if st.session_state.stage == "select":
    st.subheader("① Select a folder to organize")

    folder_input = st.text_input(
        "Folder path",
        value=st.session_state.folder_path,
        placeholder=r"C:\Users\you\Desktop\messy_folder",
    )

    col1, col2 = st.columns([1, 4])
    scan_clicked = col1.button("🔍 Scan & Classify", type="primary")

    if scan_clicked:
        folder_path = Path(folder_input.strip())

        # ── Validate folder ────────────────────────────────────────────────
        if not folder_input.strip():
            st.error("Please enter a folder path.")
        elif not folder_path.exists():
            st.error(f"Path does not exist: `{folder_path}`")
        elif not folder_path.is_dir():
            st.error(f"Not a directory: `{folder_path}`")
        else:
            st.session_state.folder_path = str(folder_path)

            with st.spinner("Scanning files…"):
                try:
                    files = scan_folder(folder_path)
                except Exception as exc:
                    st.error(f"Scan failed: {exc}")
                    st.stop()

            if not files:
                st.warning("No files found in this folder (hidden files are skipped).")
                st.stop()

            st.info(f"Found **{len(files)}** file(s). Classifying with AI…")

            with st.spinner("Classifying files with Ollama (gemma3:4b)…"):
                try:
                    proposals = classify_files(files, folder_path)
                except Exception as exc:
                    st.error(f"Classification failed: {exc}")
                    st.stop()

            st.session_state.proposals = proposals
            st.session_state.selected = [True] * len(proposals)
            st.session_state.stage = "preview"
            st.rerun()

# ── Stage 2: Dry-run preview ───────────────────────────────────────────────────

elif st.session_state.stage == "preview":
    proposals: list[Proposal] = st.session_state.proposals
    root = Path(st.session_state.folder_path)

    st.subheader("② Dry-Run Preview")
    st.info("🛡️ **No files have been moved yet.** Review the proposals below and "
            "select which ones to apply.")

    # ── Select / deselect all ──────────────────────────────────────────────
    col_a, col_b, _ = st.columns([1, 1, 3])
    if col_a.button("Select All"):
        st.session_state.selected = [True] * len(proposals)
        st.rerun()
    if col_b.button("Deselect All"):
        st.session_state.selected = [False] * len(proposals)
        st.rerun()

    # ── Proposal table with checkboxes ─────────────────────────────────────
    for i, proposal in enumerate(proposals):
        col_check, col_orig, col_cat, col_new, col_badge = st.columns(
            [0.5, 3, 1.5, 3, 1]
        )

        st.session_state.selected[i] = col_check.checkbox(
            "sel",
            value=st.session_state.selected[i],
            key=f"chk_{i}",
            label_visibility="collapsed",
        )
        col_orig.text(proposal.original_name)
        col_cat.text(proposal.category)
        col_new.text(proposal.new_name)
        col_badge.markdown(
            "🤖 AI" if proposal.ai_generated else "📂 Fallback"
        )

    st.divider()

    # ── Summary ────────────────────────────────────────────────────────────
    selected_count = sum(st.session_state.selected)
    st.markdown(f"**{selected_count}** of **{len(proposals)}** proposals selected.")

    # ── Action buttons ─────────────────────────────────────────────────────
    col_apply, col_back, _ = st.columns([1, 1, 3])

    if col_apply.button(
        "✅ Apply Selected",
        type="primary",
        disabled=selected_count == 0,
    ):
        chosen = [
            p
            for p, sel in zip(proposals, st.session_state.selected)
            if sel
        ]

        with st.spinner("Moving files…"):
            result = apply_moves(chosen, root)

        st.session_state.move_result = result
        st.session_state.stage = "done"
        st.rerun()

    if col_back.button("← Back"):
        st.session_state.stage = "select"
        st.session_state.proposals = []
        st.session_state.selected = []
        st.rerun()

# ── Stage 3: Done / Undo ───────────────────────────────────────────────────────

elif st.session_state.stage == "done":
    result = st.session_state.move_result
    root = Path(st.session_state.folder_path)

    st.subheader("③ Done!")

    if result and result.records:
        st.success(f"✅ Successfully moved **{len(result.records)}** file(s).")

        with st.expander("View move details"):
            for rec in result.records:
                st.text(f"{Path(rec.src).name}  →  {Path(rec.dest).relative_to(root)}")

    if result and result.errors:
        st.warning(f"⚠️ {len(result.errors)} issue(s) encountered:")
        for err in result.errors:
            st.text(f"  • {err}")

    if result and not result.records and not result.errors:
        st.info("Nothing to move (all proposals were self-moves or skipped).")

    st.divider()

    # ── Undo ───────────────────────────────────────────────────────────────
    col_undo, col_restart, _ = st.columns([1, 1, 3])

    has_log = read_log(root) is not None
    if col_undo.button("↩️ Undo Last Run", disabled=not has_log):
        with st.spinner("Undoing moves…"):
            undo_result = undo_last_run(root)

        if undo_result.records:
            st.success(
                f"↩️ Undone **{len(undo_result.records)}** move(s)."
            )
        if undo_result.errors:
            st.warning("Some undo operations failed:")
            for err in undo_result.errors:
                st.text(f"  • {err}")

    if col_restart.button("🔄 Organize Another Folder"):
        _reset()
        st.rerun()
