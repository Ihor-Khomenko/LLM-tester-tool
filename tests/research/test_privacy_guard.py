"""Repo hygiene: research/ must not embed personal host paths or usernames."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RESEARCH = REPO / "research"

# Patterns that indicate a personal machine / identity leaked into the tree.
FORBIDDEN = (
    "/Users/",
    "ihkh",
    "Documents/University",
    "ua.ihkh",
    "ihor.k@",
    "Ihor's",
    "owner's Mac",
    "MacBook",
)

SKIP_DIR_NAMES = {"node_modules", ".git", "__pycache__", "sandbox"}


def _iter_text_files(root: Path):
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if any(part in SKIP_DIR_NAMES for part in p.parts):
            continue
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".pyc", ".lock"}:
            # package-lock.json is text but huge / third-party; skip binary-ish and lock noise
            if p.name != "package-lock.json" and p.suffix.lower() != ".lock":
                continue
            if p.name == "package-lock.json":
                continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        yield p, text


@pytest.mark.parametrize("needle", FORBIDDEN)
def test_research_tree_has_no_personal_markers(needle):
    hits = []
    for path, text in _iter_text_files(RESEARCH):
        if needle in text:
            # Report first few line numbers for the failure message.
            lines = [i for i, ln in enumerate(text.splitlines(), 1) if needle in ln][:5]
            hits.append(f"{path.relative_to(REPO)}:{','.join(map(str, lines))}")
    assert not hits, f"forbidden personal marker {needle!r} found in:\n  " + "\n  ".join(hits)
