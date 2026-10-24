"""Collect environment metadata for env.json (no secrets)."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from llm_tester.artifacts.secrets import strip_env_secrets

# Packages whose versions are useful for thesis reproducibility.
_TRACKED_PACKAGES = (
    "llm-tester",
    "openai",
    "python-dotenv",
    "datasets",
    "llm-guard",
    "PyYAML",
    "matplotlib",
    "numpy",
    "sentence-transformers",
    "mcp",
    "pytest",
)


def _git_sha(cwd: Optional[Path] = None) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd else None,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        return out.strip() or None
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None


def _git_dirty(cwd: Optional[Path] = None) -> Optional[bool]:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(cwd) if cwd else None,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        return bool(out.strip())
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None


def _is_git_worktree(path: Path) -> bool:
    """True if ``path`` is inside a git work tree (has .git file or dir)."""
    try:
        p = path.resolve()
    except OSError:
        return False
    for cand in [p, *p.parents]:
        git = cand / ".git"
        if git.exists():
            return True
    return False


def package_source_root() -> Optional[Path]:
    """
    Locate the git root (or package parent) from the installed ``llm_tester``
    source tree — so env.json still records a SHA when cwd is outside the repo.
    """
    try:
        import llm_tester
    except ImportError:
        return None
    start = Path(llm_tester.__file__).resolve().parent
    for cand in [start, start.parent, *start.parents]:
        if (cand / ".git").exists():
            return cand
        # Stop at filesystem root.
        if cand.parent == cand:
            break
    # Editable / src layout without walking into unrelated trees: package parent.
    return start.parent


def resolve_git_root(repo_root: Optional[Path] = None) -> Path:
    """Prefer explicit root, else cwd if it is a git worktree, else package source."""
    if repo_root is not None:
        return Path(repo_root)
    cwd = Path.cwd()
    if _is_git_worktree(cwd) and _git_sha(cwd):
        return cwd
    pkg = package_source_root()
    if pkg is not None and (_is_git_worktree(pkg) or _git_sha(pkg)):
        return pkg
    return cwd


def _package_versions() -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _hardware_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "python_implementation": platform.python_implementation(),
        "cpu_count": os.cpu_count(),
    }
    try:
        if hasattr(os, "sysconf"):
            page = os.sysconf("SC_PAGE_SIZE")
            phys = os.sysconf("SC_PHYS_PAGES")
            if isinstance(page, int) and isinstance(phys, int) and page > 0 and phys > 0:
                info["ram_bytes"] = page * phys
    except (ValueError, OSError, AttributeError):
        pass
    return info


def collect_env(
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    extra_sha: Optional[Sequence[str]] = None,
    command: Optional[str] = None,
    repo_root: Optional[Path] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build env.json payload (secrets never included as values)."""
    root = resolve_git_root(repo_root)
    sha = _git_sha(root)
    # If still null (e.g. broken cwd), retry from package source explicitly.
    source = "cwd_or_explicit"
    if sha is None and repo_root is None:
        pkg = package_source_root()
        if pkg is not None:
            pkg_sha = _git_sha(pkg)
            if pkg_sha:
                root = pkg
                sha = pkg_sha
                source = "package_source"
    elif repo_root is not None:
        source = "explicit"
    else:
        source = "cwd" if root == Path.cwd() else "package_source"

    env: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "git": {
            "sha": sha,
            "dirty": _git_dirty(root),
            "root": str(root.resolve()),
            "resolved_from": source,
        },
        "extra_sha": list(extra_sha) if extra_sha else [],
        "python": {
            "version": sys.version.split()[0],
            "executable": sys.executable,
            "platform": sys.platform,
        },
        "packages": _package_versions(),
        "provider": provider,
        "model": model,
        "hardware": _hardware_info(),
        # Presence-only flags for credential env vars (never values).
        "credential_env_flags": {
            k: v
            for k, v in strip_env_secrets(os.environ).items()
            if k.upper().endswith(("_API_KEY", "_TOKEN", "_SECRET", "_PASSWORD"))
            or "API_KEY" in k.upper()
        },
    }
    if extra:
        env["extra"] = extra
    return env
