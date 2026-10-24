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
    root = repo_root or Path.cwd()
    env: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "git": {
            "sha": _git_sha(root),
            "dirty": _git_dirty(root),
            "root": str(root.resolve()),
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
