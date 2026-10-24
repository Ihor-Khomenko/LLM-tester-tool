"""Global test fixtures — guarantee tests never hit the real network or APIs."""

from __future__ import annotations

import os
import socket

import pytest

# Force Hugging Face / datasets offline before any test imports loaders that
# might otherwise try to download. Real downloads happen only via manual CLI use.
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


@pytest.fixture(autouse=True)
def _block_network(monkeypatch):
    """Fail loudly if any code opens a real network socket during tests."""

    def _guarded_socket(*args, **kwargs):
        raise RuntimeError(
            "Network access is disabled during tests. "
            "Mock providers/HF loaders instead of making real API calls."
        )

    monkeypatch.setattr(socket, "socket", _guarded_socket)

    # Also block common high-level clients if imported unmocked.
    try:
        import urllib.request

        def _urlopen_guard(*args, **kwargs):
            raise RuntimeError("urllib network call blocked in tests")

        monkeypatch.setattr(urllib.request, "urlopen", _urlopen_guard)
    except Exception:
        pass

    yield
