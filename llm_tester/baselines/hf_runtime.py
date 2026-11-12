"""Shared helpers for optional Hugging Face detector baselines (CPU / MPS).

Never downloads in CI: honour ``HF_HUB_OFFLINE`` / ``TRANSFORMERS_OFFLINE``.
Gated-model access denials surface a clear, actionable message.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Sequence, Tuple


def prefer_local_files() -> bool:
    return os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get(
        "TRANSFORMERS_OFFLINE"
    ) == "1"


def pick_torch_device(*, prefer_mps: bool = True) -> str:
    """Return ``mps`` when available and allowed, else ``cpu``.

    CUDA is intentionally not required; research hosts are CPU / Apple Silicon.
    Set ``OBERIH_DETECTOR_DEVICE=cpu|mps`` to force a device.
    """
    forced = (os.environ.get("OBERIH_DETECTOR_DEVICE") or "").strip().lower()
    if forced in ("cpu", "mps"):
        return forced
    try:
        import torch
    except ImportError:
        return "cpu"
    if prefer_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def is_hf_access_denied(exc: BaseException) -> bool:
    """True when Hugging Face refused gated / unauthenticated model access."""
    name = type(exc).__name__
    msg = str(exc).lower()
    needles = (
        "401",
        "403",
        "gated",
        "access to model",
        "cannot access",
        "restricted",
        "authorized",
        "authentication",
        "permission",
        "you need to agree",
        "accept the conditions",
        "huggingfacehubhttperror",
    )
    if name in ("GatedRepoError", "RepositoryNotFoundError", "HfHubHTTPError"):
        if "404" in msg and "gated" not in msg and "403" not in msg and "401" not in msg:
            # Missing repo ≠ access denied; still treat 401/403 variants below.
            pass
        else:
            return any(n in msg for n in ("401", "403", "gated", "restricted", "auth"))
    return any(n in msg for n in needles)


def gated_access_message(model_id: str, exc: BaseException) -> str:
    return (
        f"Access denied for gated model {model_id!r} "
        f"({type(exc).__name__}: {exc}). "
        f"Accept the model license on Hugging Face and ensure HF_TOKEN "
        f"(read-only is enough) can download it. "
        f"See https://huggingface.co/{model_id}"
    )


def unavailable_message(
    arm: str,
    model_id: str,
    exc: BaseException,
    *,
    env_path: Optional[str] = None,
) -> str:
    if is_hf_access_denied(exc):
        return gated_access_message(model_id, exc)
    hint = f" Set {env_path} for a local path." if env_path else ""
    return (
        f"{arm} unavailable ({type(exc).__name__}: {exc}). "
        f"Install transformers+torch (pip install 'llm-tester[detectors]') "
        f"and fetch weights for {model_id!r}.{hint} "
        f"See llm_tester/baselines/README.md."
    )


def move_model_to_device(model: Any, device: str) -> Any:
    try:
        return model.to(device)
    except Exception:
        return model


def resolve_positive_index(
    id2label: Any,
    *,
    positive_labels: Sequence[str],
    fallback_last: bool = True,
) -> int:
    """Pick the classifier class index that means injection / unsafe."""
    labels = {int(k): str(v).upper() for k, v in dict(id2label or {}).items()}
    wanted = {p.upper() for p in positive_labels}
    for idx, name in labels.items():
        if name in wanted or any(w in name for w in wanted):
            return int(idx)
    if fallback_last and labels:
        return max(labels)
    return 1


def softmax_positive_score(
    logits: Any,
    torch: Any,
    *,
    positive_index: int,
) -> float:
    probs = torch.softmax(logits, dim=-1)[0]
    if probs.numel() == 1:
        return float(probs[0].item())
    idx = int(positive_index)
    if idx < 0 or idx >= int(probs.numel()):
        idx = int(probs.numel()) - 1
    return float(probs[idx].item())


def hf_token_present() -> bool:
    return bool(
        (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    )
