"""ProtectAI DeBERTa prompt-injection classifier baseline.

Model: ``protectai/deberta-v3-base-prompt-injection-v2`` (sequence classifier).
Optional dependency (transformers+torch); skips gracefully when missing so CI
stays offline. Same T1/T2 surface as ``prompt_guard2``.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Sequence, Tuple

from llm_tester.baselines.hf_runtime import (
    move_model_to_device,
    pick_torch_device,
    prefer_local_files,
    resolve_positive_index,
    softmax_positive_score,
    unavailable_message,
)
from llm_tester.baselines.text_score_guard import TextScoreGuard

DEFAULT_MODEL_ID = "protectai/deberta-v3-base-prompt-injection-v2"
ENV_MODEL_PATH = "PROTECTAI_DEBERTA_MODEL_PATH"
DEFAULT_POSITIVE_LABELS: Tuple[str, ...] = (
    "INJECTION",
    "UNSAFE",
    "JAILBREAK",
    "MALICIOUS",
)


def _try_load_classifier(model_id: str, device: str):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch

    path = os.environ.get(ENV_MODEL_PATH) or model_id
    local = prefer_local_files()
    tok = AutoTokenizer.from_pretrained(path, local_files_only=local)
    model = AutoModelForSequenceClassification.from_pretrained(
        path, local_files_only=local
    )
    model.eval()
    model = move_model_to_device(model, device)
    return tok, model, torch


class ProtectAIDebertaGuard(TextScoreGuard):
    """DeBERTa-v3 prompt-injection classifier (ProtectAI v2)."""

    name = "protectai_deberta"
    skip_reason_tag = "skipped"

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        threshold: float = 0.5,
        positive_labels: Optional[Sequence[str]] = None,
        device: Optional[str] = None,
        classifier: Any = None,
    ) -> None:
        super().__init__(threshold=threshold, classifier=classifier)
        self.model_id = model_id
        self.positive_labels = tuple(positive_labels or DEFAULT_POSITIVE_LABELS)
        self.device = device or pick_torch_device()
        self._tok = None
        self._model = None
        self._torch = None
        self._positive_index = 1
        if classifier is None:
            self._lazy_load()

    def _lazy_load(self) -> None:
        try:
            self._tok, self._model, self._torch = _try_load_classifier(
                self.model_id, self.device
            )
            id2label = getattr(self._model.config, "id2label", {}) or {}
            self._positive_index = resolve_positive_index(
                id2label, positive_labels=self.positive_labels
            )
            self.available = True
            self.skip_reason = None
        except Exception as exc:  # ImportError, OSError, HTTP…
            self.available = False
            self.skip_reason = unavailable_message(
                self.name, self.model_id, exc, env_path=ENV_MODEL_PATH
            )

    def score_text(self, text: str) -> float:
        if not self.available:
            return 0.0
        if self._classifier_fn is not None:
            return float(min(1.0, max(0.0, self._classifier_fn(text or ""))))
        assert self._tok is not None and self._model is not None and self._torch is not None
        inputs = self._tok(
            text or "",
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        try:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        except Exception:
            pass
        with self._torch.no_grad():
            logits = self._model(**inputs).logits
            return softmax_positive_score(
                logits, self._torch, positive_index=self._positive_index
            )
