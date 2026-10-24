"""PromptGuard 2 (22M) baseline — Meta Llama-Prompt-Guard-2-22M via transformers.

Weights are **not** bundled. When unavailable the guard skips gracefully
(ALLOW, ``available=False``) so CI stays offline. See
``llm_tester/baselines/README.md`` for fetch instructions.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Union

from llm_tester.baselines.protocol import BaseGuard, allow_decision, as_tool_call
from llm_tester.sentinel.types import Checkpoint, Decision, GuardDecision, ToolCall

DEFAULT_MODEL_ID = "meta-llama/Llama-Prompt-Guard-2-22M"
# Local override for offline / vendored weights.
ENV_MODEL_PATH = "PROMPT_GUARD_MODEL_PATH"


def _try_load_classifier(model_id: str):
    """Return (tokenizer, model) or raise ImportError/OSError."""
    # Offline CI: never download. TRANSFORMERS_OFFLINE / HF_HUB_OFFLINE set in tests.
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch

    path = os.environ.get(ENV_MODEL_PATH) or model_id
    tok = AutoTokenizer.from_pretrained(path, local_files_only=_prefer_local())
    model = AutoModelForSequenceClassification.from_pretrained(
        path, local_files_only=_prefer_local()
    )
    model.eval()
    return tok, model, torch


def _prefer_local() -> bool:
    return os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get(
        "TRANSFORMERS_OFFLINE"
    ) == "1"


class PromptGuard2Guard(BaseGuard):
    """Classify tool outputs / call verbalizations for prompt injection."""

    name = "prompt_guard2"

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        threshold: float = 0.5,
        classifier: Any = None,
    ) -> None:
        super().__init__()
        self.model_id = model_id
        self.threshold = float(threshold)
        self._tok = None
        self._model = None
        self._torch = None
        self.available = False
        self.skip_reason = None
        if classifier is not None:
            # Injected mock: callable(text) -> float score in [0,1] (jailbreak/injection).
            self._classifier_fn = classifier
            self.available = True
        else:
            self._classifier_fn = None
            self._lazy_load()

    def _lazy_load(self) -> None:
        try:
            self._tok, self._model, self._torch = _try_load_classifier(self.model_id)
            self.available = True
            self.skip_reason = None
        except Exception as exc:  # ImportError, OSError, HTTP… — skip gracefully
            self.available = False
            self.skip_reason = (
                f"PromptGuard2 unavailable ({type(exc).__name__}: {exc}). "
                f"Install transformers+torch and fetch weights — see "
                f"llm_tester/baselines/README.md. Set {ENV_MODEL_PATH} for a local path."
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
        with self._torch.no_grad():
            logits = self._model(**inputs).logits
            probs = self._torch.softmax(logits, dim=-1)[0]
            # Label mapping varies; treat highest non-benign / index 1 as injection.
            if probs.numel() == 1:
                return float(probs[0].item())
            # Common: 0=benign, 1=injection
            return float(probs[-1].item())

    def on_tool_output(
        self, source: str, content: str, *, sanitize: bool = True
    ) -> GuardDecision:
        start = time.perf_counter()
        self._note_output(source, content or "")
        if not self.available:
            return allow_decision(
                checkpoint=Checkpoint.T1.value,
                stage=self.name,
                latency_ms=(time.perf_counter() - start) * 1000,
                reasons=["prompt_guard2:skipped"],
                meta={
                    "available": False,
                    "skip_reason": self.skip_reason,
                    "llm_calls": 0,
                },
            )
        score = self.score_text(content or "")
        decision = Decision.SANITIZE if score >= self.threshold else Decision.ALLOW
        reasons = [
            f"prompt_guard2:score={score:.3f}",
        ]
        if decision == Decision.SANITIZE:
            reasons.append("prompt_guard2:injection")
        return allow_decision(
            checkpoint=Checkpoint.T1.value,
            stage=self.name,
            risk=score,
            latency_ms=(time.perf_counter() - start) * 1000,
            reasons=reasons,
            decision=decision,
            sanitized_content=(content or "") if decision == Decision.SANITIZE else None,
            meta={"available": True, "score": score, "llm_calls": 0},
        )

    def check_tool_call(
        self,
        call: Union[ToolCall, Dict[str, Any]],
        *,
        user_request: Optional[str] = None,
    ) -> GuardDecision:
        start = time.perf_counter()
        if user_request is not None:
            self.set_user_task(user_request)
        tc = as_tool_call(call)
        if not self.available:
            return allow_decision(
                checkpoint=Checkpoint.T2.value,
                stage=self.name,
                latency_ms=(time.perf_counter() - start) * 1000,
                reasons=["prompt_guard2:skipped"],
                meta={
                    "available": False,
                    "skip_reason": self.skip_reason,
                    "llm_calls": 0,
                },
            )
        blob = f"User: {self.user_request}\nCall: {tc.verbalize()}"
        # Include recent tool outputs (injection often lives there).
        for o in self.outputs[-3:]:
            blob += f"\nOutput[{o['source']}]: {o['content'][:500]}"
        score = self.score_text(blob)
        decision = Decision.BLOCK if score >= self.threshold else Decision.ALLOW
        return allow_decision(
            checkpoint=Checkpoint.T2.value,
            stage=self.name,
            risk=score,
            latency_ms=(time.perf_counter() - start) * 1000,
            reasons=[
                f"prompt_guard2:score={score:.3f}",
                "prompt_guard2:block" if decision == Decision.BLOCK else "prompt_guard2:allow",
            ],
            decision=decision,
            meta={"available": True, "score": score, "llm_calls": 0},
        )
