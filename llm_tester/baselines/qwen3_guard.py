"""Qwen3Guard-Gen baseline (smallest size).

Model: ``Qwen/Qwen3Guard-Gen-0.6B`` — generative safety moderator with
Safe / Controversial / Unsafe labels (Apache-2.0). Stream variants exist but
Gen is the documented text-classification path for offline batch scoring.
Same T1/T2 surface as ``prompt_guard2``.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional

from llm_tester.baselines.hf_runtime import (
    move_model_to_device,
    pick_torch_device,
    prefer_local_files,
    unavailable_message,
)
from llm_tester.baselines.text_score_guard import TextScoreGuard

DEFAULT_MODEL_ID = "Qwen/Qwen3Guard-Gen-0.6B"
ENV_MODEL_PATH = "QWEN3_GUARD_MODEL_PATH"
DEFAULT_LABEL_SCORES: Dict[str, float] = {
    "safe": 0.0,
    "controversial": 0.5,
    "unsafe": 1.0,
}


def _try_load_causal(model_id: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    path = os.environ.get(ENV_MODEL_PATH) or model_id
    local = prefer_local_files()
    tok = AutoTokenizer.from_pretrained(path, local_files_only=local, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        local_files_only=local,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model.eval()
    model = move_model_to_device(model, device)
    return tok, model, torch


def parse_qwen3_guard_label(text: str) -> str:
    """Extract Safe / Controversial / Unsafe from Qwen3Guard-Gen output."""
    raw = (text or "").strip()
    if not raw:
        return "safe"
    # Prefer explicit "Safety: X" lines from the model card format.
    m = re.search(r"safety\s*:\s*(safe|controversial|unsafe)", raw, re.I)
    if m:
        return m.group(1).lower()
    lower = raw.lower()
    for label in ("unsafe", "controversial", "safe"):
        if re.search(rf"\b{label}\b", lower):
            return label
    return "safe"


class Qwen3GuardGenGuard(TextScoreGuard):
    """Qwen3Guard-Gen-0.6B generative safety classifier."""

    name = "qwen3_guard"
    skip_reason_tag = "skipped"

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        threshold: float = 0.5,
        label_scores: Optional[Dict[str, float]] = None,
        device: Optional[str] = None,
        max_new_tokens: int = 64,
        classifier: Any = None,
    ) -> None:
        super().__init__(threshold=threshold, classifier=classifier)
        self.model_id = model_id
        self.label_scores = dict(label_scores or DEFAULT_LABEL_SCORES)
        self.device = device or pick_torch_device()
        self.max_new_tokens = int(max_new_tokens)
        self._tok = None
        self._model = None
        self._torch = None
        if classifier is None:
            self._lazy_load()

    def _lazy_load(self) -> None:
        try:
            self._tok, self._model, self._torch = _try_load_causal(
                self.model_id, self.device
            )
            self.available = True
            self.skip_reason = None
        except Exception as exc:
            self.available = False
            self.skip_reason = unavailable_message(
                self.name, self.model_id, exc, env_path=ENV_MODEL_PATH
            )

    def _build_prompt(self, text: str) -> str:
        # Prompt moderation: single user turn (chat template embeds policy).
        messages = [{"role": "user", "content": text or ""}]
        tok = self._tok
        assert tok is not None
        if hasattr(tok, "apply_chat_template"):
            try:
                return tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass
        return f"User: {text or ''}\nSafety:"

    def _generate(self, text: str) -> str:
        assert self._tok is not None and self._model is not None and self._torch is not None
        prompt = self._build_prompt(text)
        inputs = self._tok(prompt, return_tensors="pt", truncation=True, max_length=2048)
        try:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        except Exception:
            pass
        with self._torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=getattr(self._tok, "eos_token_id", None),
            )
        gen = out[0][inputs["input_ids"].shape[-1] :]
        return self._tok.decode(gen, skip_special_tokens=True)

    def score_text(self, text: str) -> float:
        if not self.available:
            return 0.0
        if self._classifier_fn is not None:
            return float(min(1.0, max(0.0, self._classifier_fn(text or ""))))
        try:
            raw = self._generate(text or "")
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return 0.0
        label = parse_qwen3_guard_label(raw)
        return float(self.label_scores.get(label, 0.0))
