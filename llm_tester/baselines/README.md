# Paper baselines (MCP-Sentinel / Оберіг comparisons)

Pluggable guards implementing the same duck-typed surface as `SentinelGuard`
(`set_user_task`, `reset_session`, `check_tool_registration`, `on_tool_output`,
`check_tool_call`) so they run through `evaluate_traces` / `baselines-eval` /
`detector-eval`.

| Name | Method | Offline CI |
|------|--------|------------|
| `spotlighting` | Datamark / encode tool outputs (Hines et al. 2024) | Yes (pure Python) |
| `prompt_guard2` | Meta Llama-Prompt-Guard-2-22M classifier | Skips if weights missing |
| `protectai_deberta` | `protectai/deberta-v3-base-prompt-injection-v2` | Skips if weights missing |
| `llama_guard3` | `meta-llama/Llama-Guard-3-1B` (gated) | Skips; clear message if HF access denied |
| `qwen3_guard` | `Qwen/Qwen3Guard-Gen-0.6B` (Safe/Controversial/Unsafe) | Skips if weights missing |
| `progent` | Static privilege / capability allowlist | Yes |
| `task_shield` | LLM ALLOW/BLOCK checker (Ollama `qwen2.5:3b`) | Mock provider in tests |
| `sentinel` / `oberih` | Оберіг cascade | Yes (S1–S3 lite) |

Detector arms (`prompt_guard2`, `protectai_deberta`, `llama_guard3`,
`qwen3_guard`) share the PromptGuard pattern: **T1** sanitizes tool outputs
when `score >= threshold` (default 0.5); **T2** blocks tool calls scored from
the verbalized call + recent outputs. Per-call `latency_ms` is always set.
They are **comparison arms only** — Oberih τ=0.7 / weights / S3 / provenance
are unchanged. CPU by default; Apple Silicon MPS when available
(`OBERIH_DETECTOR_DEVICE=cpu|mps` to force).

Install optional deps once:

```bash
pip install 'llm-tester[detectors]'   # transformers + torch
```

## Fetching detector weights

CI never downloads Hugging Face models. For a real run:

```bash
# PromptGuard 2
python -c "
from transformers import AutoModelForSequenceClassification, AutoTokenizer
mid = 'meta-llama/Llama-Prompt-Guard-2-22M'
AutoTokenizer.from_pretrained(mid)
AutoModelForSequenceClassification.from_pretrained(mid)
print('prompt_guard2 ok')
"
export PROMPT_GUARD_MODEL_PATH=…   # optional local override

# ProtectAI DeBERTa v2 (ungated)
python -c "
from transformers import AutoModelForSequenceClassification, AutoTokenizer
mid = 'protectai/deberta-v3-base-prompt-injection-v2'
AutoTokenizer.from_pretrained(mid)
AutoModelForSequenceClassification.from_pretrained(mid)
print('protectai_deberta ok')
"
export PROTECTAI_DEBERTA_MODEL_PATH=…

# Llama Guard 3 (GATED — accept the license; HF_TOKEN read-only is enough)
# huggingface-cli login   # or export HF_TOKEN=…
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
mid = 'meta-llama/Llama-Guard-3-1B'
AutoTokenizer.from_pretrained(mid)
AutoModelForCausalLM.from_pretrained(mid)
print('llama_guard3 ok')
"
export LLAMA_GUARD_MODEL_PATH=…

# Qwen3Guard-Gen 0.6B (Apache-2.0)
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
mid = 'Qwen/Qwen3Guard-Gen-0.6B'
AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
AutoModelForCausalLM.from_pretrained(mid, trust_remote_code=True)
print('qwen3_guard ok')
"
export QWEN3_GUARD_MODEL_PATH=…
```

When weights are unavailable, `available` is `False` and every decision is
ALLOW with `reasons=["<arm>:skipped"]`. For gated Llama Guard, access denial
includes an explicit message to accept the HF license / check `HF_TOKEN`.

## Real-run examples

```bash
llmtest baselines-eval --baseline spotlighting --dataset synthetic --run-dir runs/base_spot
llmtest baselines-eval --baseline prompt_guard2 --dataset synthetic --run-dir runs/base_pg2
llmtest baselines-eval --baseline protectai_deberta --dataset synthetic --run-dir runs/base_deberta
llmtest baselines-eval --baseline llama_guard3 --dataset synthetic --run-dir runs/base_lg3
llmtest baselines-eval --baseline qwen3_guard --dataset synthetic --run-dir runs/base_qwen3g

# Detector-only TPR/FPR + benign FP benchmark (Wilson CIs)
llmtest detector-eval --corpus synthetic --limit 20
# After: python scripts/fetch_external_datasets.py
llmtest detector-eval --corpus bipia --limit 200
llmtest detector-eval --corpus deepset_prompt_injections --limit 200
```
