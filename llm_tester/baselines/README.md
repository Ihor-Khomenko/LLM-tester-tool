# Paper baselines (MCP-Sentinel comparisons)

Pluggable guards implementing the same duck-typed surface as `SentinelGuard`
(`set_user_task`, `reset_session`, `check_tool_registration`, `on_tool_output`,
`check_tool_call`) so they run through `evaluate_traces` / `baselines-eval`.

| Name | Method | Offline CI |
|------|--------|------------|
| `spotlighting` | Datamark / encode tool outputs (Hines et al. 2024) | Yes (pure Python) |
| `prompt_guard2` | Meta Llama-Prompt-Guard-2-22M classifier | Skips if weights missing |
| `progent` | Static privilege / capability allowlist | Yes |
| `task_shield` | LLM ALLOW/BLOCK checker (Ollama `qwen2.5:3b`) | Mock provider in tests |
| `sentinel` | MCP-Sentinel cascade | Yes (S1–S3 lite) |

## Fetching PromptGuard 2 weights

CI never downloads Hugging Face models. For a real run:

```bash
pip install 'transformers>=4.40' 'torch'  # CPU is fine
# Authenticate if the model gate requires it:
#   huggingface-cli login
python -c "
from transformers import AutoModelForSequenceClassification, AutoTokenizer
mid = 'meta-llama/Llama-Prompt-Guard-2-22M'
AutoTokenizer.from_pretrained(mid)
AutoModelForSequenceClassification.from_pretrained(mid)
print('ok')
"
# Optional: point at a local cache
export PROMPT_GUARD_MODEL_PATH=/path/to/Llama-Prompt-Guard-2-22M
```

When weights are unavailable, `PromptGuard2Guard.available` is `False` and every
decision is ALLOW with `reasons=["prompt_guard2:skipped"]` — metrics still
write to the run-dir so `index`/`compare` work.

## Real-run examples

```bash
llmtest baselines-eval --baseline spotlighting --dataset synthetic --run-dir runs/base_spot
llmtest baselines-eval --baseline progent --dataset synthetic --run-dir runs/base_prog
llmtest baselines-eval --baseline prompt_guard2 --dataset synthetic --run-dir runs/base_pg2
llmtest baselines-eval --baseline task_shield --provider ollama --model qwen2.5:3b \
  --dataset synthetic --run-dir runs/base_ts
llmtest baselines-eval --baseline sentinel --dataset synthetic --run-dir runs/base_sent
```
