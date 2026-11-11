# AgentDojo full-run (`llmtest agentdojo`)

Standard-benchmark track for paper-comparable results (MELON / PromptArmor /
CaMeL / Progent reviewers). Pins **AgentDojo package 0.1.35** with **benchmark
version `v1.2.2`**.

## Install

```sh
pip install -e '.[agentdojo]'   # agentdojo==0.1.35 (Python ≥3.10)
# Optional PromptGuard 2 gate:
pip install -e '.[baselines-prompt-guard]'
# Weights: set PROMPT_GUARD_MODEL_PATH or allow HF fetch outside CI
```

CI never installs AgentDojo or PromptGuard weights — use `--suite mock`.

## Offline / CI

```sh
llmtest agentdojo --suite mock --defense oberih --json
llmtest agentdojo --full --provider mock --arms none,oberih --limit 2 \
  --out /tmp/adj_full --run-tag ci
```

## Full matrix (live research)

Suites: `workspace`, `slack`, `banking` (`travel` optional via `--include-travel`).
Attacks: `important_instructions`, `tool_knowledge`.
Arms: `none`, `oberih`, `oberih_graded`, AgentDojo natives
(`spotlighting_with_delimiting`, `repeat_user_prompt`, `tool_filter`,
`transformers_pi_detector`), and `prompt_guard2`.

```sh
# Local Ollama (OpenAI-compatible on localhost)
llmtest agentdojo --full --provider ollama --model qwen2.5:7b \
  --out runs/adj_qwen --run-tag adj-qwen-t0

# OpenAI-compatible router (key from env only — never logged)
export REQUESTY_API_KEY=…          # or OPENAI_COMPAT_API_KEY + BASE_URL
llmtest agentdojo --full --provider requesty --model openai/gpt-4o-mini \
  --max-requests 400 --limit 48 \
  --out runs/adj_router --run-tag adj-router
```

Optional: `--translate-injections-uk` glosses **injection texts only** (not user tasks).

## Metrics

Per cell: **utility**, **utility under attack**, **targeted ASR**, each with
**Wilson 95% CIs** (`*_wilson` in metrics / summary).

## Resumable artefacts

`--full` / the research runner write plain JSONL (`results.jsonl`) keyed by
`run_tag` with `code_sha` on every row. Re-running the same tag skips done
`(suite, defense, attack, task_id)` keys.

## Research runner

Planner category `g_agentdojo` (kind `agentdojo`): **one job per suite**, ETA
kept under 2 h (`policy.yaml` → `agentdojo_min_per_suite`). Driver:
`research/runner/lib/run_agentdojo.py`.

```sh
# Preview
python research/runner/lib/planner.py --dry 12 | grep g-adj
```
