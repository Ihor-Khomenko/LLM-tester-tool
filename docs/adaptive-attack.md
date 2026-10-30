# `llmtest adaptive-attack` — Sentinel-aware adaptive harness (R-A1)

Grey-box attacker that knows MCP-Sentinel’s stage design (S1–S4, noisy-OR,
public prompt shapes) but **not** thresholds or the judge model. The attacker
controls only untrusted tool outputs and sees an end-to-end outcome bit
(success / blocked) for up to **k** tries.

## Attack families

| Family | Role |
|--------|------|
| `important_instructions` | Static AgentDojo-style control |
| `a_para` | Paraphrase / obfuscation (strip triggers, uk/en mix, base64) |
| `a_split` | Split payload across two tool outputs |
| `a_judge` | Benign / authority framing aimed at S4 |
| `a_iter` | Attacker-LLM refinement loop (k=3 pilot, k=5 full); early stop |

## Metrics

Per defense: **ASR@1**, **ASR@k** (Wilson 95% CI), **stopped k-of-n** (of jobs
that succeed with no defense, how many Sentinel still stops), **McNemar** on
paired job ids, plus attacker / agent / defense request counters.

## Offline / CI

```sh
llmtest adaptive-attack \
  --out-dir /tmp/adaptive_mock \
  --family a_para --family a_iter \
  --defense none --defense sentinel \
  --k 3 --limit 3 --mock --json
```

Uses a scripted defense (blocks classic trigger phrases). No network, no API keys.

## Live research (budget-aware)

Real AgentDojo wiring is attached in library code via
`llm_tester.attacks.adaptive.run_adaptive_job(..., run_try=...)`.
CLI refuses `--no-mock` so CI cannot accidentally burn quota.

```sh
# Ledger + hard caps (OpenRouter free tier ≈ 45 req/day shared)
export OPENROUTER_API_KEY=...   # never commit; read from env only
export OR_DAILY_CAP=45
export LLMTEST_LEDGER_DIR=runs

llmtest adaptive-attack \
  --out-dir runs/adaptive_pilot \
  --family a_iter --defense none --defense sentinel \
  --k 3 --limit 2 \
  --run-cap 30 --daily-cap 45 \
  --ledger runs/$(date -I)_openrouter_ledger.json
```

### Estimated requests per adaptive job (upper bound)

Plan assumptions: ~3 agent + ~2 Sentinel S4 + 1 attacker LLM per refinement try.
`a_iter` first try uses a template seed (0 attacker calls).

| Setting | Upper bound |
|---------|-------------|
| `a_iter`, k=3, +Sentinel | ≤ **17** (`9` agent + `6` defense + `2` attacker) |
| `a_iter`, k=5, +Sentinel | ≤ **29** |
| Static family (`a_para` / `a_split` / `a_judge`), +Sentinel | ≤ **5** |
| Any family, `defense=none` | agent only (≤ `3×tries`) |

Early stop on first success lowers the bill. Suggested daily allocation: ~30 of
the shared 45 for adaptive work (~2 `a_iter` jobs or ~5 static-family jobs).

Gate (from the research plan): start adaptive work only after the target model
shows a no-defense static ASR of at least ~0.3.

## Artefacts

Under `--out-dir`: `progress.json`, `results.jsonl`, `summary.json`
(same conventions as the daily research box runners).
