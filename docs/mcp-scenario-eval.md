# `llmtest mcp-scenario-eval` — mcp-security-research scenarios

Runs Ihor’s MCP scenarios
([mcp-security-research](https://github.com/Ihor-Khomenko/mcp-security-research))
**with and without** MCP-Sentinel.

## Modes

| `--mode` | What it measures |
|----------|------------------|
| `replay` (default) | **Detection on recorded traces** (TPR / FPR / miss_rate). Traces already contain the malicious call, so a passthrough defense never “stops” an attack by construction — do **not** call that ASR. |
| `live` | An LLM agent (OpenRouter or `--mock`) runs against **mock MCP tools**. ASR without defense is a **measured** value. Budgeted via `--run-cap` / `--ledger` (same ledger as adaptive-attack). |

## Offline / CI (bundled fixture)

```sh
# Detection-on-traces (default)
llmtest mcp-scenario-eval \
  --out-dir /tmp/mcp_eval_mock \
  --defense none --defense sentinel \
  --lang en --proxy-mode --json

# Multi-variant expansion for meaningful Wilson CIs (min_n default 5)
llmtest mcp-scenario-eval \
  --out-dir /tmp/mcp_eval_var \
  --variants 5 --min-n-for-ci 5 --lang en

# Live ASR with scripted vulnerable agent (CI-safe)
llmtest mcp-scenario-eval \
  --out-dir /tmp/mcp_live_mock \
  --mode live --mock --defense none --defense sentinel
```

## Language consistency (`--lang en|uk`)

User prompts and tool outputs / payloads are **aligned** to `--lang` (default on).
Research vault attack payloads are often Ukrainian even when export `--lang en`
only localises the user prompt; alignment rewrites known payloads / injection
markers so S3 intent is not artificially deflated. Use `--no-align-language` to
disable.

**Correctness (not threshold tuning):** S2/S3 tokenizers are Unicode-aware
(``\w`` + casefold + light UK/EN inflection), so Cyrillic is not forced into
char n-grams. Offline replay scores a call only against tool outputs that
chronologically precede it (a tool’s own result is never a taint source for
that call). The S3 mismatch→score scale (including the ~0.55 lite /
~0.85 side-effect step below ``s3_min_similarity``), cascade thresholds, and
noisy-OR weights are **unchanged** — before/after Ukrainian numbers can be
compared honestly as a correctness fix.

```sh
llmtest mcp-scenario-eval --out-dir runs/mcp_uk \
  --research-root ../mcp-security-research --lang uk

llmtest mcp-scenario-eval --out-dir runs/mcp_en \
  --research-root ../mcp-security-research --lang en
```

## Live ASR (OpenRouter, budgeted)

```sh
# Dry-run: plan cases + request upper bound, zero LLM calls
llmtest mcp-scenario-eval \
  --out-dir /tmp/mcp_live_plan \
  --mode live --dry-run --variants 5 --lang uk \
  --research-root ../mcp-security-research

export OPENROUTER_API_KEY=...   # env only — never commit
export OR_DAILY_CAP=45

llmtest mcp-scenario-eval \
  --out-dir runs/mcp_live \
  --mode live --no-mock \
  --provider openrouter --model meta-llama/llama-3.2-3b-instruct:free \
  --research-root ../mcp-security-research --lang uk \
  --defense none --defense sentinel \
  --run-cap 30 --daily-cap 45 \
  --ledger runs/$(date -I)_openrouter_ledger.json \
  --max-turns 4
```

### Estimated requests per scenario (upper bound)

| Mode | Agent | Defense (no S4) | Defense (S4) | Total |
|------|-------|-----------------|--------------|-------|
| `replay` | 0 | 0 | ≤2 | 0–2 |
| `live` | ≤ **4** (`--max-turns`) | 0 | ≤2 | ≤ **4** (no S4) / ≤ **6** (with S4) |

Early stop / blocks reduce the bill. A 12-scenario `{none, sentinel}` live matrix
without S4 is ≤ `12 × 2 × 4 = 96` agent requests (over two days at 45/day).

## Metrics

**Replay:** `detection.tpr` / `detection.fpr` / `detection.miss_rate`, Wilson CIs
(refused when `n < --min-n-for-ci`), `utility.utility_benign` (benign allow rate).
`utility_under_attack` is **not** reported (it would only be 1 − block rate).

**Live:** measured `asr.asr` (+ Wilson), `utility.utility_benign` from agent task
success.

**Families:** `by_scenario_family` uses the original MSR `attack_class`
(`path_traversal`, `guard_prefix_bypass`, `judge_evasion`, `malicious_write`, …)
instead of the coarsened `attack_type` merge.

**Keys in `results.jsonl`:** `first_detection` (and alias `first_intervention`),
`scenario_family`, `stage_scores` (S1–S4), `checkpoint_scores` (T0/T1/T2),
`reasons`.

## Variant invariants

* Benign variants are **paraphrase-only** (never retarget paths/emails to attack sinks).
* Attack retargets **preserve family** (traversal stays traversal; prefix-bypass
  is not rewritten into `..` paths).
* Every variant fingerprint is **distinct** from the base and siblings.
* `--lang` alignment covers user, payloads, **and** NL call arguments; mismatch
  checks include args.

## Sanitizer (T1)

`_sanitize_injection` uses **general** en/uk pattern classes (system/role
override headers, ignore-previous imperatives, role-claim / compulsion markers,
numbered tool-call directives) — not scenario-literal copy. When a block
starter matches, the **entire contiguous instruction block** is replaced with
`[UNTRUSTED_INSTRUCTION_BLOCK_REMOVED]` (benign lead-in text is kept). Output is
labelled `t1:sanitized` **only** when the text actually changed.
