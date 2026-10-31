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

Live mode runs an agent against **each scenario’s own tools** (not a fixed
global tool list). Scripted tool outputs are a **FIFO / path-aware queue**, so
two `read_text_file` calls each get their own planted content (injection then
secret). Attack success is judged by the scenario’s **expected sink call**
(name **and** path/to/iban/… args) — a benign read of `report_ok.txt` is not
success. Families that plant an injection must deliver it into an observation
(or rug-pull mutated description) before the attack step.

**Loop order:** cases are the outer loop, defenses the inner loop, so a
`--run-cap` cut leaves paired `none`/`sentinel` rows for completed cases.
`--order alternate` interleaves attack and benign (frozen into
`progress.json`). `--resume` continues from `out-dir/progress.json` +
`results.jsonl` and skips finished `(id, defense)` pairs.

HTTP **429** responses are counted in the request ledger (`n_429`) and those
cases are logged with `status: rate_limited` (kept separate from `error`).

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
  --order alternate --resume \
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

**`risk_score` (case-level):** maximum `risk_score` across all checkpoint
decisions in the replay — *not* the last decision alone. Chronological replay
may emit a trailing T1 tool-output `allow` at 0.0 after a T2 block; using the
last decision would zero out blocked attacks. Detection (TPR/FPR) still uses
`intervened` / `blocked`; this field is reporting only.

**`checkpoint_scores`:** per checkpoint, aggregated over multiple decisions:
`risk_score` = max risk at that checkpoint; `decision` = strongest action
(`block` > `sanitize` > `allow`); `decisions` lists each raw decision. A later
T1 `allow` must not hide an earlier T1 `sanitize`.

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
