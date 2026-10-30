# `llmtest mcp-scenario-eval` — mcp-security-research scenarios

Runs Ihor’s MCP scenarios
([mcp-security-research](https://github.com/Ihor-Khomenko/mcp-security-research))
**with and without** MCP-Sentinel, reporting ASR, utility proxies, and
Sentinel detections (first checkpoint, by attack type).

Scenarios cover tool-description poisoning, rug-pull / tool redefinition after
approval, indirect prompt injection, and exfiltration through tool parameters
(plus benign controls). Schema **1.0** exports are converted via the same path
as [`import-mcp-sim`](import-mcp-sim.md).

## Offline / CI (bundled fixture)

```sh
# Default input: llm_tester/data/agent/mcp_sim_import_fixture.jsonl
llmtest mcp-scenario-eval \
  --out-dir /tmp/mcp_eval_mock \
  --defense none --defense sentinel \
  --proxy-mode --json

# Explicit fixture / imported v1 JSONL
llmtest import-mcp-sim \
  --input llm_tester/data/agent/mcp_sim_import_fixture.jsonl \
  --out /tmp/mcp_v1.jsonl

llmtest mcp-scenario-eval \
  --out-dir /tmp/mcp_eval \
  --input /tmp/mcp_v1.jsonl
```

`--proxy-mode` marks the summary as T0/T1/T2-comparable to
`llmtest sentinel-proxy` (same checkpoints; still offline trace replay unless
you attach a live MCP server).

## From a local research checkout

```sh
llmtest mcp-scenario-eval \
  --out-dir runs/mcp_research \
  --research-root ../mcp-security-research \
  --defense none --defense sentinel \
  --lang en
```

## Live S4 (optional)

```sh
export OPENROUTER_API_KEY=...   # env only — never commit
llmtest mcp-scenario-eval \
  --out-dir runs/mcp_s4 \
  --input /tmp/mcp_v1.jsonl \
  --enable-s4 --s4-mock          # offline judge
# For a real judge, attach a provider through guard-eval / research-box code;
# this command keeps CI offline by default (S1–S3, 0 LLM calls).
```

Proxy front-end for a live MCP server:

```sh
llmtest sentinel-proxy --config proxy.yaml -- -- python -m my_mcp_server
```

## Estimated requests per scenario

| Mode | Defense LLM requests (upper bound) |
|------|--------------------------------------|
| `none` | **0** |
| `sentinel` without S4 (default) | **0** (rules / taint / lite intent) |
| `sentinel` with S4 | ≤ **2** judge calls |

Agent / tool LLM calls (if you drive a live agent through the proxy) are
separate — budget them with `--max-requests` on `agent-bench` / your box runner.
A 5-scenario offline matrix `{none, sentinel}` costs **0** OpenRouter requests.

## Metrics in `summary.json`

Per defense: `asr.*`, `utility.utility_benign` (≈ 1 − FPR),
`utility.utility_under_attack`, `detections` (TPR/FPR, `first_detection`,
`by_checkpoint`, `by_attack_type`). Comparison block when both `none` and
`sentinel` are present.
