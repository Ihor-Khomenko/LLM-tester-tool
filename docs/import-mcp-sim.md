# `llmtest import-mcp-sim` — mcp-security-research → guard-eval

Converts [mcp-security-research](https://github.com/Ihor-Khomenko/mcp-security-research)
scenario exports (schema **1.0**, from `python -m scenarios.export`) into
[`mcp-sentinel-trace/v1`](mcp-sentinel-trace-format.md) JSONL for
`llmtest guard-eval`.

## Usage

```sh
# From an exported JSONL (CI / offline)
llmtest import-mcp-sim \
  --input llm_tester/data/agent/mcp_sim_import_fixture.jsonl \
  --out /tmp/mcp_sim_v1.jsonl

# From a local research checkout (runs scenarios.export, then converts)
llmtest import-mcp-sim \
  --research-root ../mcp-security-research \
  --out /tmp/mcp_sim_v1.jsonl

llmtest guard-eval --dataset jsonl --dataset-path /tmp/mcp_sim_v1.jsonl
```

`--id` may be repeated to restrict scenario ids. A sibling
`*_import_manifest.json` records counts and the checkpoint mapping summary.

## Checkpoint mapping (T0 / T1 / T2)

| Research export field / event | Sentinel checkpoint | Guard API |
|------------------------------|---------------------|-----------|
| `tools` / `trace[].event=tools_listed` | **T0** | `check_tool_registration(tools)` |
| `tools_after` / `trace[].event=tools_mutated` | **T0** (second pass) | `check_tool_registration(tools_after)` |
| `trace[].event=tool_result` (content, optional `untrusted` / `payload_file`) | **T1** | `on_tool_output(source, content)` |
| First untrusted / payload-bearing `tool_result` | **T1 injection source** | stored in `meta.injection_source` |
| `trace[].event=tool_call` + `arguments` (prefer `attack_step: true`) | **T2** | `check_tool_call({name, arguments})` |

Each converted case includes `meta.checkpoints` with the exercised hooks, e.g.:

```json
"meta": {
  "source": "mcp-security-research",
  "attack_class": "indirect_prompt_injection",
  "injection_source": {
    "tool": "email.read",
    "content_preview": "IMPORTANT!!! Ignore all previous…",
    "checkpoint": "T1"
  },
  "call_arguments": {"iban": "DE89…", "amount": "5000"},
  "checkpoints": {
    "T0": ["tools_listed→check_tool_registration"],
    "T1": ["tool_result[email.read]→on_tool_output"],
    "T2": ["tool_call[bank.transfer](args=['amount','iban'])→check_tool_call"],
    "exercised": ["T0", "T1", "T2"]
  }
}
```

## Labels

| Export | v1 `label` / `attack_type` |
|--------|----------------------------|
| `attack_class: benign` or `label: benign` | `benign` / `benign` |
| Other classes (`indirect_prompt_injection`, `tool_poisoning`, `rug_pull`, …) | `attack` / mapped type (see `ATTACK_CLASS_MAP` in `llm_tester/agent/mcpsr.py`) |
| `labels.guard_should_block` | copied into `meta.guard_should_block` for analysis |

## Fixture

Bundled offline fixture (no network):
`llm_tester/data/agent/mcp_sim_import_fixture.jsonl`
(schema 1.0 attack + benign examples covering T0/T1/T2).

`guard-eval --dataset jsonl` also auto-converts schema 1.0 on the fly; this
command is the explicit importer that materialises v1 + checkpoint metadata
for paper pipelines and CI.
