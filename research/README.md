# Oberih research tooling

Offline / local-Ollama job runner and real-MCP sandbox track used for thesis experiments. Lives under `research/` so the core `llm_tester` package stays unchanged (τ=0.7, S3 scale, weights, detection logic).

```
research/
  runner/          # 24/7 LaunchAgent supervisor + planners/drivers/analyzer
  real_mcp/        # Node stdio MCP servers + npm lockfile (no node_modules in git)
```

## Layout vs the Mac box

On the owner's Mac the research workspace is `~/Documents/University/oberih-tests` with sibling dirs `runner/`, `real_mcp/`, frozen `LLM-tester-tool-*` checkouts, and corpora. In this repo the same relative layout is under `research/` (`runner/` + `real_mcp/`); point `LLM_TESTER_ROOT` at the repo root (or set `llm_tester_dir: ..` in a local policy override) so `import llm_tester` resolves.

| Variable | Default | Meaning |
|----------|---------|---------|
| `OBERIH_RUNNER_ROOT` | `research/runner` | Queue, policy, logs, status |
| `OBERIH_TESTS_ROOT` / `RESEARCH_ROOT` | parent of runner | Corpora + `real_mcp/` + `runs_auto/` |
| `LLM_TESTER_ROOT` | `policy.llm_tester_dir` under workspace | PYTHONPATH for `llm_tester` |
| `OBERIH_VENV_PYTHON` | `policy.venv/bin/python` if present | Interpreter for jobs |
| `OBERIH_RUNS_AUTO` | `$RESEARCH_ROOT/runs_auto` | Job output tree |
| `OLLAMA_BIN` / `OBERIH_PATH` | Homebrew + `PATH` | Local Ollama CLI / child PATH |
| `REAL_MCP_ROOT` / `REAL_MCP_NODE` | `$RESEARCH_ROOT/real_mcp` / `node` on PATH | Real-MCP track |

`policy.yaml` keeps the Mac frozen-checkout names (`LLM-tester-tool-fafd500`, `venv-fafd500`) so the existing LaunchAgent keeps working. No API keys belong in the repo or in policy — export them only in the shell/launchd env if a paid provider is ever used (this runner does not call paid APIs).

## Install (macOS LaunchAgent)

1. Place/link this tree so `runner/` and `real_mcp/` sit under your research workspace (or use the in-repo `research/` paths with env overrides).
2. `python3 -m venv …` and `pip install -e ".[oberih]"` on the LLM-tester checkout named in `policy.yaml` (or set `LLM_TESTER_ROOT`).
3. `cd research/real_mcp && npm ci` (pins `@modelcontextprotocol/server-filesystem@2.2.0`).
4. Copy `runner/launchd/ua.ihkh.oberih-runner.plist.example` → `~/Library/LaunchAgents/ua.ihkh.oberih-runner.plist`, edit absolute paths, create `runner/logs/`.
5. `chmod +x research/runner/bin/ctl.sh && research/runner/bin/ctl.sh load`

Guards (disk / swap / battery / Ollama-up) are in `policy.yaml`. `battery_min_pct: 0` disables the battery pause latch.

## `ctl.sh` commands

| Command | Effect |
|---------|--------|
| `status` | Heartbeat / queue / LaunchAgent |
| `pause` / `resume` | Soft pause (current job finishes) / clear pause |
| `stop-now` | Pause + preempt current job back to `pending/` |
| `enqueue '<json>'` | Manual job |
| `drain` / `undrain` | Move pending aside + disable planner / restore |
| `plan [N]` | Dry-run next N planner jobs |
| `analyze` | Rebuild `summary.json` / `.md` / `findings.md` / `proposals.md` |
| `logs [N]` | Tail supervisor log |
| `load` / `unload` / `restart` | launchctl bootstrap / bootout / kickstart |
| `selftest` | Mock jobs only (no Ollama) |

## Real-MCP track

Episodes drive **real** stdio MCP servers:

- `servers/sink_server.mjs` — fake `send_email` / `http_post` (JSONL sink log only; no network)
- `servers/helper_server.mjs` — poisoned / rug-pull / injection tools from per-episode `HELPER_SPEC`
- `@modelcontextprotocol/server-filesystem` — rooted at the episode sandbox only

Arms: `none` (direct) / `oberih` (one Oberih proxy per server) / `oberih_gw` (one proxy in front of `real_mcp_gateway.py`).

After PR #31 the proxy gives backends a secret-free env whitelist (no `SINK_LOG` / `HELPER_*` / `GW_*`). The driver keeps those working by prefixing backends with `/usr/bin/env KEY=VAL …` in the proxy argv — **without** widening the proxy whitelist. A future explicit `backend_env` passthrough on the proxy would be cleaner; until then do not add sandbox keys to the default whitelist.

Mock LLM smoke (CI / laptop, no Ollama):

```sh
export OBERIH_TESTS_ROOT="$PWD/research" LLM_TESTER_ROOT="$PWD" PYTHONPATH="$PWD"
cd research/real_mcp && npm ci
python research/runner/lib/real_mcp_driver.py \
  --model mock --llm mock --langs en --arms none,oberih \
  --ids rmcp-ipi-email-exfil,rmcp-ben-email --run-tag ci-smoke --out /tmp/rmcp_smoke.jsonl
```

## CI

Pytest covers planner rotation, guard/battery latch behaviour, analyzer on a tiny fixture, and (when Node + `node_modules` exist) a scripted-agent real_mcp smoke. No Ollama, no network APIs, no paid providers.
