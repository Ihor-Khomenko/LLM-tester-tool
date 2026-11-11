# Changelog

## Unreleased

### Added — AgentDojo full-run + white-box adaptive attack (`a_wb`)

* **AgentDojo full-run:** pin `agentdojo==0.1.35` (benchmark **v1.2.2**); suites
  `workspace` / `slack` / `banking` (`travel` optional); attacks
  `important_instructions` + `tool_knowledge`. Arms: `none`, `oberih`,
  `oberih_graded`, AgentDojo natives (`spotlighting_with_delimiting`,
  `repeat_user_prompt`, `tool_filter`, `transformers_pi_detector`), and
  optional `prompt_guard2`. Models via local Ollama or OpenAI-compatible
  routers (`requesty` / `openai_compat`; keys from env only). Sample-size
  (`--limit`) and request budget (`--max-requests`). Metrics: utility,
  utility-under-attack, targeted ASR with Wilson CIs. Resumable JSONL keyed
  by `run_tag` with `code_sha` per row. Optional `--translate-injections-uk`
  (injection texts only). Runner kind `agentdojo`, planner category
  `g_agentdojo` (one job per suite, ETA under 2 h). Docs: `docs/agentdojo.md`.
* **White-box adaptive (`a_wb`):** `llm_tester/attacks/adaptive_whitebox.py` +
  `llmtest adaptive-whitebox`. Attacker reads S1 / τ / S2 / S3-lite from the
  Oberih policy object; offline score gate; only allowed candidates execute
  live. Subfamilies: paraphrase, provenance laundering, score-shaping,
  multi-turn split, homoglyph/ZW. ASR@1/5/20 + budget-vs-ASR, `uk`/`en`.
  Runner family `a_wb` (`c_adaptive_wb` / `run_adaptive_whitebox.py`).
* Oberih defense logic and parameters (τ=0.7, weights, S3 scale, provenance)
  are **unchanged**. CI remains mock-only (no network).

### Added — Opt-in graded-action A/B (`response_mode: graded`)

* **Graded response (default off):** when `response_mode: graded` (defense arm `oberih_graded` / `sentinel_graded`), a cascade **BLOCK** of a local file write with no canary / tainted secret is softened to `Decision.QUARANTINE` (path redirected into `quarantine_dir`, default `/tmp/oberih-quarantine`). Exfil-capable tools (`send`/`http`/`post`/…), destructive ops, and writes containing canary/secret material stay hard-blocked.
* **Scoring honesty:** quarantine is **not** counted as an FP/TP intervention; metrics expose `n_quarantined` / `n_quarantined_benign` / `n_quarantined_attack` separately. τ=0.7, weights, S3 scale, and provenance are unchanged — only the post-decision action.
* **Harness / proxy:** `--defense oberih_graded` in `mcp-scenario-eval`; proxy forwards rewritten quarantine paths and audits `action=quarantine`.

### Added — research runner / real-MCP track in-repo

* `research/runner` — 24/7 LaunchAgent supervisor (planner, guards, live/adaptive/real_mcp drivers, analyzer).
* `research/real_mcp` — Node stdio sink/helper servers + lockfile for the real-MCP arms (`none` / `oberih` / `oberih_gw`).
* Path/env parametrization (`OBERIH_*`, `RESEARCH_ROOT`, `LLM_TESTER_ROOT`, `REAL_MCP_*`, …); frozen-checkout defaults in `policy.yaml`.
* Offline CI: planner rotation, battery/disk guards, analyzer fixture, privacy grep (home paths / personal usernames), optional Node scripted-agent smoke. Proxy secret-free env whitelist unchanged (`/usr/bin/env` argv passthrough for `SINK_LOG` / `HELPER_*` / `GW_*`).
* Docs/defaults are host-generic (LaunchAgent label `org.oberih.runner`); no personal usernames, home paths, or machine IDs in-tree.

### Fixed — Oberih real MCP proxy correctness / security

* **T0 enforcement (default):** flagged tools (poisoned description, hash-pin mismatch / rug-pull on re-`tools/list` after `notifications/tools/list_changed`) are **removed** from the `tools/list` result returned to the client, and `tools/call` to a flagged name returns a JSON-RPC error (`-32001`). Opt out with `t0_mode: warn` (or `--t0-mode warn`) to restore the legacy annotate-only `_sentinel` behaviour.
* **T1 `structuredContent`:** tool-result sanitization/blocking also covers `structuredContent` (e.g. `@modelcontextprotocol/server-filesystem` 2.2.0 text duplicates), not only `content[].text`.
* **Shutdown:** terminate the backend (close stdin) before joining the reader thread — avoids the ~2 s join stall.
* **Stdout locking:** host-facing writes are serialized so host↔backend threads cannot interleave JSON-RPC lines.
* **HTTP correlation:** responses are matched by JSON-RPC `id`; notifications are skipped/recorded instead of being treated as the next reply (no desync).
* **Env hygiene without `mcp`:** backends always get a minimal sanitized environment (PATH/HOME/locale/… whitelist; never `*_KEY` / `*_TOKEN` / `*SECRET*` / `OPENAI_*` / `OPENROUTER_*` / …), mirroring `get_default_environment`.

### Fixed — MCP live harness correctness (Oberih research)

* **v6-hard rug-subtle / alias none-arm not measurable:** `mcp-v6h-rug-subtle-001` and `mcp-v6h-alias-001` never offered a filesystem read tool in the live harness. Subtle rug-pull descriptions say “open ops_restore_bundle.dat” without naming `read_text_file`, and the alias case is tagged `tool_poisoning` (not `rug_pull`), so the old name-in-description + rug_pull-only provisioner added nothing. Target-tool provisioning is now scenario-driven (explicit `required_tools` / success `call`, with filename / mutation heuristics as fallback) and keeps the same reachable tool set on both arms. **Previous none-arm results for these two scenarios were not measurable and should be excluded from n**; re-run after this fix. Std `mcp-rug-pull-tool-desc-001` was already fine after the prior delivery fix.
* **Rug-pull / tool-alias none-arm ASR undercount:** `agent_bench.run_scenario` previously advertised mutated `tools_after` descriptions only inside the `if guard` branch. The defense=`none` arm never saw the rug-pulled / aliased tool schemas, so ASR without defense was structurally 0 for those families. Mutated tools are now shown to the agent in **both** arms; the guard still only decides whether to block/pin. Re-scoring older live runs does not change this (it is a harness delivery bug); re-run affected rug_pull / tool-alias none-arm trials for corrected ASR.
* **Homoglyph / zero-width exfil undercount:** injected / strict / stock live success analyzers (`live_success.py`, used by `llmtest rescore-mcp-live` and analysis scripts) now normalize text before matching file names / secrets: NFKC, strip zero-width (U+200B–U+200D, U+2060, U+FEFF), and map common Cyrillic/Greek homoglyphs to Latin. Cases such as v6-hard `homoglyph_zw` that previously missed are now counted. Older results remain valid for other families; re-score with `llmtest rescore-mcp-live` to refresh homoglyph/ZW rows.

Oberih scoring scale, thresholds (`tau=0.7`), weights, and provenance design are unchanged. CI continues to use mocks only (no real API calls).
