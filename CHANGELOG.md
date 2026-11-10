# Changelog

## Unreleased

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
