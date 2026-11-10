# Changelog

## Unreleased

### Fixed — MCP live harness correctness (Oberih research)

* **Rug-pull / tool-alias none-arm ASR undercount:** `agent_bench.run_scenario` previously advertised mutated `tools_after` descriptions only inside the `if guard` branch. The defense=`none` arm never saw the rug-pulled / aliased tool schemas, so ASR without defense was structurally 0 for those families. Mutated tools are now shown to the agent in **both** arms; the guard still only decides whether to block/pin. Re-scoring older live runs does not change this (it is a harness delivery bug); re-run affected rug_pull / tool-alias none-arm trials for corrected ASR.
* **Homoglyph / zero-width exfil undercount:** injected / strict / stock live success analyzers (`live_success.py`, used by `llmtest rescore-mcp-live` and analysis scripts) now normalize text before matching file names / secrets: NFKC, strip zero-width (U+200B–U+200D, U+2060, U+FEFF), and map common Cyrillic/Greek homoglyphs to Latin. Cases such as v6-hard `homoglyph_zw` that previously missed are now counted. Older results remain valid for other families; re-score with `llmtest rescore-mcp-live` to refresh homoglyph/ZW rows.

Oberih scoring scale, thresholds (`tau=0.7`), weights, and provenance design are unchanged. CI continues to use mocks only (no real API calls).
