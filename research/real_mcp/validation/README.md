# Validation reference

Checked-in mock validation JSONL/logs are omitted so the repo stays free of
machine-specific artefacts and secret-scanner false positives.

Regenerate locally (no network / no Ollama):

```sh
export OBERIH_TESTS_ROOT="$PWD/research" LLM_TESTER_ROOT="$PWD" PYTHONPATH="$PWD"
cd research/real_mcp && npm ci
python ../runner/lib/real_mcp_driver.py \
  --model mock --llm mock --langs en --arms none,oberih,oberih_gw \
  --run-tag val-mock-all --out validation/mock_all.jsonl
```

Sandbox secret files use `FAKE_API_KEY=canary_…` placeholders (scoring is by
random canary hex).
