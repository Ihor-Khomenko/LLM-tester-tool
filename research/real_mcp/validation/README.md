# Validation reference

Checked-in mock validation JSONL logs are omitted: episode canaries historically
looked like Stripe `sk_live_…` strings and tripped GitHub push protection.
Sandbox secrets now use a `FAKE_API_KEY=canary_…` shape (scoring is still by
random canary hex).

Regenerate locally (no network / no Ollama):

```sh
export OBERIH_TESTS_ROOT="$PWD/research" LLM_TESTER_ROOT="$PWD" PYTHONPATH="$PWD"
cd research/real_mcp && npm ci
python ../runner/lib/real_mcp_driver.py \
  --model mock --llm mock --langs en --arms none,oberih,oberih_gw \
  --run-tag val-mock-all --out validation/mock_all.jsonl
```
