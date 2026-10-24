# Reproducible run artifacts

Every `llmtest run`, `llmtest guard-eval`, and `llmtest agent-bench` writes a
self-contained run directory suitable for archival in a thesis repository under
`experiments/runs/`.

## Layout

```text
runs/<YYYYMMDD>_<name>/
  config.yaml       # exact CLI args, params, thresholds (no secrets)
  env.json          # git SHAs, Python/package versions, provider/model, hardware, timestamp
  results.jsonl.gz  # per-case records
  metrics.json      # aggregate metrics
  README.md         # short human summary
```

Default path: `runs/<date>_<name>` where `<date>` is UTC `YYYYMMDD` and `<name>`
comes from `--run-name` or is derived from command / dataset / provider / model.

Override with `--run-dir <dir>`. Optional `--extra-sha name=sha` (or a path to
another git repo such as `mcp-security-research`) is recorded under
`env.json → git.extra`.

**Secrets are never written.** Keys matching `api_key`, `token`, `password`,
`secret`, etc. are replaced with `<redacted>`.

## Commands

### Produce a run

```sh
# Jailbreak / refusal eval
llmtest run --dataset harmfulqa --provider ollama --model llama3 --range 10 \
  --run-name baseline-phrase

# MCP-Sentinel offline eval
llmtest guard-eval --dataset synthetic --run-dir experiments/runs/20261024_guard-synthetic

# Pin a sibling research repo SHA
llmtest guard-eval --dataset jsonl --dataset-path scenarios.jsonl \
  --extra-sha mcp-security-research=abc1234 \
  --run-name msr-jsonl

# Agent bench
llmtest agent-bench --provider ollama --model llama3 --without-guard \
  --max-requests 2 --run-name agent-raw
```

During the run, results are appended as plain JSONL (for `--resume`); on
completion they are compressed to `results.jsonl.gz` inside the run directory.

### Index many runs

```sh
llmtest index experiments/runs/
# → experiments/runs/index.csv
```

### Compare metrics (thesis tables)

```sh
llmtest compare runs/20261024_a runs/20261024_b
llmtest compare runs/a runs/b --markdown table.md --csv table.csv
```

Prints a metrics diff table (values + Δ vs the first run). Optional markdown/CSV
outputs are suitable for dissertation tables.

### Report on a gzipped results file

```sh
llmtest report runs/20261024_baseline/results.jsonl.gz
```

## Thesis workflow

1. Run experiments with this tool (locally or on a lab machine).
2. Copy selected `runs/<date>_<name>/` trees into the thesis git repo at
   `experiments/runs/`.
3. Commit `index.csv` (regenerated with `llmtest index`) and comparison tables.
4. Cite `env.json` git SHAs and `config.yaml` thresholds in the methods chapter.

This package’s local `runs/` directory is gitignored; treat the thesis tree as
the long-term store.
