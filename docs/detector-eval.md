# Detector baselines & third-party FP evaluation

Comparison defense arms (not Oberih variants) plus a **detector-only** mode that
measures TPR/FPR with Wilson CIs on third-party benign and attack text — the
benign-lookalike false-positive benchmark.

Oberih parameters stay fixed (τ=0.7, weights, S3 scale, provenance). CI never
downloads models or datasets.

## Defense arms

| Arm | Model | Notes |
|-----|-------|-------|
| `prompt_guard2` | `meta-llama/Llama-Prompt-Guard-2-22M` | Existing; threshold 0.5 |
| `protectai_deberta` | `protectai/deberta-v3-base-prompt-injection-v2` | Sequence classifier |
| `llama_guard3` | `meta-llama/Llama-Guard-3-1B` | Gated; clear error if `HF_TOKEN` denied |
| `qwen3_guard` | `Qwen/Qwen3Guard-Gen-0.6B` | Gen labels Safe/Controversial/Unsafe |

Applied at the same T1 (tool outputs → SANITIZE) / T2 (tool calls → BLOCK)
points as PromptGuard 2. Per-call `latency_ms` recorded. Lazy optional deps
(`pip install 'llm-tester[detectors]'`). CPU default; MPS optional
(`OBERIH_DETECTOR_DEVICE`).

Weights fetch: [`llm_tester/baselines/README.md`](../llm_tester/baselines/README.md).

## External datasets

Pinned in [`llm_tester/data/external_datasets/manifest.yaml`](../llm_tester/data/external_datasets/manifest.yaml):

| Dataset | License | What we use |
|---------|---------|-------------|
| [InjecAgent](https://github.com/uiuc-kang-lab/InjecAgent) | MIT | Attack cases + benign twins; agentic traces |
| [BIPIA](https://github.com/microsoft/BIPIA) | MIT (+ CC-BY-SA / MIT components) | Attack texts + email/table/code benign contexts; agentic email compositions. NewsQA/XSum contexts **not** fetched (upstream requires local generation). |
| [deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections) | Apache-2.0 | Labeled injection + benign text |

```sh
python scripts/fetch_external_datasets.py                 # injecagent + bipia
python scripts/fetch_external_datasets.py --deepset       # HF datasets API
python scripts/fetch_external_datasets.py --dry-run --json
```

Cache: `.cache/llm_tester/external/` (gitignored). Checksums verified on fetch.

## Detector-only eval

```sh
llmtest detector-eval --corpus synthetic --limit 20
llmtest detector-eval --corpus bipia --limit 200
llmtest detector-eval --corpus deepset_prompt_injections --limit 200
llmtest detector-eval --corpus injecagent --limit 100 \
  --arms oberih_s1,oberih_s1s2,oberih,prompt_guard2,protectai_deberta,llama_guard3,qwen3_guard
```

Reports per arm × dataset × language: TPR/FPR with Wilson 95% CIs (gated when
`n < 5`) and mean latency. `benign_fp_benchmark` highlights third-party FPR.

## Research runner

- Live arms: supervisor routes external detectors like `prompt_guard2`.
- Kind `detector`: `run_detector_eval.py` (no Ollama agent loop).
- Planner category `i_detector` → job ids `i-det-<corpus>-s<seed>`.

```sh
research/runner/bin/ctl.sh enqueue '{"id":"i-det-bipia-s42","kind":"detector","corpus":"bipia","model":"detector","arms":["oberih_s1","oberih","prompt_guard2","protectai_deberta","llama_guard3","qwen3_guard"],"langs":["en"],"seed":42}'
```

Pre-registered confirmatory analysis (`prereg_stats`) is **unchanged by default**.
