"""AgentDojo full-suite driver for the research runner (kind=agentdojo).

One job = one suite (workspace|slack|banking|travel). Arms and attacks come
from the job JSON. Resumable JSONL keyed by ``run_tag`` with ``code_sha`` per
row. Uses local Ollama or an OpenAI-compatible router from env (never logs keys).

CI / dry path: ``--llm mock`` (scripted AgentDojo LLM or offline mock suite).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--suite", required=True, help="workspace|slack|banking|travel|mock")
ap.add_argument("--model", default="qwen2.5:7b")
ap.add_argument("--llm", default="ollama", help="ollama|requesty|openai_compat|mock")
ap.add_argument("--base-url", default=None)
ap.add_argument(
    "--arms",
    default="none,oberih,oberih_graded,spotlighting_with_delimiting,repeat_user_prompt,tool_filter,transformers_pi_detector,prompt_guard2,protectai_deberta,llama_guard3,qwen3_guard",
)
ap.add_argument(
    "--attacks",
    default="important_instructions,tool_knowledge",
)
ap.add_argument("--benchmark-version", default="v1.2.2")
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--max-requests", type=int, default=None)
ap.add_argument("--temperature", type=float, default=0.0)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--run-tag", default="")
ap.add_argument("--out", required=True, help="results JSONL path")
ap.add_argument("--translate-injections-uk", action="store_true")
ap.add_argument("--include-travel", action="store_true")
a = ap.parse_args()

# Never echo secrets.
for k in list(os.environ):
    ku = k.upper()
    if any(s in ku for s in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
        # Keep the var for the client; do not print.
        pass

from llm_tester.agent.agentdojo_fullrun import (  # noqa: E402
    FullRunConfig,
    run_full_matrix,
)
from llm_tester.agent.provider_adapter import build_llm_chat  # noqa: E402

out_path = Path(a.out)
out_dir = out_path.parent
out_dir.mkdir(parents=True, exist_ok=True)

arms = [x.strip() for x in a.arms.split(",") if x.strip()]
attacks = [x.strip() for x in a.attacks.split(",") if x.strip()]
provider = (a.llm or "mock").lower().strip()

llm = None
if provider not in ("mock", "none", "scripted", "ci"):
    llm = build_llm_chat(
        provider,
        a.model,
        base_url=a.base_url,
        temperature=a.temperature,
        seed=a.seed,
        max_retries=0,
    )

if a.suite == "mock" or provider == "mock":
    # Prefer full matrix when AgentDojo is present; else offline mock suite.
    cfg = FullRunConfig(
        suites=[a.suite] if a.suite != "mock" else ["workspace"],
        attacks=attacks,
        arms=arms,
        benchmark_version=a.benchmark_version,
        provider_name="mock",
        model=a.model,
        temperature=a.temperature,
        seed=a.seed,
        limit=a.limit,
        max_requests=a.max_requests,
        run_tag=a.run_tag,
        translate_injections_uk=bool(a.translate_injections_uk),
        include_travel=bool(a.include_travel),
    )
    # Write into out_dir so results.jsonl is the canonical name; then copy/link
    # to the supervisor-requested --out path if different.
    summary = run_full_matrix(out_dir, config=cfg, llm_provider=None)
else:
    cfg = FullRunConfig(
        suites=[a.suite],
        attacks=attacks,
        arms=arms,
        benchmark_version=a.benchmark_version,
        provider_name=provider,
        model=a.model,
        temperature=a.temperature,
        seed=a.seed,
        limit=a.limit,
        max_requests=a.max_requests,
        run_tag=a.run_tag,
        translate_injections_uk=bool(a.translate_injections_uk),
        include_travel=False,
    )
    summary = run_full_matrix(out_dir, config=cfg, llm_provider=llm)

# Supervisor expects --out to be the JSONL file.
src = out_dir / "results.jsonl"
if src.exists() and src.resolve() != out_path.resolve():
    out_path.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
elif not out_path.exists() and src.exists():
    pass  # same path via parent
(out_dir / "summary_agentdojo.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n",
    encoding="utf-8",
)
print(
    json.dumps(
        {
            "ok": True,
            "suite": a.suite,
            "run_tag": a.run_tag,
            "requests_used": summary.get("requests_used"),
            "results": str(out_path),
        }
    ),
    flush=True,
)
