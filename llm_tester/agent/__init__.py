"""
Evaluation harness for MCP-Sentinel (outside the self-contained sentinel package).

- ``guard-eval`` / offline metrics
- ``agent-bench`` scaffold
- Trace loaders (synthetic, InjecAgent, mcp-security-research JSONL)
- Provider adapter bridging ``llm_tester.providers`` → sentinel ``LLMChat``
"""

from llm_tester.agent.eval import EvalMetrics, evaluate_traces, format_metrics
from llm_tester.agent.traces import TraceStep, resolve_traces

__all__ = [
    "EvalMetrics",
    "evaluate_traces",
    "format_metrics",
    "TraceStep",
    "resolve_traces",
]
