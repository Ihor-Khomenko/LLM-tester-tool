"""Orchestrate thesis figures / tables / manifest from one or more run dirs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from llm_tester.artifacts.index import summarise_run
from llm_tester.artifacts.run_dir import load_metrics, load_results
from llm_tester.figures import charts


def _caption(lang: str, kind: str, n: int, detail: str = "") -> str:
    if (lang or "uk").lower().startswith("en"):
        prefix = f"Figure 4.{n}"
        templates = {
            "asr": "Attack success rate with and without guard",
            "ablation": "TPR, FPR and F1 across cascade ablations",
            "latency_box": "Latency distribution (box plot; p50/p95 marked)",
            "latency_cdf": "Latency empirical CDF (p50/p95 marked)",
            "escalation": "Escalation rate by run / ablation",
            "roc": "ROC curve for risk-score threshold sweep",
            "compare": "Run-over-run metrics comparison",
        }
        base = templates.get(kind, kind)
        return f"{prefix} – {base}{(': ' + detail) if detail else ''}"
    prefix = f"Рисунок 4.{n}"
    templates = {
        "asr": "ASR з захистом і без захисту",
        "ablation": "TPR, FPR та F1 для абляції стадій каскаду",
        "latency_box": "Розподіл затримки (boxplot; позначено p50/p95)",
        "latency_cdf": "Емпірична CDF затримки (позначено p50/p95)",
        "escalation": "Частка ескалацій за прогоном / абляцією",
        "roc": "ROC-крива для перебору порогу оцінки ризику",
        "compare": "Порівняння метрик між прогонами",
    }
    base = templates.get(kind, kind)
    return f"{prefix} – {base}{(': ' + detail) if detail else ''}"


def _table_caption(lang: str, n: int, detail: str) -> str:
    if (lang or "uk").lower().startswith("en"):
        return f"Table 4.{n} – {detail}"
    return f"Таблиця 4.{n} – {detail}"


def _latencies_from_results(results: List[Dict[str, Any]]) -> Tuple[List[float], str]:
    ms: List[float] = []
    secs: List[float] = []
    for r in results:
        if isinstance(r.get("latency_ms"), (int, float)):
            ms.append(float(r["latency_ms"]))
        elif isinstance(r.get("latency_s"), (int, float)):
            secs.append(float(r["latency_s"]))
        # guard-eval style nested
        elif isinstance(r.get("latency_ms_total"), (int, float)):
            ms.append(float(r["latency_ms_total"]))
    if ms:
        return ms, "ms"
    return secs, "s"


def _scores_and_labels(results: List[Dict[str, Any]]) -> Tuple[List[float], List[int]]:
    scores: List[float] = []
    labels: List[int] = []
    for r in results:
        score = r.get("risk_score")
        if score is None:
            score = r.get("score")
        if score is None and isinstance(r.get("input_scores"), dict):
            # average numeric scanner scores if present
            nums = [float(v) for v in r["input_scores"].values() if isinstance(v, (int, float))]
            score = sum(nums) / len(nums) if nums else None
        lab = r.get("label")
        is_attack = None
        if isinstance(lab, str):
            is_attack = lab.lower() in ("attack", "malicious", "adversarial", "1", "true")
        elif r.get("attack_success") is True and r.get("guard_blocked") is not True:
            # fallback — not ideal for ROC
            is_attack = True
        elif "is_attack" in r:
            is_attack = bool(r["is_attack"])
        if score is None or is_attack is None:
            continue
        scores.append(float(score))
        labels.append(1 if is_attack else 0)
    return scores, labels


def _run_row(run_dir: Path) -> Dict[str, Any]:
    row = summarise_run(run_dir)
    metrics = {}
    try:
        metrics = load_metrics(run_dir)
    except FileNotFoundError:
        pass
    row.update({k: v for k, v in metrics.items() if k not in row or row[k] in ("", None)})
    row["label"] = row.get("run_id") or run_dir.name
    row["run"] = run_dir.name
    row["path"] = str(run_dir)
    return row


def generate_figures(
    run_paths: Sequence[Union[str, Path]],
    out_dir: Union[str, Path],
    *,
    lang: str = "uk",
    figure_start: int = 1,
    table_start: int = 1,
) -> Dict[str, Any]:
    """
    Emit PNG+SVG figures, markdown tables with suggested captions, and a manifest.

    Returns manifest dict.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs = [Path(p) for p in run_paths]
    rows = [_run_row(p) for p in runs]

    manifest: Dict[str, Any] = {
        "lang": lang,
        "runs": [str(p) for p in runs],
        "figures": [],
        "tables": [],
        "font_note": "Times New Roman with fallback Liberation Serif / DejaVu Serif",
        "max_width_mm": 158,
        "dpi": 300,
    }
    fig_n = figure_start
    tab_n = table_start
    table_md_parts: List[str] = []

    # --- ASR with vs without ---
    asr_rows = [
        r
        for r in rows
        if r.get("asr_with_guard") is not None or r.get("asr_without_guard") is not None
        or r.get("asr") is not None
        or r.get("attack_success_rate") is not None
    ]
    if asr_rows:
        for r in asr_rows:
            if r.get("asr") is None and r.get("attack_success_rate") is not None:
                r["asr"] = r["attack_success_rate"]
            if r.get("asr_with_guard") is None and r.get("asr") is not None:
                # single-condition run: still plotable as with-guard bar
                if r.get("guard") not in ("", "none", None):
                    r["asr_with_guard"] = r["asr"]
                else:
                    r["asr_without_guard"] = r.get("asr_without_guard", r["asr"])
                    r["asr_with_guard"] = r.get("asr_with_guard", r["asr"])
        base = out / "fig_asr_with_without"
        png, svg = charts.plot_asr_bars(asr_rows, base, lang=lang)
        cap = _caption(lang, "asr", fig_n)
        manifest["figures"].append(
            {
                "id": f"fig_4_{fig_n}",
                "file_png": str(png),
                "file_svg": str(svg),
                "suggested_caption": cap,
            }
        )
        fig_n += 1

    # --- Stage ablation (needs multiple runs or by_attack ignored) ---
    ablation_rows = [
        r
        for r in rows
        if r.get("tpr") is not None or r.get("fpr") is not None or r.get("f1") is not None
    ]
    if len(ablation_rows) >= 1:
        base = out / "fig_stage_ablation"
        png, svg = charts.plot_stage_ablation(ablation_rows, base, lang=lang)
        cap = _caption(lang, "ablation", fig_n)
        manifest["figures"].append(
            {
                "id": f"fig_4_{fig_n}",
                "file_png": str(png),
                "file_svg": str(svg),
                "suggested_caption": cap,
            }
        )
        fig_n += 1

    # --- Latency distributions ---
    latency_series: Dict[str, List[float]] = {}
    unit = "ms"
    for run_dir, row in zip(runs, rows):
        try:
            results = load_results(run_dir)
        except FileNotFoundError:
            results = []
        lats, u = _latencies_from_results(results)
        if not lats:
            # fall back to metrics-only synthetic single points
            m = row
            if m.get("latency_p50_ms") is not None:
                # approximate series from percentiles if full dist missing
                pts = []
                for key in ("latency_p50_ms", "latency_mean_ms", "latency_p95_ms"):
                    if isinstance(m.get(key), (int, float)):
                        pts.append(float(m[key]))
                lats = pts
                u = "ms"
            elif m.get("mean_latency_s") is not None:
                lats = [float(m["mean_latency_s"])]
                u = "s"
        if lats:
            latency_series[row["run"]] = lats
            unit = u
    if latency_series:
        base = out / "fig_latency_box"
        png, svg = charts.plot_latency_box(latency_series, base, lang=lang, unit=unit)
        cap = _caption(lang, "latency_box", fig_n)
        manifest["figures"].append(
            {
                "id": f"fig_4_{fig_n}",
                "file_png": str(png),
                "file_svg": str(svg),
                "suggested_caption": cap,
            }
        )
        fig_n += 1
        base = out / "fig_latency_cdf"
        png, svg = charts.plot_latency_cdf(latency_series, base, lang=lang, unit=unit)
        cap = _caption(lang, "latency_cdf", fig_n)
        manifest["figures"].append(
            {
                "id": f"fig_4_{fig_n}",
                "file_png": str(png),
                "file_svg": str(svg),
                "suggested_caption": cap,
            }
        )
        fig_n += 1

    # --- Escalation ---
    esc_rows = [r for r in rows if r.get("escalation_rate") is not None]
    if esc_rows:
        base = out / "fig_escalation"
        png, svg = charts.plot_escalation(esc_rows, base, lang=lang)
        cap = _caption(lang, "escalation", fig_n)
        manifest["figures"].append(
            {
                "id": f"fig_4_{fig_n}",
                "file_png": str(png),
                "file_svg": str(svg),
                "suggested_caption": cap,
            }
        )
        fig_n += 1

    # --- ROC / threshold sweep ---
    for run_dir, row in zip(runs, rows):
        try:
            results = load_results(run_dir)
        except FileNotFoundError:
            continue
        scores, labs = _scores_and_labels(results)
        if len(scores) >= 2:
            base = out / f"fig_roc_{row['run']}"
            saved = charts.plot_roc_threshold(scores, labs, base, lang=lang)
            if saved:
                png, svg = saved
                cap = _caption(lang, "roc", fig_n, detail=row["run"])
                manifest["figures"].append(
                    {
                        "id": f"fig_4_{fig_n}",
                        "file_png": str(png),
                        "file_svg": str(svg),
                        "suggested_caption": cap,
                    }
                )
                fig_n += 1

    # --- Run-over-run comparison ---
    if len(rows) >= 2:
        metric_keys = [
            k
            for k in ("tpr", "fpr", "f1", "asr_with_guard", "escalation_rate", "attack_success_rate")
            if any(r.get(k) is not None for r in rows)
        ]
        if metric_keys:
            base = out / "fig_run_comparison"
            png, svg = charts.plot_run_comparison(rows, metric_keys, base, lang=lang)
            cap = _caption(lang, "compare", fig_n)
            manifest["figures"].append(
                {
                    "id": f"fig_4_{fig_n}",
                    "file_png": str(png),
                    "file_svg": str(svg),
                    "suggested_caption": cap,
                }
            )
            fig_n += 1

    # --- Tables (markdown) ---
    # Metrics overview
    overview_cols = [
        "run",
        "command",
        "dataset",
        "model",
        "ablation",
        "n",
        "tpr",
        "fpr",
        "f1",
        "asr_with_guard",
        "asr_without_guard",
        "escalation_rate",
        "attack_success_rate",
        "refusal_rate",
    ]
    overview_cap = _table_caption(
        lang,
        tab_n,
        "Зведення метрик експериментів" if not lang.lower().startswith("en") else "Experiment metrics overview",
    )
    overview_path = out / "table_metrics_overview.md"
    _write_md_table(overview_path, rows, overview_cols, caption=overview_cap)
    manifest["tables"].append(
        {
            "id": f"tab_4_{tab_n}",
            "file": str(overview_path),
            "suggested_caption": overview_cap,
        }
    )
    table_md_parts.append(overview_path.read_text(encoding="utf-8"))
    tab_n += 1

    # CSV twin for spreadsheet paste
    csv_path = out / "table_metrics_overview.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=overview_cols, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c, "") for c in overview_cols})

    # Captions file
    captions_path = out / "captions.md"
    cap_lines = ["# Suggested captions / Рекомендовані підписи", ""]
    for fig in manifest["figures"]:
        cap_lines.append(f"- **{fig['id']}** (`{Path(fig['file_png']).name}`): {fig['suggested_caption']}")
    cap_lines.append("")
    for tab in manifest["tables"]:
        cap_lines.append(f"- **{tab['id']}** (`{Path(tab['file']).name}`): {tab['suggested_caption']}")
    captions_path.write_text("\n".join(cap_lines) + "\n", encoding="utf-8")

    # Manifest
    manifest_path = out / "figures_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    manifest["manifest_path"] = str(manifest_path)
    manifest["captions_path"] = str(captions_path)
    return manifest


def _write_md_table(
    path: Path,
    rows: Sequence[Dict[str, Any]],
    columns: Sequence[str],
    *,
    caption: str,
) -> None:
    # Caption above table (Ukrainian thesis rule for tables).
    lines = [caption, "", "| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for r in rows:
        cells = []
        for c in columns:
            v = r.get(c, "")
            if isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append("" if v is None else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
