"""Figure smoke tests — offline, B/W thesis outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_tester.cli import main
from llm_tester.figures.generate import generate_figures
from llm_tester.figures.style import MAX_WIDTH_MM, apply_thesis_style, figure_size, mm_to_inches

FIXTURES = Path(__file__).parent / "fixtures" / "runs"

matplotlib = pytest.importorskip("matplotlib")


def test_thesis_style_font_and_width():
    font = apply_thesis_style("uk")
    assert font  # some serif family resolved
    w_in, h_in = figure_size()
    assert w_in <= mm_to_inches(MAX_WIDTH_MM) + 1e-9
    assert h_in > 0


def test_generate_figures_smoke(tmp_path: Path):
    out = tmp_path / "figs"
    manifest = generate_figures(
        [FIXTURES / "run_a", FIXTURES / "run_b", FIXTURES / "run_ablation_s1"],
        out,
        lang="uk",
    )
    assert manifest["figures"], "expected at least one figure"
    assert manifest["tables"]
    assert (out / "figures_manifest.json").exists()
    assert (out / "captions.md").exists()
    assert (out / "table_metrics_overview.md").exists()

    for fig in manifest["figures"]:
        png = Path(fig["file_png"])
        svg = Path(fig["file_svg"])
        assert png.exists() and png.stat().st_size > 0
        assert svg.exists() and svg.stat().st_size > 0
        assert fig["suggested_caption"].startswith("Рисунок 4.")
        # Rough width check via PNG DPI metadata when available
        from PIL import Image

        im = Image.open(png)
        # width in mm at 300 dpi
        dpi = im.info.get("dpi", (300, 300))[0] or 300
        width_mm = im.size[0] / dpi * 25.4
        # Allow small overshoot from bbox_inches=tight labels
        assert width_mm <= MAX_WIDTH_MM + 25, f"{png.name} width {width_mm:.1f} mm"

    table_text = (out / "table_metrics_overview.md").read_text(encoding="utf-8")
    assert table_text.startswith("Таблиця 4.")

    data = json.loads((out / "figures_manifest.json").read_text(encoding="utf-8"))
    assert data["max_width_mm"] == 158
    assert data["dpi"] == 300


def test_generate_figures_english_captions(tmp_path: Path):
    out = tmp_path / "figs_en"
    manifest = generate_figures([FIXTURES / "run_a"], out, lang="en")
    assert any(f["suggested_caption"].startswith("Figure 4.") for f in manifest["figures"])


def test_cli_figures(tmp_path: Path):
    out = tmp_path / "out_figs"
    rc = main(
        [
            "figures",
            str(FIXTURES / "run_a"),
            str(FIXTURES / "run_b"),
            "--out",
            str(out),
            "--lang",
            "uk",
        ]
    )
    assert rc == 0
    assert (out / "figures_manifest.json").exists()
    assert list(out.glob("fig_*.png"))
    assert list(out.glob("fig_*.svg"))
