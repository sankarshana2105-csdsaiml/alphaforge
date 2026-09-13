from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from alphaforge.reporting import load_table, render_figures, write_manifest


def test_missing_artifacts_are_handled_without_rendering(tmp_path: Path) -> None:
    tables, figures = tmp_path / "tables", tmp_path / "figures"
    assert load_table(tables, "missing.csv") is None
    assert render_figures(tables, figures) == []
    assert (tables / "report_manifest.json").is_file()


def test_renderer_creates_only_supported_available_figure(tmp_path: Path) -> None:
    tables, figures = tmp_path / "tables", tmp_path / "figures"
    tables.mkdir()
    pd.DataFrame({"target": ["down", "up"], "count": [3, 7]}).to_csv(
        tables / "target_distribution.csv", index=False
    )
    created = render_figures(tables, figures)
    assert [path.name for path in created] == ["target-class-distribution.png"]
    assert created[0].is_file() and created[0].stat().st_size > 0


def test_manifest_is_deterministic_and_reports_preserve_verdict(tmp_path: Path) -> None:
    tables = tmp_path / "tables"
    tables.mkdir()
    (tables / "z.csv").write_text("x\n1\n", encoding="utf-8")
    (tables / "a.csv").write_text("x\n2\n", encoding="utf-8")
    first = write_manifest(tables).read_text(encoding="utf-8")
    second = write_manifest(tables).read_text(encoding="utf-8")
    assert first == second
    assert json.loads(first)["tables"] == ["a.csv", "z.csv"]
    assert "NO PERSISTENT SIGNAL" in Path("RESEARCH_REPORT.md").read_text(encoding="utf-8")
    assert "NO PERSISTENT SIGNAL" in Path("EXECUTIVE_SUMMARY.md").read_text(encoding="utf-8")


def test_saved_verification_artifacts_render_all_requested_figures() -> None:
    created = render_figures(Path("reports/tables"), Path("reports/figures"))
    assert len(created) == 11
    assert all(path.is_file() and path.stat().st_size > 0 for path in created)
