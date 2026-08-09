"""Tests for feature_to_mcherry.report_figures.a5_paths."""

from __future__ import annotations

from pathlib import Path

from src.feature_to_mcherry.report_figures.a5_paths import resolve_area_p90_csv
from src.feature_to_mcherry.report_figures.config import ReportFiguresConfig


def test_resolves_existing_csv(tmp_path: Path) -> None:
    config = ReportFiguresConfig(a5_area_p90_dir=str(tmp_path))
    csv_path = tmp_path / "HD1509_cellpose_sam.csv"
    csv_path.write_text("area,percentile_90\n")

    resolved = resolve_area_p90_csv("HD1509", "cellpose_sam", config)

    assert resolved == csv_path


def test_returns_none_when_missing(tmp_path: Path) -> None:
    config = ReportFiguresConfig(a5_area_p90_dir=str(tmp_path))

    resolved = resolve_area_p90_csv("HD1509", "scportrait", config)

    assert resolved is None
