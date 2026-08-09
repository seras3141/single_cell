"""Tests for feature_to_mcherry.report_figures.pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.feature_to_mcherry.report_figures.config import ReportFiguresConfig
from src.feature_to_mcherry.report_figures.pipeline import FIGURE_FUNCTIONS, run


def _make_config(tmp_path: Path) -> ReportFiguresConfig:
    return ReportFiguresConfig(
        experiments=["Ew2-1"],
        scportrait_experiments=[],
        a5_experiments=[],
        feature_to_mcherry_dir=str(tmp_path / "feature_to_mcherry"),
        informativeness_dir=str(tmp_path / "morphology_informativeness"),
        data_quality_dir=str(tmp_path / "data_quality"),
        data_root=str(tmp_path / "data_root"),
        output_dir=str(tmp_path / "out"),
        formats=["png"],
    )


def test_run_covers_all_eighteen_brief_figures() -> None:
    assert set(FIGURE_FUNCTIONS) == {
        "A1",
        "A2",
        "A3",
        "A4",
        "A5",
        "A6",
        "B1",
        "B2",
        "B3",
        "B4",
        "C1",
        "C2",
        "C3",
        "C4",
        "D1",
        "D2",
        "D3",
        "D4",
    }


def test_run_with_no_inputs_produces_a_full_manifest_without_crashing(
    tmp_path: Path,
) -> None:
    config = _make_config(tmp_path)

    manifest = run(config)

    assert {e.figure_id for e in manifest.entries} == set(FIGURE_FUNCTIONS)
    # D1 has no data dependency, so it must generate even with nothing else present.
    d1 = next(e for e in manifest.entries if e.figure_id == "D1")
    assert d1.status == "generated"
    # Everything data-dependent should be blocked, not crash the run.
    for status in manifest.entries:
        assert status.status in ("generated", "partial", "blocked")

    assert (Path(config.output_dir) / "manifest.json").exists()
    assert (Path(config.output_dir) / "manifest.md").exists()


def test_run_with_subset_of_figure_ids(tmp_path: Path) -> None:
    config = _make_config(tmp_path)

    manifest = run(config, figure_ids=["D1", "D4"])

    assert {e.figure_id for e in manifest.entries} == {"D1", "D4"}


def test_run_rejects_unknown_figure_id(tmp_path: Path) -> None:
    config = _make_config(tmp_path)

    with pytest.raises(ValueError, match="Unknown figure id"):
        run(config, figure_ids=["Z9"])


def test_run_records_a_blocking_exception_as_blocked_status(
    tmp_path: Path, monkeypatch
) -> None:
    config = _make_config(tmp_path)

    def _boom(_config: ReportFiguresConfig):
        raise RuntimeError("simulated failure")

    monkeypatch.setitem(FIGURE_FUNCTIONS, "D1", _boom)

    manifest = run(config, figure_ids=["D1", "D4"])

    d1 = next(e for e in manifest.entries if e.figure_id == "D1")
    assert d1.status == "blocked"
    assert "simulated failure" in d1.note
    # D4 must still have run despite D1's failure.
    d4 = next(e for e in manifest.entries if e.figure_id == "D4")
    assert d4.status in ("generated", "partial", "blocked")
