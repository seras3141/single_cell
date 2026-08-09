"""Tests for feature_to_mcherry.report_figures.group_c."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.feature_to_mcherry.report_figures import group_c
from src.feature_to_mcherry.report_figures.config import ReportFiguresConfig

EXTREME_COLUMNS = [
    "source",
    "value_column",
    "group_type",
    "group_value",
    "n",
    "n_extreme",
    "extreme_rate",
    "overall_rate",
    "enrichment",
]


def _make_config(tmp_path: Path, experiments) -> ReportFiguresConfig:
    return ReportFiguresConfig(
        experiments=experiments,
        scportrait_experiments=[],
        a5_experiments=[],
        feature_to_mcherry_dir=str(tmp_path / "feature_to_mcherry"),
        informativeness_dir=str(tmp_path / "morphology_informativeness"),
        data_quality_dir=str(tmp_path / "data_quality"),
        data_root=str(tmp_path / "data_root"),
        output_dir=str(tmp_path / "out"),
        formats=["png"],
    )


def _write_extreme_report(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=EXTREME_COLUMNS).to_csv(path, index=False)


def test_figure_c1_generated_for_available_experiments(tmp_path: Path) -> None:
    config = _make_config(tmp_path, ["Ew2-1", "Ew2-2"])
    rows = []
    for exp in ("Ew2-1", "Ew2-2"):
        for tp, enrichment in [(1, 0.5), (11, 3.2), (21, 1.6)]:
            rows.append(
                [exp, "percentile_95", "timepoint", tp, 100, 5, 0.05, 0.02, enrichment]
            )
    _write_extreme_report(
        Path(config.data_quality_dir) / "Ew2-1_Ew2-2" / "extreme_value_report.csv", rows
    )

    status = group_c.figure_c1(config)

    assert status.status == "generated"
    assert Path(status.output_paths[0]).exists()


def test_figure_c1_blocked_when_no_data_quality_csv(tmp_path: Path) -> None:
    config = _make_config(tmp_path, ["Ew2-1"])

    status = group_c.figure_c1(config)

    assert status.status == "blocked"


def test_figure_c2_resolves_plate_join_and_skips_bad_well(tmp_path: Path) -> None:
    config = _make_config(tmp_path, ["Ew2-1"])
    rows = [
        ["Ew2-1", "area", "sample_id", "C02", 100, 5, 0.05, 0.02, 1.2],
        ["Ew2-1", "area", "sample_id", "ZZ99", 100, 5, 0.05, 0.02, 1.2],
    ]
    _write_extreme_report(
        Path(config.data_quality_dir) / "Ew2-1" / "extreme_value_report.csv", rows
    )

    status = group_c.figure_c2(config)

    assert status.status == "generated"
    assert Path(status.output_paths[0]).exists()


def test_figure_c4_partial_when_only_counts_available(tmp_path: Path) -> None:
    config = _make_config(tmp_path, ["Ew2-1"])
    rows = [
        ["Ew2-1", "area", "z_index", z, 1000 - z * 10, 5, 0.05, 0.02, 1.0]
        for z in range(1, 5)
    ]
    _write_extreme_report(
        Path(config.data_quality_dir) / "Ew2-1" / "extreme_value_report.csv", rows
    )
    # No feature_dir under data_root -> mean-area panel can't be built for Ew2-1.

    status = group_c.figure_c4(config)

    assert status.status == "partial"
    assert status.missing == ["Ew2-1"]
    assert Path(status.output_paths[0]).exists()


def test_figure_c4_uses_raw_feature_csvs_for_mean_area(tmp_path: Path) -> None:
    config = _make_config(tmp_path, ["Ew2-1"])
    feature_dir = (
        Path(config.data_root)
        / "Ew2-1 MF5V1 0-72h 06-03-26"
        / "inference_tracked"
        / "cellpose_sam"
        / "features_incarta"
        / "split_data"
    )
    feature_dir.mkdir(parents=True)
    pd.DataFrame({"area": [10, 20, 30], "z_index": [1, 1, 2]}).to_csv(
        feature_dir / "slice1.csv", index=False
    )

    status = group_c.figure_c4(config)

    # No data_quality/extreme_value_report.csv exists for Ew2-1, so the count
    # panel has no series for it even though the area panel does.
    assert status.status == "partial"
    assert status.missing == ["Ew2-1"]
    assert Path(status.output_paths[0]).exists()
