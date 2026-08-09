"""Tests for feature_to_mcherry.report_figures.group_d."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.feature_to_mcherry.report_figures import group_d
from src.feature_to_mcherry.report_figures.config import ReportFiguresConfig


def _make_config(tmp_path: Path, experiments, cohort_sizes=None) -> ReportFiguresConfig:
    return ReportFiguresConfig(
        experiments=experiments,
        scportrait_experiments=[],
        a5_experiments=[],
        cohort_sizes=cohort_sizes or {},
        feature_to_mcherry_dir=str(tmp_path / "feature_to_mcherry"),
        informativeness_dir=str(tmp_path / "morphology_informativeness"),
        data_quality_dir=str(tmp_path / "data_quality"),
        data_root=str(tmp_path / "data_root"),
        output_dir=str(tmp_path / "out"),
        formats=["png"],
    )


def test_figure_d1_renders_without_error(tmp_path: Path) -> None:
    config = _make_config(tmp_path, ["Ew2-1"])

    status = group_d.figure_d1(config)

    assert status.status == "generated"
    assert Path(status.output_paths[0]).exists()


def test_figure_d2_blocked_without_target_distributions(tmp_path: Path) -> None:
    config = _make_config(tmp_path, ["Ew2-1"])

    status = group_d.figure_d2(config)

    assert status.status == "blocked"


def test_figure_d2_generated_when_png_present(tmp_path: Path) -> None:
    config = _make_config(tmp_path, ["Ew2-1"])
    info_dir = Path(config.informativeness_dir) / "Ew2-1"
    figures_dir = info_dir / "figures"
    figures_dir.mkdir(parents=True)

    fig, ax = plt.subplots()
    ax.hist([1, 2, 3])
    fig.savefig(figures_dir / "target_distributions.png")
    plt.close(fig)

    status = group_d.figure_d2(config)

    assert status.status == "generated"
    assert Path(status.output_paths[0]).exists()


def test_figure_d3_blocked_with_reason(tmp_path: Path) -> None:
    config = ReportFiguresConfig(
        experiments=["HD1509"],
        scportrait_experiments=["HD1509"],
        a5_experiments=[],
        feature_to_mcherry_dir=str(tmp_path / "feature_to_mcherry"),
        output_dir=str(tmp_path / "out"),
    )

    status = group_d.figure_d3(config)

    assert status.status == "blocked"
    assert "PCA" in status.note


def test_figure_d4_uses_live_count_when_available(tmp_path: Path) -> None:
    config = _make_config(tmp_path, ["Ew2-1"], cohort_sizes={"Ew2-1": 999999})
    exp_dir = Path(config.data_root) / "Ew2-1 MF5V1 0-72h 06-03-26"
    target_dir = exp_dir / "mcherry_metrics" / "cellpose_sam"
    target_dir.mkdir(parents=True)
    pd.DataFrame({"cell_id": range(50)}).to_csv(
        target_dir / "instance_metrics.csv", index=False
    )

    status = group_d.figure_d4(config)

    assert status.status == "generated"
    assert status.note == ""
    assert Path(status.output_paths[0]).exists()


def test_figure_d4_falls_back_to_configured_count(tmp_path: Path) -> None:
    config = _make_config(tmp_path, ["Ew2-1"], cohort_sizes={"Ew2-1": 445113})
    # No instance_metrics.csv anywhere -> must fall back.

    status = group_d.figure_d4(config)

    assert status.status == "generated"
    assert "Ew2-1" in status.note
    assert Path(status.output_paths[0]).exists()


def test_figure_d4_blocked_without_any_count(tmp_path: Path) -> None:
    config = _make_config(tmp_path, ["Ew2-1"], cohort_sizes={})

    status = group_d.figure_d4(config)

    assert status.status == "blocked"
