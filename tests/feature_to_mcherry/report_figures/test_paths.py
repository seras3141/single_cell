"""Tests for feature_to_mcherry.report_figures.paths."""

from __future__ import annotations

from pathlib import Path

from src.feature_to_mcherry.report_figures.config import ReportFiguresConfig
from src.feature_to_mcherry.report_figures.paths import (
    missing_fields,
    resolve_experiment_paths,
)


def _make_config(tmp_path: Path) -> ReportFiguresConfig:
    return ReportFiguresConfig(
        feature_to_mcherry_dir=str(tmp_path / "feature_to_mcherry"),
        informativeness_dir=str(tmp_path / "morphology_informativeness"),
        data_quality_dir=str(tmp_path / "data_quality"),
        data_root=str(tmp_path / "data_root"),
    )


def test_resolves_subdirectory_naming_convention(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    exp_dir = Path(config.feature_to_mcherry_dir) / "Ew2-1"
    exp_dir.mkdir(parents=True)
    (exp_dir / "baseline_ladder.csv").write_text("model,target\n")

    paths = resolve_experiment_paths("Ew2-1", config)

    assert paths.baseline_ladder_csv == exp_dir / "baseline_ladder.csv"


def test_resolves_underscore_naming_convention(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    exp_dir = tmp_path / "feature_to_mcherry_HD1509"
    exp_dir.mkdir(parents=True)
    (exp_dir / "baseline_ladder.csv").write_text("model,target\n")

    paths = resolve_experiment_paths("HD1509", config)

    assert paths.baseline_ladder_csv == exp_dir / "baseline_ladder.csv"


def test_resolves_hyphenated_experiment_scportrait_dir_folds_hyphen_to_underscore(
    tmp_path: Path,
) -> None:
    # Regression test: real HPC dir is feature_to_mcherry_ew2_2_scportrait (hyphen
    # folded to underscore too), not feature_to_mcherry_ew2-2_scportrait.
    config = _make_config(tmp_path)
    scportrait_dir = tmp_path / "feature_to_mcherry_ew2_2_scportrait"
    scportrait_dir.mkdir(parents=True)
    (scportrait_dir / "baseline_ladder.csv").write_text("model,target\n")

    paths = resolve_experiment_paths("Ew2-2", config)

    assert (
        paths.scportrait_baseline_ladder_csv == scportrait_dir / "baseline_ladder.csv"
    )


def test_resolves_scportrait_floor_metrics_csv(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    scportrait_dir = tmp_path / "feature_to_mcherry_HD1509_scportrait"
    info_dir = scportrait_dir / "morphology_informativeness"
    info_dir.mkdir(parents=True)
    (info_dir / "floor_metrics.csv").write_text("variant,model\n")

    paths = resolve_experiment_paths("HD1509", config)

    assert paths.scportrait_floor_metrics_csv == info_dir / "floor_metrics.csv"


def test_ignores_alpha_sweep_variant_dirs(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    real_dir = Path(config.feature_to_mcherry_dir) / "Ew2-2"
    real_dir.mkdir(parents=True)
    (real_dir / "baseline_ladder.csv").write_text("model,target\n")
    variant_dir = Path(config.feature_to_mcherry_dir) / "Ew2-2_alpha1p0"
    variant_dir.mkdir(parents=True)
    (variant_dir / "baseline_ladder.csv").write_text("SHOULD_NOT_BE_PICKED\n")

    paths = resolve_experiment_paths("Ew2-2", config)

    assert paths.baseline_ladder_csv == real_dir / "baseline_ladder.csv"


def test_missing_experiment_returns_none_fields_not_exception(tmp_path: Path) -> None:
    config = _make_config(tmp_path)

    paths = resolve_experiment_paths("DoesNotExist", config)

    assert paths.baseline_ladder_csv is None
    assert set(missing_fields(paths)) == {
        "baseline_ladder_csv",
        "scportrait_baseline_ladder_csv",
        "scportrait_floor_metrics_csv",
        "informativeness_dir",
        "floor_metrics_csv",
        "univariate_correlations_csv",
        "feature_importances_csv",
        "data_quality_csv",
        "feature_dir",
        "instance_metrics_csv",
    }


def test_resolves_data_quality_csv_by_source_membership_in_pair_dir(
    tmp_path: Path,
) -> None:
    config = _make_config(tmp_path)
    pair_dir = Path(config.data_quality_dir) / "Ew2-1_Ew2-2"
    pair_dir.mkdir(parents=True)
    (pair_dir / "extreme_value_report.csv").write_text("source,value_column\n")

    paths = resolve_experiment_paths("Ew2-2", config)

    assert paths.data_quality_csv == pair_dir / "extreme_value_report.csv"


def test_resolves_underscore_naming_convention_case_insensitively(
    tmp_path: Path,
) -> None:
    # Real HPC layout: feature_to_mcherry_hd1509/ is lowercase even though the
    # canonical experiment name is "HD1509".
    config = _make_config(tmp_path)
    exp_dir = tmp_path / "feature_to_mcherry_hd1509"
    exp_dir.mkdir(parents=True)
    (exp_dir / "baseline_ladder.csv").write_text("model,target\n")

    paths = resolve_experiment_paths("HD1509", config)

    assert paths.baseline_ladder_csv == exp_dir / "baseline_ladder.csv"


def test_resolves_scportrait_dir_case_insensitively(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    exp_dir = tmp_path / "feature_to_mcherry_hd1509_scportrait"
    exp_dir.mkdir(parents=True)
    (exp_dir / "baseline_ladder.csv").write_text("model,target\n")

    paths = resolve_experiment_paths("HD1509", config)

    assert paths.scportrait_baseline_ladder_csv == exp_dir / "baseline_ladder.csv"


def test_resolves_nested_per_experiment_convention(tmp_path: Path) -> None:
    # Current HPC layout (2026-07-28): baselines/data_quality/morphology_informativeness
    # nested under feature_to_mcherry/<Exp>/, not top-level trees.
    config = _make_config(tmp_path)
    exp_dir = Path(config.feature_to_mcherry_dir) / "HD1883"

    baselines_dir = exp_dir / "baselines"
    baselines_dir.mkdir(parents=True)
    (baselines_dir / "baseline_ladder.csv").write_text("model,target\n")

    dq_dir = exp_dir / "data_quality"
    dq_dir.mkdir(parents=True)
    (dq_dir / "extreme_value_report.csv").write_text("source,value_column\n")

    info_dir = exp_dir / "morphology_informativeness"
    info_dir.mkdir(parents=True)
    (info_dir / "floor_metrics.csv").write_text("variant,model\n")
    (info_dir / "univariate_correlations.csv").write_text("feature,target\n")

    paths = resolve_experiment_paths("HD1883", config)

    assert paths.baseline_ladder_csv == baselines_dir / "baseline_ladder.csv"
    assert paths.data_quality_csv == dq_dir / "extreme_value_report.csv"
    assert paths.informativeness_dir == info_dir
    assert paths.floor_metrics_csv == info_dir / "floor_metrics.csv"
    assert paths.univariate_correlations_csv == info_dir / "univariate_correlations.csv"


def test_top_level_convention_preferred_over_nested_when_both_exist(
    tmp_path: Path,
) -> None:
    config = _make_config(tmp_path)
    exp_dir = Path(config.feature_to_mcherry_dir) / "Ew2-1"

    # Convention 3 (top-level, direct)
    exp_dir.mkdir(parents=True)
    (exp_dir / "baseline_ladder.csv").write_text("top-level\n")

    # Convention 2 (nested under baselines/) also present.
    nested_dir = exp_dir / "baselines"
    nested_dir.mkdir(parents=True)
    (nested_dir / "baseline_ladder.csv").write_text("nested\n")

    paths = resolve_experiment_paths("Ew2-1", config)

    assert paths.baseline_ladder_csv == exp_dir / "baseline_ladder.csv"
    assert paths.baseline_ladder_csv.read_text() == "top-level\n"


def test_resolves_raw_feature_dir_and_instance_metrics(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    exp_dir = Path(config.data_root) / "Ew2-1 MF5V1 0-72h 06-03-26"
    feature_dir = (
        exp_dir
        / "inference_tracked"
        / "cellpose_sam"
        / "features_incarta"
        / "split_data"
    )
    feature_dir.mkdir(parents=True)
    target_dir = exp_dir / "mcherry_metrics" / "cellpose_sam"
    target_dir.mkdir(parents=True)
    (target_dir / "instance_metrics.csv").write_text("cell_id\n")

    paths = resolve_experiment_paths("Ew2-1", config)

    assert paths.feature_dir == feature_dir
    assert paths.instance_metrics_csv == target_dir / "instance_metrics.csv"
