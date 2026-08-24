"""End-to-end test for feature_to_mcherry.pipeline.run."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from src.feature_to_mcherry.config import FeatureToMcherryConfig
from src.feature_to_mcherry.pipeline import _run_model, run


class _RecordingMeanModel:
    """Predicts the training mean; records the fit-time X shape for assertions."""

    fit_shapes: list = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_RecordingMeanModel":
        _RecordingMeanModel.fit_shapes.append(X.shape)
        self._mean = y.mean(axis=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.tile(self._mean, (len(X), 1))


def test_run_model_subsamples_training_rows_but_predicts_on_full_validation_fold() -> (
    None
):
    rng = np.random.default_rng(0)
    n_cells = 200
    X = rng.uniform(0, 10, size=(n_cells, 2))
    y = rng.uniform(0, 10, size=(n_cells, 3))
    groups = np.array(["A"] * (n_cells // 2) + ["B"] * (n_cells // 2))

    _RecordingMeanModel.fit_shapes = []
    result = _run_model(
        model_factory=_RecordingMeanModel,
        X=X,
        y=y,
        groups=groups,
        n_splits=2,
        group_by="sample_id",
        taus=[0.75, 0.9, 0.95],
        target_names=["percentile_75", "percentile_90", "percentile_95"],
        apply_sort=False,
        model_name="test_model",
        train_subsample_size=10,
    )

    assert all(shape[0] == 10 for shape in _RecordingMeanModel.fit_shapes)
    assert np.isfinite(result.oof_predictions).all()
    assert result.oof_predictions.shape == y.shape


def _write_synthetic_csvs(
    tmp_path: Path, n_per_group: int = 20, seed: int = 0
) -> Tuple[Path, Path]:
    rng = np.random.default_rng(seed)
    sample_ids = ["A01", "A02"]

    rows_features = []
    rows_targets = []
    for sample_id in sample_ids:
        f1 = rng.uniform(0, 10, size=n_per_group)
        f2 = rng.uniform(0, 10, size=n_per_group)
        noise = rng.normal(0, 0.5, size=n_per_group)
        base = 2 * f1 + 3 * f2 + noise

        for i in range(n_per_group):
            cell_id = i + 1
            rows_features.append(
                {
                    "instance_id": cell_id,
                    "well": sample_id,
                    "frame": 1,
                    "z": 1,
                    "feature_1": f1[i],
                    "feature_2": f2[i],
                }
            )
            rows_targets.append(
                {
                    "sample_id": sample_id,
                    "timepoint": 1,
                    "z_index": 1,
                    "cell_id": cell_id,
                    "percentile_75": base[i],
                    "percentile_90": base[i] + 5.0,
                    "percentile_95": base[i] + 10.0,
                }
            )

    feature_csv = tmp_path / "features.csv"
    target_csv = tmp_path / "instance_metrics.csv"
    pd.DataFrame(rows_features).to_csv(feature_csv, index=False)
    pd.DataFrame(rows_targets).to_csv(target_csv, index=False)
    return feature_csv, target_csv


def _write_synthetic_directory_csvs(
    tmp_path: Path, n_per_group: int = 20, seed: int = 0
) -> Tuple[Path, Path]:
    """Ew2-2-style layout: ``feature_csv`` is a directory of per-well CSVs, each
    already carrying ``sample_id``/``timepoint``/``z_index``/``cell_id`` columns."""
    rng = np.random.default_rng(seed)
    sample_ids = ["A01", "A02"]

    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    rows_targets = []
    for sample_id in sample_ids:
        f1 = rng.uniform(0, 10, size=n_per_group)
        f2 = rng.uniform(0, 10, size=n_per_group)
        noise = rng.normal(0, 0.5, size=n_per_group)
        base = 2 * f1 + 3 * f2 + noise

        rows_features = []
        for i in range(n_per_group):
            cell_id = i + 1
            rows_features.append(
                {
                    "cell_id": cell_id,
                    "sample_id": sample_id,
                    "timepoint": 1,
                    "z_index": 1,
                    "feature_1": f1[i],
                    "feature_2": f2[i],
                }
            )
            rows_targets.append(
                {
                    "sample_id": sample_id,
                    "timepoint": 1,
                    "z_index": 1,
                    "cell_id": cell_id,
                    "percentile_75": base[i],
                    "percentile_90": base[i] + 5.0,
                    "percentile_95": base[i] + 10.0,
                }
            )
        pd.DataFrame(rows_features).to_csv(
            feature_dir / f"p_{sample_id}_t1_z1_features.csv", index=False
        )

    target_csv = tmp_path / "instance_metrics.csv"
    pd.DataFrame(rows_targets).to_csv(target_csv, index=False)
    return feature_dir, target_csv


def _write_synthetic_directory_target_csvs(
    tmp_path: Path, n_per_group: int = 20, seed: int = 0
) -> Tuple[Path, Path]:
    """mcherry_metrics split_data-style layout: both ``feature_csv`` and
    ``target_csv`` are directories of per-well CSVs, each already carrying
    ``sample_id``/``timepoint``/``z_index``/``cell_id`` columns."""
    rng = np.random.default_rng(seed)
    sample_ids = ["A01", "A02"]

    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    target_dir = tmp_path / "targets"
    target_dir.mkdir()
    for sample_id in sample_ids:
        f1 = rng.uniform(0, 10, size=n_per_group)
        f2 = rng.uniform(0, 10, size=n_per_group)
        noise = rng.normal(0, 0.5, size=n_per_group)
        base = 2 * f1 + 3 * f2 + noise

        rows_features = []
        rows_targets = []
        for i in range(n_per_group):
            cell_id = i + 1
            rows_features.append(
                {
                    "cell_id": cell_id,
                    "sample_id": sample_id,
                    "timepoint": 1,
                    "z_index": 1,
                    "feature_1": f1[i],
                    "feature_2": f2[i],
                }
            )
            rows_targets.append(
                {
                    "sample_id": sample_id,
                    "timepoint": 1,
                    "z_index": 1,
                    "cell_id": cell_id,
                    "percentile_75": base[i],
                    "percentile_90": base[i] + 5.0,
                    "percentile_95": base[i] + 10.0,
                }
            )
        pd.DataFrame(rows_features).to_csv(
            feature_dir / f"p_{sample_id}_t1_z1_features.csv", index=False
        )
        pd.DataFrame(rows_targets).to_csv(
            target_dir / f"p_{sample_id}_t1_z1_mCherry_metrics.csv", index=False
        )

    return feature_dir, target_dir


def test_pipeline_end_to_end_produces_finite_metrics_for_both_models(
    tmp_path: Path,
) -> None:
    feature_csv, target_csv = _write_synthetic_csvs(tmp_path)
    output_dir = tmp_path / "results"

    config = FeatureToMcherryConfig(
        feature_csv=str(feature_csv),
        target_csv=str(target_csv),
        id_column="instance_id",
        sample_id_column="well",
        timepoint_column="frame",
        z_index_column="z",
        group_by="sample_id",
        n_splits=2,
        output_dir=str(output_dir),
    )

    results = run(config)

    assert results.n_cells == 40
    assert results.n_features == 2
    assert set(results.feature_names) == {"feature_1", "feature_2"}

    for model_result in (results.ridge, results.linear_quantile):
        assert len(model_result.pooled_metrics) == 3
        for metrics in model_result.pooled_metrics:
            assert math.isfinite(metrics["mae"])
            assert math.isfinite(metrics["r2"])
            assert math.isfinite(metrics["pinball_loss"])
        assert math.isfinite(model_result.pooled_crossing_rate)

    assert (output_dir / "baseline_ladder.csv").exists()
    assert (output_dir / "baseline_ladder.md").exists()

    report = pd.read_csv(output_dir / "baseline_ladder.csv")
    assert set(report["model"]) == {"ridge", "linear_quantile"}
    assert len(report) == 6  # 2 models x 3 targets

    assert (output_dir / "oof_predictions.csv").exists()
    oof = pd.read_csv(output_dir / "oof_predictions.csv")
    assert len(oof) == results.n_cells * len(results.target_names)
    assert set(oof["target_name"]) == set(results.target_names)
    assert oof["y_pred"].notna().all()

    for j, target_name in enumerate(results.target_names):
        subset = oof[oof["target_name"] == target_name].reset_index(drop=True)
        np.testing.assert_allclose(subset["y_true"].to_numpy(), results.y[:, j])
        np.testing.assert_allclose(
            subset["y_pred"].to_numpy(), results.ridge.oof_predictions[:, j]
        )
        assert list(subset["group_id"]) == list(results.groups)


def test_pipeline_end_to_end_with_directory_feature_csv(tmp_path: Path) -> None:
    feature_dir, target_csv = _write_synthetic_directory_csvs(tmp_path)
    output_dir = tmp_path / "results"

    config = FeatureToMcherryConfig(
        feature_csv=str(feature_dir),
        target_csv=str(target_csv),
        id_column="cell_id",
        group_by="sample_id",
        n_splits=2,
        output_dir=str(output_dir),
    )

    results = run(config)

    assert results.n_cells == 40
    assert results.n_features == 2
    assert set(results.feature_names) == {"feature_1", "feature_2"}

    for model_result in (results.ridge, results.linear_quantile):
        for metrics in model_result.pooled_metrics:
            assert math.isfinite(metrics["mae"])
            assert math.isfinite(metrics["r2"])
            assert math.isfinite(metrics["pinball_loss"])


def test_pipeline_end_to_end_with_quantile_train_subsample(tmp_path: Path) -> None:
    feature_csv, target_csv = _write_synthetic_csvs(tmp_path, n_per_group=50)
    output_dir = tmp_path / "results"

    config = FeatureToMcherryConfig(
        feature_csv=str(feature_csv),
        target_csv=str(target_csv),
        id_column="instance_id",
        sample_id_column="well",
        timepoint_column="frame",
        z_index_column="z",
        group_by="sample_id",
        n_splits=2,
        quantile_train_subsample_size=10,
        output_dir=str(output_dir),
    )

    results = run(config)

    assert results.n_cells == 100
    assert np.isfinite(results.linear_quantile.oof_predictions).all()
    assert np.isfinite(results.ridge.oof_predictions).all()
    for metrics in results.linear_quantile.pooled_metrics:
        assert math.isfinite(metrics["mae"])
        assert math.isfinite(metrics["r2"])


def test_pipeline_end_to_end_excludes_configured_feature_columns(
    tmp_path: Path,
) -> None:
    feature_csv, target_csv = _write_synthetic_csvs(tmp_path)

    df = pd.read_csv(feature_csv)
    df["bad_feature"] = np.nan  # e.g. an all-NaN column from a real backend
    df.to_csv(feature_csv, index=False)

    output_dir = tmp_path / "results"
    config = FeatureToMcherryConfig(
        feature_csv=str(feature_csv),
        target_csv=str(target_csv),
        id_column="instance_id",
        sample_id_column="well",
        timepoint_column="frame",
        z_index_column="z",
        exclude_feature_columns=["bad_feature"],
        group_by="sample_id",
        n_splits=2,
        output_dir=str(output_dir),
    )

    results = run(config)

    assert results.n_features == 2
    assert set(results.feature_names) == {"feature_1", "feature_2"}
    for model_result in (results.ridge, results.linear_quantile):
        for metrics in model_result.pooled_metrics:
            assert math.isfinite(metrics["mae"])
            assert math.isfinite(metrics["r2"])


def test_pipeline_end_to_end_with_directory_feature_and_target_csv(
    tmp_path: Path,
) -> None:
    feature_dir, target_dir = _write_synthetic_directory_target_csvs(tmp_path)
    output_dir = tmp_path / "results"

    config = FeatureToMcherryConfig(
        feature_csv=str(feature_dir),
        target_csv=str(target_dir),
        id_column="cell_id",
        group_by="sample_id",
        n_splits=2,
        output_dir=str(output_dir),
    )

    results = run(config)

    assert results.n_cells == 40
    assert results.n_features == 2
    assert set(results.feature_names) == {"feature_1", "feature_2"}

    for model_result in (results.ridge, results.linear_quantile):
        for metrics in model_result.pooled_metrics:
            assert math.isfinite(metrics["mae"])
            assert math.isfinite(metrics["r2"])
            assert math.isfinite(metrics["pinball_loss"])


_DMSO_VALID_PER_WELL_T = 6  # cells per (well, valid timepoint)
_DMSO_RETAINED = 3 * _DMSO_VALID_PER_WELL_T * 2  # 3 wells x 6 cells x timepoints {1,2}


def _write_synthetic_dmso_csvs(tmp_path: Path, seed: int = 0) -> Tuple[Path, Path]:
    """DMSO well (M11) + two drug wells. Timepoints 1-2 are valid; timepoint 3 has an
    INVALID DMSO reference (only 1 DMSO cell < min_cells), so every t3 row must be
    dropped by the normalize path — exercising NaN-z drop with a known retained count.
    """
    rng = np.random.default_rng(seed)
    rows_features = []
    rows_targets = []

    def _add(well: str, timepoint: int, n: int, offset: float) -> None:
        for j in range(n):
            cell_id = timepoint * 100 + j  # unique within (well, timepoint)
            f1 = float(rng.uniform(0, 10))
            f2 = float(rng.uniform(0, 10))
            base = 2 * f1 + 3 * f2 + float(rng.normal(0, 0.5)) + 5.0 * offset
            rows_features.append(
                {
                    "instance_id": cell_id,
                    "well": well,
                    "frame": timepoint,
                    "z": 1,
                    "feature_1": f1,
                    "feature_2": f2,
                }
            )
            rows_targets.append(
                {
                    "sample_id": well,
                    "timepoint": timepoint,
                    "z_index": 1,
                    "cell_id": cell_id,
                    "percentile_75": base,
                    "percentile_90": base + 5.0,
                    "percentile_95": base + 10.0,
                }
            )

    for offset, well in enumerate(["M11", "A01", "A02"]):
        _add(well, 1, _DMSO_VALID_PER_WELL_T, offset)
        _add(well, 2, _DMSO_VALID_PER_WELL_T, offset)
    # timepoint 3: DMSO reference invalid (1 cell) -> all t3 rows dropped by normalize.
    _add("M11", 3, 1, 0)
    _add("A01", 3, 2, 1)
    _add("A02", 3, 2, 2)

    feature_csv = tmp_path / "features.csv"
    target_csv = tmp_path / "instance_metrics.csv"
    pd.DataFrame(rows_features).to_csv(feature_csv, index=False)
    pd.DataFrame(rows_targets).to_csv(target_csv, index=False)
    return feature_csv, target_csv


def test_run_with_dmso_normalization(tmp_path: Path) -> None:
    """normalize_to_dmso: z_ targets, NaN-z rows dropped, crossing n/a for z targets."""
    feature_csv, target_csv = _write_synthetic_dmso_csvs(tmp_path)
    config = FeatureToMcherryConfig(
        feature_csv=str(feature_csv),
        target_csv=str(target_csv),
        id_column="instance_id",
        sample_id_column="well",
        timepoint_column="frame",
        z_index_column="z",
        n_splits=2,
        normalize_to_dmso=True,
        dmso_well="M11",
        output_dir=str(tmp_path / "out"),
    )
    results = run(config)

    assert results.target_names == [
        "z_percentile_75",
        "z_percentile_90",
        "z_percentile_95",
    ]
    # invalid-reference timepoint 3 (5 rows) dropped; only t1/t2 retained.
    assert results.n_cells == _DMSO_RETAINED
    assert np.isfinite(results.ridge.oof_predictions).all()
    # crossing rate is not meaningful for z targets -> reported as NaN.
    assert math.isnan(results.linear_quantile.pooled_crossing_rate)
    # persisted oof targets are exactly the z_ names.
    oof = pd.read_csv(tmp_path / "out" / "oof_predictions.csv")
    assert set(oof["target_name"].unique()) == {
        "z_percentile_75",
        "z_percentile_90",
        "z_percentile_95",
    }
