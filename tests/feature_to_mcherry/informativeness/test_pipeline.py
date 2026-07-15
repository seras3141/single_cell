"""End-to-end smoke test for feature_to_mcherry.informativeness.pipeline.run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from src.feature_to_mcherry.informativeness.config import InformativenessConfig
from src.feature_to_mcherry.informativeness.pipeline import run

REPO_ROOT = Path(__file__).resolve().parents[3]
REAL_LAYOUT_PATH = REPO_ROOT / "config" / "MF5v1_plate_layout.json"

# Doxorubicin rows C (replicate 1) / D (replicate 2); quadrant-1 offsets 2 and 3 give
# conc_1 (1.0 uM) and conc_2 (0.1 uM) respectively.
WELLS = ["C02", "D02", "C03", "D03"]


def _write_synthetic_csvs(
    tmp_path: Path, n_per_well: int = 15, seed: int = 0
) -> Tuple[Path, Path]:
    rng = np.random.default_rng(seed)

    rows_features = []
    rows_targets = []
    for well in WELLS:
        area = rng.uniform(0, 10, size=n_per_well)
        perimeter = rng.uniform(0, 10, size=n_per_well)
        mean_intensity = rng.uniform(0, 10, size=n_per_well)
        noise = rng.normal(0, 0.3, size=n_per_well)
        base = 2 * area + perimeter + noise

        for i in range(n_per_well):
            cell_id = i + 1
            rows_features.append(
                {
                    "instance_id": cell_id,
                    "well": well,
                    "frame": 1,
                    "z": 1,
                    "area": area[i],
                    "perimeter": perimeter[i],
                    "mean_intensity": mean_intensity[i],
                }
            )
            rows_targets.append(
                {
                    "sample_id": well,
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
    tmp_path: Path, n_per_well: int = 15, seed: int = 0
) -> Tuple[Path, Path]:
    """Ew2-2-style layout: ``feature_csv`` is a directory of per-well CSVs, each
    already carrying ``sample_id``/``timepoint``/``z_index``/``cell_id`` columns."""
    rng = np.random.default_rng(seed)

    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    rows_targets = []
    for well in WELLS:
        area = rng.uniform(0, 10, size=n_per_well)
        perimeter = rng.uniform(0, 10, size=n_per_well)
        mean_intensity = rng.uniform(0, 10, size=n_per_well)
        noise = rng.normal(0, 0.3, size=n_per_well)
        base = 2 * area + perimeter + noise

        rows_features = []
        for i in range(n_per_well):
            cell_id = i + 1
            rows_features.append(
                {
                    "cell_id": cell_id,
                    "sample_id": well,
                    "timepoint": 1,
                    "z_index": 1,
                    "area": area[i],
                    "perimeter": perimeter[i],
                    "mean_intensity": mean_intensity[i],
                }
            )
            rows_targets.append(
                {
                    "sample_id": well,
                    "timepoint": 1,
                    "z_index": 1,
                    "cell_id": cell_id,
                    "percentile_75": base[i],
                    "percentile_90": base[i] + 5.0,
                    "percentile_95": base[i] + 10.0,
                }
            )
        pd.DataFrame(rows_features).to_csv(
            feature_dir / f"p_{well}_t1_z1_features.csv", index=False
        )

    target_csv = tmp_path / "instance_metrics.csv"
    pd.DataFrame(rows_targets).to_csv(target_csv, index=False)
    return feature_dir, target_csv


def _write_synthetic_directory_feature_and_target_csvs(
    tmp_path: Path, n_per_well: int = 15, seed: int = 0
) -> Tuple[Path, Path]:
    """mcherry_metrics split_data-style layout: both ``feature_csv`` and
    ``target_csv`` are directories of per-well CSVs, each already carrying
    ``sample_id``/``timepoint``/``z_index``/``cell_id`` columns."""
    rng = np.random.default_rng(seed)

    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    target_dir = tmp_path / "targets"
    target_dir.mkdir()
    for well in WELLS:
        area = rng.uniform(0, 10, size=n_per_well)
        perimeter = rng.uniform(0, 10, size=n_per_well)
        mean_intensity = rng.uniform(0, 10, size=n_per_well)
        noise = rng.normal(0, 0.3, size=n_per_well)
        base = 2 * area + perimeter + noise

        rows_features = []
        rows_targets = []
        for i in range(n_per_well):
            cell_id = i + 1
            rows_features.append(
                {
                    "cell_id": cell_id,
                    "sample_id": well,
                    "timepoint": 1,
                    "z_index": 1,
                    "area": area[i],
                    "perimeter": perimeter[i],
                    "mean_intensity": mean_intensity[i],
                }
            )
            rows_targets.append(
                {
                    "sample_id": well,
                    "timepoint": 1,
                    "z_index": 1,
                    "cell_id": cell_id,
                    "percentile_75": base[i],
                    "percentile_90": base[i] + 5.0,
                    "percentile_95": base[i] + 10.0,
                }
            )
        pd.DataFrame(rows_features).to_csv(
            feature_dir / f"p_{well}_t1_z1_features.csv", index=False
        )
        pd.DataFrame(rows_targets).to_csv(
            target_dir / f"p_{well}_t1_z1_mCherry_metrics.csv", index=False
        )

    return feature_dir, target_dir


def test_pipeline_end_to_end_writes_all_outputs(tmp_path: Path) -> None:
    feature_csv, target_csv = _write_synthetic_csvs(tmp_path)
    output_dir = tmp_path / "results"

    config = InformativenessConfig(
        feature_csv=str(feature_csv),
        target_csv=str(target_csv),
        id_column="instance_id",
        sample_id_column="well",
        timepoint_column="frame",
        z_index_column="z",
        group_by="sample_id",
        n_splits=2,
        morphology_feature_patterns=["area", "perimeter"],
        suspect_feature_patterns=["mean_intensity"],
        plate_layout_json=str(REAL_LAYOUT_PATH),
        output_dir=str(output_dir),
    )

    bundle = run(config)

    assert bundle.n_cells == len(WELLS) * 15
    assert bundle.n_features_all == 3  # area, perimeter, mean_intensity
    assert bundle.n_features_clean == 2  # area, perimeter (mean_intensity is suspect)
    assert bundle.suspect_feature_names == ["mean_intensity"]

    assert (output_dir / "univariate_correlations.csv").exists()
    assert (output_dir / "floor_metrics.csv").exists()
    assert (output_dir / "noise_ceiling.csv").exists()
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "report.md").exists()
    assert (output_dir / "figures").is_dir()
    assert any((output_dir / "figures").glob("*.png"))

    floor_metrics = pd.read_csv(output_dir / "floor_metrics.csv")
    assert set(floor_metrics["variant"]) == {"with_suspect", "without_suspect"}
    assert set(floor_metrics["model"]) == {"ridge", "gradient_boosting"}
    assert floor_metrics["r2"].apply(np.isfinite).all()

    noise_ceiling = pd.read_csv(output_dir / "noise_ceiling.csv")
    assert len(noise_ceiling) == 3  # one row per target

    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["n_cells"] == bundle.n_cells

    report_text = (output_dir / "report.md").read_text().lower()
    assert "no automatic go/no-go verdict" in report_text
    for forbidden in ("verdict: go", "verdict: no-go", "recommendation: proceed"):
        assert forbidden not in report_text


def test_pipeline_skips_without_suspect_variant_when_no_clean_features_remain(
    tmp_path: Path,
) -> None:
    feature_csv, target_csv = _write_synthetic_csvs(tmp_path)
    output_dir = tmp_path / "results"

    config = InformativenessConfig(
        feature_csv=str(feature_csv),
        target_csv=str(target_csv),
        id_column="instance_id",
        sample_id_column="well",
        timepoint_column="frame",
        z_index_column="z",
        group_by="sample_id",
        n_splits=2,
        morphology_feature_patterns=["mean_intensity"],
        suspect_feature_patterns=["mean_intensity"],
        plate_layout_json=None,
        output_dir=str(output_dir),
    )

    bundle = run(config)

    assert bundle.n_features_clean == 0
    floor_metrics = pd.read_csv(output_dir / "floor_metrics.csv")
    assert set(floor_metrics["variant"]) == {"with_suspect"}

    noise_ceiling = pd.read_csv(output_dir / "noise_ceiling.csv")
    assert noise_ceiling["ceiling"].isna().all()


def test_pipeline_end_to_end_with_directory_feature_csv(tmp_path: Path) -> None:
    feature_dir, target_csv = _write_synthetic_directory_csvs(tmp_path)
    output_dir = tmp_path / "results"

    config = InformativenessConfig(
        feature_csv=str(feature_dir),
        target_csv=str(target_csv),
        id_column="cell_id",
        group_by="sample_id",
        n_splits=2,
        morphology_feature_patterns=["area", "perimeter"],
        suspect_feature_patterns=["mean_intensity"],
        plate_layout_json=str(REAL_LAYOUT_PATH),
        output_dir=str(output_dir),
    )

    bundle = run(config)

    assert bundle.n_cells == len(WELLS) * 15
    assert bundle.n_features_all == 3
    assert bundle.n_features_clean == 2

    assert (output_dir / "univariate_correlations.csv").exists()
    assert (output_dir / "floor_metrics.csv").exists()
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "report.md").exists()
    assert any((output_dir / "figures").glob("*.png"))

    floor_metrics = pd.read_csv(output_dir / "floor_metrics.csv")
    assert floor_metrics["r2"].apply(np.isfinite).all()


def test_pipeline_end_to_end_with_directory_feature_and_target_csv(
    tmp_path: Path,
) -> None:
    feature_dir, target_dir = _write_synthetic_directory_feature_and_target_csvs(
        tmp_path
    )
    output_dir = tmp_path / "results"

    config = InformativenessConfig(
        feature_csv=str(feature_dir),
        target_csv=str(target_dir),
        id_column="cell_id",
        group_by="sample_id",
        n_splits=2,
        morphology_feature_patterns=["area", "perimeter"],
        suspect_feature_patterns=["mean_intensity"],
        plate_layout_json=str(REAL_LAYOUT_PATH),
        output_dir=str(output_dir),
    )

    bundle = run(config)

    assert bundle.n_cells == len(WELLS) * 15
    assert bundle.n_features_all == 3
    assert bundle.n_features_clean == 2

    assert (output_dir / "univariate_correlations.csv").exists()
    assert (output_dir / "floor_metrics.csv").exists()
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "report.md").exists()
    assert any((output_dir / "figures").glob("*.png"))

    floor_metrics = pd.read_csv(output_dir / "floor_metrics.csv")
    assert floor_metrics["r2"].apply(np.isfinite).all()
