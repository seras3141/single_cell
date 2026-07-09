"""Tests for feature_to_mcherry.data.loaders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.feature_to_mcherry.data.loaders import load_features, load_targets


def test_load_targets_keeps_cell_key_and_targets_and_drops_nan(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "sample_id": ["C09", "C09", "C09"],
            "timepoint": [11, 11, 11],
            "label_id": [1, 2, 3],
            "percentile_75": [10.0, 20.0, np.nan],
            "percentile_90": [15.0, 25.0, 35.0],
            "percentile_95": [18.0, 28.0, 38.0],
            "mean_intensity": [1.0, 2.0, 3.0],  # not a target column, must be dropped
        }
    )
    csv_path = tmp_path / "instance_metrics.csv"
    df.to_csv(csv_path, index=False)

    result = load_targets(csv_path)

    assert list(result.columns) == [
        "sample_id",
        "timepoint",
        "label_id",
        "percentile_75",
        "percentile_90",
        "percentile_95",
    ]
    assert len(result) == 2  # the NaN row is dropped
    assert result["label_id"].tolist() == [1, 2]


def test_load_targets_raises_on_missing_columns(tmp_path: Path) -> None:
    df = pd.DataFrame({"sample_id": ["C09"], "timepoint": [11], "label_id": [1]})
    csv_path = tmp_path / "instance_metrics.csv"
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_targets(csv_path)


def test_load_features_with_explicit_sample_id_and_timepoint_columns(
    tmp_path: Path,
) -> None:
    image_filename = "pMF5V1_C09_t11_z10_BF.tif"
    df = pd.DataFrame(
        {
            "instance_id": [1, 2],
            "well": ["C09", "C09"],
            "frame": [11, 11],
            "area": [100.0, 120.0],
            "mean_intensity": [5.0, 6.0],
            "image_filename": [image_filename, image_filename],
        }
    )
    csv_path = tmp_path / "features.csv"
    df.to_csv(csv_path, index=False)

    result = load_features(
        csv_path,
        id_column="instance_id",
        sample_id_column="well",
        timepoint_column="frame",
    )

    expected_columns = {"sample_id", "timepoint", "label_id", "area", "mean_intensity"}
    assert set(result.columns) == expected_columns
    assert result["sample_id"].tolist() == ["C09", "C09"]
    assert result["timepoint"].tolist() == [11, 11]
    assert result["label_id"].tolist() == [1, 2]


def test_load_features_derives_sample_id_and_timepoint_from_image_filename(
    tmp_path: Path,
) -> None:
    image_filename = "pMF5V1_C09_t11_z10_BF.tif"
    df = pd.DataFrame(
        {
            "instance_id": [1, 2],
            "area": [100.0, 120.0],
            "mean_intensity": [5.0, 6.0],
            "image_filename": [image_filename, image_filename],
            "mask_filename": ["pMF5V1_C09_t11_z10_pred_mask.tif"] * 2,
            "processing_timestamp": ["2026-07-09T00:00:00"] * 2,
        }
    )
    csv_path = tmp_path / "features.csv"
    df.to_csv(csv_path, index=False)

    result = load_features(csv_path, id_column="instance_id")

    assert result["sample_id"].tolist() == ["C09", "C09"]
    assert result["timepoint"].tolist() == ["11", "11"]
    expected_columns = {"sample_id", "timepoint", "label_id", "area", "mean_intensity"}
    assert set(result.columns) == expected_columns


def test_load_features_derives_empty_timepoint_when_absent_from_filename(
    tmp_path: Path,
) -> None:
    # Real example: p2126_J04_z2_BF.tif has no "_t<N>" component.
    df = pd.DataFrame(
        {
            "instance_id": [1],
            "area": [276.0],
            "image_filename": ["p2126_J04_z2_BF.tif"],
        }
    )
    csv_path = tmp_path / "features.csv"
    df.to_csv(csv_path, index=False)

    result = load_features(csv_path, id_column="instance_id")

    assert result["sample_id"].tolist() == ["J04"]
    assert result["timepoint"].tolist() == [""]


def test_load_features_raises_on_missing_id_column(tmp_path: Path) -> None:
    df = pd.DataFrame({"area": [1.0], "image_filename": ["p2126_J04_z2_BF.tif"]})
    csv_path = tmp_path / "features.csv"
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="id_column"):
        load_features(csv_path, id_column="instance_id")


def test_load_features_raises_when_no_numeric_feature_columns_remain(
    tmp_path: Path,
) -> None:
    df = pd.DataFrame(
        {
            "instance_id": [1],
            "image_filename": ["p2126_J04_z2_BF.tif"],
            "mask_filename": ["p2126_J04_z2_Cells.tif"],
        }
    )
    csv_path = tmp_path / "features.csv"
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="No numeric feature columns"):
        load_features(csv_path, id_column="instance_id")
