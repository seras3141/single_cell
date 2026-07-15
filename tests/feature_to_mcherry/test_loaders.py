"""Tests for feature_to_mcherry.data.loaders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.feature_to_mcherry.data.loaders import (
    load_features,
    load_features_from_directory,
    load_targets,
    load_targets_from_directory,
)


def test_load_targets_keeps_cell_key_and_targets_and_drops_nan(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "sample_id": ["C09", "C09", "C09"],
            "timepoint": [11, 11, 11],
            "z_index": [10, 10, 10],
            "cell_id": [1, 2, 3],
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
        "z_index",
        "cell_id",
        "percentile_75",
        "percentile_90",
        "percentile_95",
    ]
    assert len(result) == 2  # the NaN row is dropped
    assert result["cell_id"].tolist() == [1, 2]


def test_load_targets_raises_on_missing_columns(tmp_path: Path) -> None:
    df = pd.DataFrame({"sample_id": ["C09"], "timepoint": [11], "cell_id": [1]})
    csv_path = tmp_path / "instance_metrics.csv"
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_targets(csv_path)


def test_load_targets_from_directory_concatenates_per_slice_files(
    tmp_path: Path,
) -> None:
    # Real-world shape: mcherry_metrics/cellpose_sam/split_data, one CSV per
    # (well, timepoint, z), same schema as the combined instance_metrics.csv.
    for z in (10, 11):
        df = pd.DataFrame(
            {
                "sample_id": ["E07", "E07"],
                "timepoint": [101, 101],
                "z_index": [z, z],
                "cell_id": [3, 17],
                "percentile_75": [156.0, 158.0],
                "percentile_90": [158.0, 160.0],
                "percentile_95": [159.0, 161.0],
            }
        )
        df.to_csv(tmp_path / f"pMF5V1_E07_t101_z{z}_mCherry_metrics.csv", index=False)

    result = load_targets_from_directory(tmp_path)

    assert len(result) == 4
    assert set(result["z_index"]) == {10, 11}
    assert (result["sample_id"] == "E07").all()


def test_load_targets_from_directory_drops_nan_rows_per_file(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "sample_id": ["E07", "E07"],
            "timepoint": [101, 101],
            "z_index": [10, 10],
            "cell_id": [3, 17],
            "percentile_75": [156.0, np.nan],
            "percentile_90": [158.0, 160.0],
            "percentile_95": [159.0, 161.0],
        }
    )
    df.to_csv(tmp_path / "pMF5V1_E07_t101_z10_mCherry_metrics.csv", index=False)

    result = load_targets_from_directory(tmp_path)

    assert len(result) == 1
    assert result["cell_id"].tolist() == [3]


def test_load_targets_from_directory_raises_when_no_files_match(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="No files matching pattern"):
        load_targets_from_directory(tmp_path)


def test_load_features_with_explicit_sample_id_timepoint_and_z_index_columns(
    tmp_path: Path,
) -> None:
    df = pd.DataFrame(
        {
            "instance_id": [1, 2],
            "well": ["C09", "C09"],
            "frame": [11, 11],
            "z": [10, 10],
            "area": [100.0, 120.0],
            "mean_intensity": [5.0, 6.0],
        }
    )
    csv_path = tmp_path / "features.csv"
    df.to_csv(csv_path, index=False)

    result = load_features(
        csv_path,
        id_column="instance_id",
        sample_id_column="well",
        timepoint_column="frame",
        z_index_column="z",
    )

    expected_columns = {
        "sample_id",
        "timepoint",
        "z_index",
        "cell_id",
        "area",
        "mean_intensity",
    }
    assert set(result.columns) == expected_columns
    assert result["sample_id"].tolist() == ["C09", "C09"]
    assert result["timepoint"].tolist() == [11, 11]
    assert result["z_index"].tolist() == [10, 10]
    assert result["cell_id"].tolist() == [1, 2]


def test_load_features_uses_already_present_sample_id_timepoint_z_index_columns(
    tmp_path: Path,
) -> None:
    # Real-world shape (incarta split_data, current format): sample_id/timepoint/
    # z_index are already native columns in every row; no image_filename column at
    # all. No explicit *_column args should be needed.
    df = pd.DataFrame(
        {
            "cell_id": [31, 33],
            "area": [138.0, 181.0],
            "sample_id": ["I07", "I07"],
            "timepoint": [301, 301],
            "z_index": [10, 10],
        }
    )
    csv_path = tmp_path / "features.csv"
    df.to_csv(csv_path, index=False)

    result = load_features(csv_path, id_column="cell_id")

    expected_columns = {"sample_id", "timepoint", "z_index", "cell_id", "area"}
    assert set(result.columns) == expected_columns
    assert result["sample_id"].tolist() == ["I07", "I07"]
    assert result["timepoint"].tolist() == [301, 301]
    assert result["z_index"].tolist() == [10, 10]
    assert result["cell_id"].tolist() == [31, 33]


def test_load_features_derives_sample_id_timepoint_and_z_index_from_image_filename(
    tmp_path: Path,
) -> None:
    # Real-world shape (regionprops): no native sample_id/timepoint/z_index columns,
    # but an image_filename column to derive them from.
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
    assert result["z_index"].tolist() == ["10", "10"]
    expected_columns = {
        "sample_id",
        "timepoint",
        "z_index",
        "cell_id",
        "area",
        "mean_intensity",
    }
    assert set(result.columns) == expected_columns


def test_load_features_derives_empty_timepoint_when_absent_from_filename(
    tmp_path: Path,
) -> None:
    # Real example: p2126_J04_z2_BF.tif has no "_t<N>" component, but does have z.
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
    assert result["z_index"].tolist() == ["2"]


def test_load_features_raises_on_missing_id_column(tmp_path: Path) -> None:
    df = pd.DataFrame({"area": [1.0], "image_filename": ["p2126_J04_z2_BF.tif"]})
    csv_path = tmp_path / "features.csv"
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="id_column"):
        load_features(csv_path, id_column="instance_id")


def test_load_features_raises_on_missing_explicit_column(tmp_path: Path) -> None:
    df = pd.DataFrame({"instance_id": [1], "area": [100.0], "well": ["C09"]})
    csv_path = tmp_path / "features.csv"
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="timepoint_column"):
        load_features(
            csv_path,
            id_column="instance_id",
            sample_id_column="well",
            timepoint_column="nonexistent",
        )


def test_load_features_raises_when_metadata_unresolvable(tmp_path: Path) -> None:
    # No explicit *_column args, no already-present sample_id/timepoint/z_index
    # columns, and no image_filename column to derive from.
    df = pd.DataFrame({"instance_id": [1], "area": [100.0]})
    csv_path = tmp_path / "features.csv"
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Cannot resolve"):
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


def test_load_features_from_directory_concatenates_already_present_columns(
    tmp_path: Path,
) -> None:
    # Real-world shape (incarta split_data, current format): one file per
    # (well, timepoint, z), each already carrying sample_id/timepoint/z_index/cell_id.
    for z in (10, 11):
        df = pd.DataFrame(
            {
                "cell_id": [1, 2],
                "area": [100.0 + z, 120.0 + z],
                "sample_id": ["I07", "I07"],
                "timepoint": [301, 301],
                "z_index": [z, z],
            }
        )
        df.to_csv(tmp_path / f"pMF5V1_I07_t301_z{z}_BF_features.csv", index=False)

    result = load_features_from_directory(tmp_path, id_column="cell_id")

    assert len(result) == 4
    assert set(result["z_index"]) == {10, 11}
    assert (result["sample_id"] == "I07").all()


def test_load_features_from_directory_derives_via_image_filename_column(
    tmp_path: Path,
) -> None:
    for z in (10, 11):
        image_filename = f"pMF5V1_C09_t11_z{z}_BF.tif"
        df = pd.DataFrame(
            {
                "instance_id": [1, 2],
                "area": [100.0, 120.0],
                "image_filename": [image_filename, image_filename],
            }
        )
        df.to_csv(tmp_path / f"pMF5V1_C09_t11_z{z}_BF_features.csv", index=False)

    result = load_features_from_directory(tmp_path, id_column="instance_id")

    assert len(result) == 4
    assert set(result["z_index"]) == {"10", "11"}


def test_load_features_from_directory_raises_when_no_files_match(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="No files matching pattern"):
        load_features_from_directory(tmp_path, id_column="instance_id")


def test_load_features_from_directory_raises_when_metadata_unresolvable(
    tmp_path: Path,
) -> None:
    df = pd.DataFrame({"instance_id": [1], "area": [100.0]})
    df.to_csv(tmp_path / "features.csv", index=False)

    with pytest.raises(ValueError, match="Cannot resolve"):
        load_features_from_directory(tmp_path, id_column="instance_id")
