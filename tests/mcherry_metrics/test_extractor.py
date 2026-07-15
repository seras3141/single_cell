"""Tests for mCherry instance metric extraction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from src.mcherry_metrics import ExtractionConfig, MetricsExtractor


def _write_image_pair(
    image_path: Path,
    label_path: Path,
    image: np.ndarray,
    labels: np.ndarray,
) -> None:
    tifffile.imwrite(image_path, image)
    tifffile.imwrite(label_path, labels)


def test_metrics_extractor_returns_expected_contract(tmp_path: Path) -> None:
    image = np.array(
        [
            [10, 20, 0],
            [5, 15, 25],
            [0, 0, 0],
        ],
        dtype=np.uint16,
    )
    labels = np.array(
        [
            [1, 1, 0],
            [2, 2, 2],
            [0, 0, 0],
        ],
        dtype=np.uint16,
    )
    image_path = tmp_path / "p2426_A01_t1_z1_mCherry.tif"
    label_path = tmp_path / "p2426_A01_t1_z1_Cells.tif"
    _write_image_pair(image_path, label_path, image, labels)

    extractor = MetricsExtractor(
        ExtractionConfig(percentiles=[50], min_area_px=1, exclude_z0=False)
    )
    metrics_df = extractor.run([image_path], [label_path])

    assert list(metrics_df["cell_id"]) == [1, 2]
    assert metrics_df["image_path"].nunique() == 1
    assert metrics_df["label_path"].nunique() == 1
    assert metrics_df["sample_id"].tolist() == ["A01", "A01"]
    assert metrics_df["z_index"].tolist() == [1, 1]
    assert metrics_df["timepoint"].tolist() == ["1", "1"]
    assert metrics_df["image"].tolist() == [image_path.name, image_path.name]

    first = metrics_df.loc[metrics_df["cell_id"] == 1].iloc[0]
    second = metrics_df.loc[metrics_df["cell_id"] == 2].iloc[0]

    assert first["area"] == 2
    assert first["mean_intensity"] == pytest.approx(15.0)
    assert first["sum_intensity"] == pytest.approx(30.0)
    assert first["percentile_75"] == pytest.approx(17.5)
    assert first["percentile_90"] == pytest.approx(19.0)
    assert first["percentile_95"] == pytest.approx(19.5)
    assert first["percentile_50"] == pytest.approx(15.0)

    assert second["area"] == 3
    assert second["mean_intensity"] == pytest.approx(15.0)
    assert second["sum_intensity"] == pytest.approx(45.0)
    assert second["percentile_90"] == pytest.approx(23.0)


def test_metrics_extractor_uses_first_plane_for_3d_inputs(tmp_path: Path) -> None:
    image = np.stack(
        [
            np.array([[1, 2], [3, 4]], dtype=np.uint16),
            np.array([[100, 100], [100, 100]], dtype=np.uint16),
        ]
    )
    labels = np.stack(
        [
            np.array([[1, 1], [0, 0]], dtype=np.uint16),
            np.array([[0, 0], [2, 2]], dtype=np.uint16),
        ]
    )
    image_path = tmp_path / "p2426_A02_t1_z2_mCherry.tif"
    label_path = tmp_path / "p2426_A02_t1_z2_Cells.tif"
    _write_image_pair(image_path, label_path, image, labels)

    extractor = MetricsExtractor(
        ExtractionConfig(percentiles=[75], min_area_px=1, exclude_z0=False)
    )
    metrics_df = extractor.run([image_path], [label_path])

    assert len(metrics_df) == 1
    assert metrics_df.iloc[0]["sum_intensity"] == pytest.approx(3.0)