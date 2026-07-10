"""Batch tests for mCherry metrics extraction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from src.mcherry_metrics import ExtractionConfig, run_extraction


def _write_pair(directory: Path, z_index: int) -> None:
    image_path = directory / f"p2426_B02_t1_z{z_index}_mCherry.tif"
    label_path = directory / f"p2426_B02_t1_z{z_index}_Cells.tif"
    image = np.array([[10, 20], [30, 40]], dtype=np.uint16)
    labels = np.array([[1, 1], [1, 0]], dtype=np.uint16)
    tifffile.imwrite(image_path, image)
    tifffile.imwrite(label_path, labels)


def test_run_extraction_writes_outputs_and_filters_z0(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _write_pair(image_dir, z_index=0)
    _write_pair(image_dir, z_index=1)

    output_dir = tmp_path / "out"
    metrics_df = run_extraction(
        mcherry_dir=image_dir,
        output_dir=output_dir,
        config=ExtractionConfig(min_area_px=1, exclude_z0=True, write_analytics=False),
    )

    assert metrics_df["z_index"].tolist() == [1]
    assert (output_dir / "instance_metrics.csv").exists()
    assert (output_dir / "metrics_summary.csv").exists()

    written = pd.read_csv(output_dir / "instance_metrics.csv")
    assert written["sample_id"].tolist() == ["B02"]


def test_run_extraction_preserves_labeling_compatibility_columns(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _write_pair(image_dir, z_index=1)

    output_dir = tmp_path / "out"
    run_extraction(
        mcherry_dir=image_dir,
        output_dir=output_dir,
        config=ExtractionConfig(min_area_px=1, exclude_z0=False, write_analytics=False),
    )

    written = pd.read_csv(output_dir / "instance_metrics.csv")

    expected_columns = {
        "image",
        "sample",
        "time",
        "ID",
        "mean_intensity",
        "max_intensity",
        "sum_intensity",
        "percentile_75",
        "percentile_90",
        "percentile_95",
    }

    assert expected_columns.issubset(written.columns)
    assert written.loc[0, "image"] == "p2426_B02_t1_z1_mCherry.tif"
    assert written.loc[0, "sample"] == "B02"
    assert str(written.loc[0, "time"]) == "1"
    assert written.loc[0, "ID"] == "p2426_B02_t1"