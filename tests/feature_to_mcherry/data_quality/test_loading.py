"""Tests for feature_to_mcherry.data_quality.loading.load_source."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.feature_to_mcherry.data_quality.config import SourceConfig
from src.feature_to_mcherry.data_quality.loading import load_source


def _write_parquet_source(tmp_path: Path) -> SourceConfig:
    pytest.importorskip("pyarrow")
    feature_dir = tmp_path / "features"
    (feature_dir / "coverage").mkdir(parents=True)
    for well in ("E07", "F08"):
        pd.DataFrame(
            {
                "cell_id": [1, 2],
                "sample_id": [well, well],
                "timepoint": ["101", "101"],
                "z_index": [3, 3],
                "original_shape2D_Elongation": [0.5, 0.6],
            }
        ).to_parquet(feature_dir / f"{well}.parquet", index=False)
        pd.DataFrame({"sample_id": [well], "status": ["ok"]}).to_parquet(
            feature_dir / "coverage" / f"{well}.parquet", index=False
        )
    target_csv = tmp_path / "instance_metrics.csv"
    pd.DataFrame(
        {
            "sample_id": ["E07", "F08"],
            "timepoint": ["101", "101"],
            "z_index": [3, 3],
            "cell_id": [1, 1],
            "percentile_75": [1.0, 2.0],
            "percentile_90": [1.0, 2.0],
            "percentile_95": [1.0, 2.0],
        }
    ).to_csv(target_csv, index=False)
    return SourceConfig(
        label="HD1509",
        feature_csv=str(feature_dir),
        feature_pattern="*.parquet",
        target_csv=str(target_csv),
        id_column="cell_id",
    )


def test_load_source_reads_a_parquet_feature_directory(tmp_path: Path) -> None:
    features, targets = load_source(_write_parquet_source(tmp_path))

    assert sorted(features["sample_id"].unique()) == ["E07", "F08"]
    assert len(features) == 4
    assert "status" not in features.columns
    assert len(targets) == 2


def test_the_default_pattern_finds_no_parquet_tables(tmp_path: Path) -> None:
    source = _write_parquet_source(tmp_path)
    source.feature_pattern = "*.csv"

    with pytest.raises(ValueError, match="No files matching"):
        load_source(source)
