"""Parquet feature tables, as written by the pyradiomics per-well output."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("pyarrow")

from src.feature_to_mcherry.config import load_config  # noqa: E402
from src.feature_to_mcherry.data.loaders import (  # noqa: E402
    load_features,
    load_features_from_directory,
)

RADIOMICS_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "config"
    / "feature_to_mcherry_radiomics_config.yaml"
)


def _well_table(well: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cell_id": [1, 2],
            "sample_id": [well, well],
            "timepoint": ["101", "101"],
            "z_index": [3, 3],
            "original_shape2D_Elongation": [0.5, 0.6],
            "touches_border": [False, True],
        }
    )


def _coverage_table(well: str) -> pd.DataFrame:
    return pd.DataFrame(
        {"sample_id": [well], "timepoint": ["101"], "z_index": [3], "status": ["ok"]}
    )


def test_load_features_reads_parquet_by_suffix(tmp_path: Path) -> None:
    path = tmp_path / "E07.parquet"
    _well_table("E07").to_parquet(path, index=False)

    result = load_features(path, id_column="cell_id")

    assert list(result.columns) == [
        "sample_id",
        "timepoint",
        "z_index",
        "cell_id",
        "original_shape2D_Elongation",
    ]
    assert len(result) == 2


def test_directory_loader_ignores_the_coverage_subfolder(tmp_path: Path) -> None:
    (tmp_path / "coverage").mkdir()
    for well in ("E07", "F08"):
        _well_table(well).to_parquet(tmp_path / f"{well}.parquet", index=False)
        _coverage_table(well).to_parquet(
            tmp_path / "coverage" / f"{well}.parquet", index=False
        )

    result = load_features_from_directory(
        tmp_path, id_column="cell_id", pattern="*.parquet"
    )

    assert sorted(result["sample_id"].unique()) == ["E07", "F08"]
    assert len(result) == 4
    assert "status" not in result.columns


def test_radiomics_config_excludes_the_13_screened_features() -> None:
    config = load_config(RADIOMICS_CONFIG)

    assert config.feature_pattern == "*.parquet"
    assert config.id_column == "cell_id"
    assert len(set(config.exclude_feature_columns)) == 13
    assert all(
        column.startswith("original_") for column in config.exclude_feature_columns
    )
