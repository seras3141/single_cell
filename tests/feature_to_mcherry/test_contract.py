"""Tests for feature_to_mcherry.data.contract."""

from __future__ import annotations

import pandas as pd
import pytest

from src.feature_to_mcherry.data.contract import (
    CELL_KEY,
    TARGET_COLUMNS,
    normalize_cell_key,
    taus_from_target_columns,
)


def test_taus_from_target_columns_default() -> None:
    assert taus_from_target_columns(TARGET_COLUMNS) == [0.75, 0.90, 0.95]


def test_taus_from_target_columns_custom_order() -> None:
    assert taus_from_target_columns(["percentile_95", "percentile_50"]) == [0.95, 0.50]


def test_taus_from_target_columns_rejects_unrecognized_name() -> None:
    with pytest.raises(ValueError, match="Cannot derive a quantile level"):
        taus_from_target_columns(["mean_intensity"])


def test_normalize_cell_key_casts_to_string() -> None:
    df = pd.DataFrame(
        {
            "sample_id": ["C09", "C09"],
            "timepoint": [11, 11],
            "z_index": [10, 10],
            "cell_id": [1, 2],
            "other": [1.0, 2.0],
        }
    )
    normalized = normalize_cell_key(df)
    for column in CELL_KEY:
        assert normalized[column].dtype == object
        assert all(isinstance(value, str) for value in normalized[column])
    # untouched columns are left alone
    assert normalized["other"].dtype != object
