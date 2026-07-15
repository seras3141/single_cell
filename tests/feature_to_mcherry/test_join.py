"""Tests for feature_to_mcherry.data.join."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_to_mcherry.data.join import build_matrix


def _features_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["C09", "C09", "C09"],
            "timepoint": [11, 11, 11],
            "z_index": [10, 10, 10],
            "cell_id": [1, 2, 3],  # cell 3 has no matching target row
            "area": [100.0, 120.0, 90.0],
            "mean_intensity": [5.0, 6.0, 7.0],
        }
    )


def _targets_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["C09", "C09", "C09"],
            "timepoint": [11, 11, 11],
            "z_index": [10, 10, 10],
            "cell_id": [1, 2, 4],  # cell 4 has no matching feature row
            "percentile_75": [10.0, 20.0, 30.0],
            "percentile_90": [15.0, 25.0, 35.0],
            "percentile_95": [18.0, 28.0, 38.0],
        }
    )


def test_build_matrix_inner_joins_and_drops_unmatched_cells() -> None:
    X, y, groups, feature_names = build_matrix(
        _features_df(),
        _targets_df(),
        target_columns=["percentile_75", "percentile_90", "percentile_95"],
        group_by="sample_id",
    )

    assert X.shape == (2, 2)  # cells 1 and 2 matched; cells 3 and 4 dropped
    assert y.shape == (2, 3)
    assert feature_names == ["area", "mean_intensity"]
    assert list(groups) == ["C09", "C09"]
    np.testing.assert_array_equal(X, np.array([[100.0, 5.0], [120.0, 6.0]]))
    np.testing.assert_array_equal(y, np.array([[10.0, 15.0, 18.0], [20.0, 25.0, 28.0]]))


def test_build_matrix_joins_each_z_slice_independently_when_cell_id_recurs() -> None:
    # Same cell_id recurs across z-slices within one (sample_id, timepoint) -- this
    # must NOT collapse rows; each z-slice is an independent join, per contract.py's
    # CELL_KEY docstring and implementation_prompt.md §8.
    features = pd.DataFrame(
        {
            "sample_id": ["C09", "C09", "C09", "C09"],
            "timepoint": [11, 11, 11, 11],
            "z_index": [10, 11, 10, 11],
            "cell_id": [1, 1, 2, 2],
            "area": [100.0, 105.0, 120.0, 125.0],
        }
    )
    targets = pd.DataFrame(
        {
            "sample_id": ["C09", "C09", "C09", "C09"],
            "timepoint": [11, 11, 11, 11],
            "z_index": [10, 11, 10, 11],
            "cell_id": [1, 1, 2, 2],
            "percentile_75": [10.0, 11.0, 20.0, 21.0],
            "percentile_90": [15.0, 16.0, 25.0, 26.0],
            "percentile_95": [18.0, 19.0, 28.0, 29.0],
        }
    )

    X, y, groups, feature_names = build_matrix(
        features,
        targets,
        target_columns=["percentile_75", "percentile_90", "percentile_95"],
        group_by="sample_id",
    )

    assert X.shape == (4, 1)  # all 4 (cell_id, z_index) slices joined independently
    assert y.shape == (4, 3)
    np.testing.assert_array_equal(X.flatten(), [100.0, 105.0, 120.0, 125.0])


def test_build_matrix_raises_on_duplicate_cell_key_in_features() -> None:
    features = _features_df()
    features = pd.concat([features, features.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate cell_key"):
        build_matrix(
            features,
            _targets_df(),
            target_columns=["percentile_75", "percentile_90", "percentile_95"],
            group_by="sample_id",
        )


def test_build_matrix_raises_on_empty_join() -> None:
    features = _features_df().assign(cell_id=[100, 101, 102])

    with pytest.raises(ValueError, match="zero matched cells"):
        build_matrix(
            features,
            _targets_df(),
            target_columns=["percentile_75", "percentile_90", "percentile_95"],
            group_by="sample_id",
        )


def test_build_matrix_raises_on_missing_group_by_column() -> None:
    with pytest.raises(ValueError, match="group_by column"):
        build_matrix(
            _features_df(),
            _targets_df(),
            target_columns=["percentile_75", "percentile_90", "percentile_95"],
            group_by="plate",
        )
