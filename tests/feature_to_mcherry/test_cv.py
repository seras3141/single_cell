"""Tests for feature_to_mcherry.evaluation.cv."""

from __future__ import annotations

import numpy as np
import pytest

from src.feature_to_mcherry.evaluation.cv import (
    grouped_kfold_indices,
    validate_group_count,
)


def test_validate_group_count_raises_when_too_few_groups() -> None:
    # Mirrors the real single-well MF5V1 dev sample (only one unique group).
    groups = np.array(["C09"] * 10)
    with pytest.raises(ValueError, match="only 1 unique group"):
        validate_group_count(groups, n_splits=5, group_by="sample_id")


def test_validate_group_count_passes_when_enough_groups() -> None:
    groups = np.array(["A", "B", "C", "D", "E"])
    validate_group_count(groups, n_splits=5, group_by="sample_id")


def test_grouped_kfold_never_splits_a_group_across_train_and_val() -> None:
    groups = np.array(["A", "A", "B", "B", "C", "C", "D", "D"])
    X = np.arange(len(groups)).reshape(-1, 1)

    splits = grouped_kfold_indices(X, groups, n_splits=4, group_by="sample_id")
    for train_idx, val_idx in splits:
        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])
        assert train_groups.isdisjoint(val_groups)


def test_grouped_kfold_covers_every_index_exactly_once() -> None:
    groups = np.array(["A", "A", "B", "B", "C", "C", "D", "D"])
    X = np.arange(len(groups)).reshape(-1, 1)

    seen = []
    splits = grouped_kfold_indices(X, groups, n_splits=4, group_by="sample_id")
    for _, val_idx in splits:
        seen.extend(val_idx.tolist())

    assert sorted(seen) == list(range(len(groups)))
