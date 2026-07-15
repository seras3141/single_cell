"""Grouped cross-validation utilities."""

from __future__ import annotations

import logging
from typing import Iterator, Tuple

import numpy as np
from sklearn.model_selection import GroupKFold

logger = logging.getLogger(__name__)


def validate_group_count(groups: np.ndarray, n_splits: int, group_by: str) -> None:
    """Raise a clear error if there are fewer unique groups than requested folds.

    ``GroupKFold`` itself fails on this with a generic sklearn error; this gives a
    message that names the actual counts and the grouping key involved.
    """
    n_unique = len(np.unique(groups))
    if n_splits > n_unique:
        raise ValueError(
            f"Requested n_splits={n_splits} but only {n_unique} unique group(s) exist "
            f"for group_by={group_by!r}. Reduce n_splits to at most {n_unique}, or "
            "choose a different group_by key with more distinct values."
        )


def grouped_kfold_indices(
    X: np.ndarray, groups: np.ndarray, n_splits: int, group_by: str
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, val_idx) index pairs from GroupKFold, logging fold composition.

    Never splits randomly per-cell — always by the provided ``groups`` array (e.g.
    well or timepoint), so no group straddles both the train and validation side of a
    fold.
    """
    validate_group_count(groups, n_splits, group_by)
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (train_idx, val_idx) in enumerate(splitter.split(X, groups=groups)):
        train_groups = sorted(set(groups[train_idx].tolist()))
        val_groups = sorted(set(groups[val_idx].tolist()))
        logger.info(
            "Fold %d: %d train cells (%d groups), %d val cells (%d groups). "
            "Val groups: %s",
            fold,
            len(train_idx),
            len(train_groups),
            len(val_idx),
            len(val_groups),
            val_groups,
        )
        yield train_idx, val_idx
