"""Join a feature table and a target table into a model-ready matrix."""

from __future__ import annotations

import logging
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

from .contract import CELL_KEY, normalize_cell_key

logger = logging.getLogger(__name__)


def _assert_unique_keys(df: pd.DataFrame, name: str) -> None:
    duplicated = df.duplicated(subset=CELL_KEY, keep=False)
    if duplicated.any():
        dup_keys = df.loc[duplicated, CELL_KEY].drop_duplicates()
        preview = dup_keys.to_dict("records")
        raise ValueError(
            f"{name} table has duplicate cell_key rows {CELL_KEY}: "
            f"{preview[:10]}{'...' if len(preview) > 10 else ''}"
        )


def build_matrix(
    features_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    target_columns: Sequence[str],
    group_by: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Inner-join features and targets on ``CELL_KEY``.

    Parameters
    ----------
    features_df : pd.DataFrame
        Output of :func:`load_features` (``CELL_KEY + feature columns``).
    targets_df : pd.DataFrame
        Output of :func:`load_targets` (``CELL_KEY + target_columns``).
    target_columns : Sequence[str]
        Target column names to extract into ``y``, in order.
    group_by : str
        Column (present after the join) used for grouped cross-validation.

    Returns
    -------
    X : np.ndarray, shape (n_cells, n_features)
    y : np.ndarray, shape (n_cells, n_targets)
    groups : np.ndarray, shape (n_cells,)
    feature_names : list[str]
    """
    features_df = normalize_cell_key(features_df)
    targets_df = normalize_cell_key(targets_df)

    _assert_unique_keys(features_df, "feature")
    _assert_unique_keys(targets_df, "target")

    feature_keys = set(map(tuple, features_df[CELL_KEY].to_numpy()))
    target_keys = set(map(tuple, targets_df[CELL_KEY].to_numpy()))

    dropped_from_features = feature_keys - target_keys
    dropped_from_targets = target_keys - feature_keys

    if dropped_from_features:
        logger.warning(
            "%d/%d feature-table cells have no matching target row; dropped. "
            "Examples: %s",
            len(dropped_from_features),
            len(feature_keys),
            list(dropped_from_features)[:10],
        )
    if dropped_from_targets:
        logger.warning(
            "%d/%d target-table cells have no matching feature row; dropped. "
            "Examples: %s",
            len(dropped_from_targets),
            len(target_keys),
            list(dropped_from_targets)[:10],
        )

    merged = features_df.merge(
        targets_df, on=CELL_KEY, how="inner", validate="one_to_one"
    )

    if merged.empty:
        raise ValueError(
            "Join between feature and target tables produced zero matched cells. "
            "Check that both CSVs describe the same cells (same segmentation source, "
            "same sample_id/timepoint/z_index/cell_id numbering)."
        )

    if group_by not in merged.columns:
        raise ValueError(
            f"group_by column {group_by!r} not present after join; "
            f"available columns: {list(merged.columns)}"
        )

    feature_names = [column for column in features_df.columns if column not in CELL_KEY]

    X = merged[feature_names].to_numpy(dtype=float)
    y = merged[list(target_columns)].to_numpy(dtype=float)
    groups = merged[group_by].to_numpy()

    logger.info(
        "Built matrix: %d matched cells, %d features, %d targets, grouped by %r "
        "(%d unique groups)",
        len(merged),
        len(feature_names),
        len(target_columns),
        group_by,
        len(np.unique(groups)),
    )

    return X, y, groups, feature_names
