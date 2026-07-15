"""Shared constants and the cell-key join contract for feature_to_mcherry."""

from __future__ import annotations

import re
from typing import List

import pandas as pd

TARGET_COLUMNS: List[str] = ["percentile_75", "percentile_90", "percentile_95"]
# 4-column, per-z-slice key: both mcherry_metrics and handcrafted feature CSVs are
# split per z-slice, and cell_id (a 3D-consistent label) recurs across the z-slices
# of one physical cell within a (sample_id, timepoint) — dropping z_index collapses
# distinct slices onto the same key and either mis-joins or raises on duplicates.
CELL_KEY: List[str] = ["sample_id", "timepoint", "z_index", "cell_id"]

_PERCENTILE_COLUMN_RE = re.compile(r"^percentile_(\d+)$")


def taus_from_target_columns(target_columns: List[str]) -> List[float]:
    """Derive quantile levels (tau) from ``percentile_<N>`` column names.

    Parameters
    ----------
    target_columns : list[str]
        mCherry percentile column names, e.g. ``["percentile_75", "percentile_90"]``.

    Returns
    -------
    list[float]
        One tau per target column, in the same order (e.g. ``[0.75, 0.90]``).
    """
    taus = []
    for column in target_columns:
        match = _PERCENTILE_COLUMN_RE.match(column)
        if not match:
            raise ValueError(
                f"Cannot derive a quantile level (tau) from target column {column!r}; "
                "expected the 'percentile_<N>' naming convention used by "
                "mcherry_metrics."
            )
        taus.append(int(match.group(1)) / 100.0)
    return taus


def normalize_cell_key(df: pd.DataFrame) -> pd.DataFrame:
    """Cast the CELL_KEY columns to str so joins aren't broken by int/str mismatches."""
    df = df.copy()
    for column in CELL_KEY:
        df[column] = df[column].astype(str)
    return df
