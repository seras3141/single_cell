"""The per-cell observation unit, defined once.

The mCherry metrics tables this package reads are one row per
``(cell, z_slice)``, typically 3-13 slices per cell. Any statistic taken over the
raw rows therefore weights each cell by how many z-slices it spans -- and because
treatment changes cell morphology, that weighting *correlates with the effect being
measured*. Collapsing to one observation per cell before computing anything is
mandatory, not a refinement.

This module holds the single definition so that separate analyses cannot silently
disagree about what one observation is.

The collapse is a **median across z**. Note that the representative-slice pipeline
instead picks one slice per cell by **sharpness** -- same intent, but per-cell values
from the two are not numerically identical, so results from the two are not directly
comparable number-for-number. See section B5 of
``docs/feature_to_mcherry/plan_dataset_design_assessment.md``.
"""

from __future__ import annotations

import logging
from typing import List, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

# One observation = one tracked cell at one timepoint. Deliberately CELL_KEY *without*
# ``z_index``: the z-slice is what gets collapsed away.
CELL_OBSERVATION_KEY: List[str] = ["sample_id", "timepoint", "cell_id"]

__all__ = ["CELL_OBSERVATION_KEY", "collapse_slices_to_cells"]


def collapse_slices_to_cells(
    frame: pd.DataFrame,
    value_columns: Sequence[str],
    *,
    well_column: str = "sample_id",
    time_column: str = "timepoint",
    cell_id_column: str = "cell_id",
) -> pd.DataFrame:
    """One row per ``(well, timepoint, cell_id)``, values median-reduced over z.

    Args:
        frame: Per-``(cell, z_slice)`` rows carrying the three key columns.
        value_columns: Numeric columns to reduce. Non-key columns not listed are
            dropped, since carrying a slice-level column through a per-cell frame
            invites the very confusion this collapse exists to prevent.

    Returns:
        The collapsed frame: key columns followed by ``value_columns``, in that order.

    Raises:
        ValueError: If any key or value column is absent, or ``value_columns`` is empty.
            There is deliberately no fall-back to the raw rows: silently returning
            slice-weighted data is the bug this function guards against.
    """
    values = list(value_columns)
    if not values:
        raise ValueError("value_columns is empty; nothing to collapse")

    keys = [well_column, time_column, cell_id_column]
    missing = [c for c in keys + values if c not in frame.columns]
    if missing:
        raise ValueError(
            f"frame missing {missing!r}; cannot establish the observation unit "
            f"(columns: {sorted(frame.columns)})"
        )

    # Cast the key columns to str before grouping, for the same reason
    # ``contract.normalize_cell_key`` does it on the join path: a column holding both
    # ``"1"`` and ``1`` describes one cell but groups as two, silently leaving a
    # slice-inflated observation behind -- the exact defect this function exists to
    # prevent. Values are left untouched.
    keyed = frame.copy()
    for column in keys:
        keyed[column] = keyed[column].astype(str)

    collapsed = (
        keyed.groupby(keys, sort=False)[values]  # type: ignore[index]
        .median()
        .reset_index()
    )
    if len(collapsed) < len(frame):
        logger.info(
            "collapsed %d slice rows to %d cell observations (%.2fx) over %d column(s)",
            len(frame),
            len(collapsed),
            len(frame) / max(1, len(collapsed)),
            len(values),
        )
    return collapsed
