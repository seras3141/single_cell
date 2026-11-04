"""Per-(well, timepoint) metrics and low-count timepoints, for the figures over time."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd

from ..data.normalize import compute_confidence_flags

LOW_COUNT_MIN_PEAK_FRACTION = 0.10
LOW_COUNT_ABSOLUTE_FLOOR = 30
_METRIC_COLUMNS = [
    "model",
    "target",
    "sample_id",
    "timepoint",
    "n_rows",
    "n_cells",
    "r2",
    "mae",
]


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    total = float(np.sum((y_true - y_true.mean()) ** 2))
    if len(y_true) < 2 or total == 0.0:
        return float("nan")
    return 1.0 - float(np.sum((y_true - y_pred) ** 2)) / total


def well_timepoint_metrics(
    metadata: pd.DataFrame,
    y: np.ndarray,
    oof_predictions: np.ndarray,
    target_names: Sequence[str],
    model_name: str,
) -> pd.DataFrame:
    """One row per target, well and timepoint: counts, R² and MAE.

    ``metadata`` is row-aligned with ``y`` and ``oof_predictions`` and carries
    ``sample_id``, ``timepoint`` and ``cell_id``. ``n_rows`` counts matrix rows, which
    repeat a cell across z-slices on the per-slice tree; ``n_cells`` counts distinct
    cells. R² is NaN when a group has fewer than two rows or a constant target.
    """
    rows: List[dict] = []
    positions = metadata.groupby(["sample_id", "timepoint"], sort=False).indices
    for (well, timepoint), index in positions.items():
        n_cells = int(metadata["cell_id"].iloc[index].nunique())
        for j, target in enumerate(target_names):
            truth, pred = y[index, j], oof_predictions[index, j]
            rows.append(
                {
                    "model": model_name,
                    "target": target,
                    "sample_id": well,
                    "timepoint": timepoint,
                    "n_rows": len(index),
                    "n_cells": n_cells,
                    "r2": _r2(truth, pred),
                    "mae": float(np.mean(np.abs(truth - pred))),
                }
            )
    table = pd.DataFrame(rows, columns=_METRIC_COLUMNS)
    table["_t"] = pd.to_numeric(table["timepoint"], errors="coerce")
    return (
        table.sort_values(["model", "target", "sample_id", "_t"])
        .drop(columns="_t")
        .reset_index(drop=True)
    )


def first_low_count_timepoints(
    targets_df: pd.DataFrame,
    dmso_well: str,
    min_peak_fraction: float = LOW_COUNT_MIN_PEAK_FRACTION,
    absolute_floor: int = LOW_COUNT_ABSOLUTE_FLOOR,
) -> pd.DataFrame:
    """Per well, the first timepoint with a low distinct-cell count (NaN if none).

    A timepoint is low when the well's distinct-cell count is below
    ``min_peak_fraction`` times its running maximum, or below ``absolute_floor``. The
    gate's third condition (a degraded DMSO reference) is ignored. Computed from the raw
    targets, so a run with the gate off gets the same marks.
    """
    flags = compute_confidence_flags(
        targets_df,
        dmso_well,
        min_peak_fraction=min_peak_fraction,
        absolute_floor=absolute_floor,
    )
    flags = flags.assign(
        low_count=flags["flag_relative"] | flags["flag_absolute_floor"],
        _t=pd.to_numeric(flags["timepoint"], errors="coerce"),
    )
    first = []
    for well, group in flags.groupby("sample_id", sort=True):
        low = group[group["low_count"]].sort_values("_t")
        first.append(
            {
                "sample_id": well,
                "first_low_count_timepoint": (
                    str(low["timepoint"].iloc[0]) if len(low) else None
                ),
            }
        )
    return pd.DataFrame(first, dtype=object)
