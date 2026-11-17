"""Per-fold R² spread and nested (within-timepoint) R² from out-of-fold predictions."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def _sse(truth: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sum((truth - pred) ** 2))


def _total_ss(truth: np.ndarray) -> float:
    return float(np.sum((truth - truth.mean()) ** 2))


def _within_group_ss(truth: np.ndarray, labels: np.ndarray) -> float:
    centred = truth - pd.Series(truth).groupby(labels).transform("mean").to_numpy()
    return float(np.sum(centred**2))


def _r2(sse: float, variance_ss: float) -> float:
    return float("nan") if variance_ss == 0.0 else 1.0 - sse / variance_ss


def per_fold_r2(
    well: np.ndarray,
    truth: np.ndarray,
    pred: np.ndarray,
    well_fold: Dict[str, int],
) -> pd.DataFrame:
    """R² of the held-out predictions of each fold (one row per fold)."""
    fold = pd.Series(well).map(well_fold)
    unassigned = sorted(set(pd.Series(well)[fold.isna()]))
    if unassigned:
        raise ValueError(f"wells without a fold: {unassigned}")
    fold = fold.to_numpy()
    rows: List[dict] = []
    for fold_id in sorted(set(fold)):
        mask = fold == fold_id
        rows.append(
            {
                "fold": fold_id,
                "n_rows": int(mask.sum()),
                "r2": _r2(_sse(truth[mask], pred[mask]), _total_ss(truth[mask])),
            }
        )
    return pd.DataFrame(rows)


def nested_r2(
    well: np.ndarray, timepoint: np.ndarray, truth: np.ndarray, pred: np.ndarray
) -> Dict[str, float]:
    """R² against the overall, per-timepoint and per-(well, timepoint) mean.

    The residuals are the same in all three; only the variance they are compared with
    changes, so the later levels show how much of the pooled R² survives once the
    differences between timepoints (and between wells at a timepoint) are removed. A
    level whose variance is zero is NaN.
    """
    sse = _sse(truth, pred)
    by_timepoint = np.asarray(timepoint, dtype=str)
    by_well_timepoint = np.char.add(np.asarray(well, dtype=str), "|" + by_timepoint)
    return {
        "pooled": _r2(sse, _total_ss(truth)),
        "within_timepoint": _r2(sse, _within_group_ss(truth, by_timepoint)),
        "within_well_timepoint": _r2(sse, _within_group_ss(truth, by_well_timepoint)),
    }
