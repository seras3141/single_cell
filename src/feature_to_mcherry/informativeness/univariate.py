"""Univariate Spearman association between morphology features and mCherry targets.

Spearman (rank) correlation is used rather than Pearson because mCherry intensity
percentiles are skewed. Both pooled and per-group associations are computed: a high
pooled rho next to a near-zero per-group rho signals a batch effect (Simpson's
paradox) rather than genuine per-cell biology.
"""

from __future__ import annotations

import logging
from typing import List, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

MIN_GROUP_SIZE = 3


def _safe_spearmanr(x: np.ndarray, y: np.ndarray) -> tuple:
    """Spearman rho/p-value, or (nan, nan) if either side is constant or too small."""
    if len(x) < MIN_GROUP_SIZE or np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan"), float("nan")
    rho, pvalue = spearmanr(x, y)
    return float(rho), float(pvalue)


def compute_univariate_associations(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: Sequence[str],
    target_names: Sequence[str],
) -> pd.DataFrame:
    """Compute pooled and per-group Spearman rho for every (feature, target) pair.

    Parameters
    ----------
    X : np.ndarray, shape (n_cells, n_features)
    y : np.ndarray, shape (n_cells, n_targets)
    groups : np.ndarray, shape (n_cells,)
        Grouping key (e.g. well) used for the per-group breakdown.
    feature_names, target_names : Sequence[str]

    Returns
    -------
    pd.DataFrame
        Long/tidy: columns ``feature``, ``target``, ``scope`` (``"pooled"`` or
        ``"per_group"``), ``group_id`` (empty string for ``"pooled"`` rows), ``rho``,
        ``pvalue``, ``n``.
    """
    rows: List[dict] = []
    unique_groups = np.unique(groups)

    for j, target in enumerate(target_names):
        y_col = y[:, j]
        for i, feature in enumerate(feature_names):
            x_col = X[:, i]
            rho, pvalue = _safe_spearmanr(x_col, y_col)
            rows.append(
                {
                    "feature": feature,
                    "target": target,
                    "scope": "pooled",
                    "group_id": "",
                    "rho": rho,
                    "pvalue": pvalue,
                    "n": len(x_col),
                }
            )

            for group_id in unique_groups:
                mask = groups == group_id
                n = int(mask.sum())
                if n < MIN_GROUP_SIZE:
                    rho_g, pvalue_g = float("nan"), float("nan")
                else:
                    rho_g, pvalue_g = _safe_spearmanr(x_col[mask], y_col[mask])
                rows.append(
                    {
                        "feature": feature,
                        "target": target,
                        "scope": "per_group",
                        "group_id": str(group_id),
                        "rho": rho_g,
                        "pvalue": pvalue_g,
                        "n": n,
                    }
                )

    result = pd.DataFrame(rows)
    logger.info(
        "Computed univariate associations for %d features x %d targets "
        "(pooled + %d groups)",
        len(feature_names),
        len(target_names),
        len(unique_groups),
    )
    return result


def top_associations(
    univariate_df: pd.DataFrame, target: str, top_k: int, scope: str = "pooled"
) -> pd.DataFrame:
    """Return the top-K features by |rho| for one target, within one scope."""
    subset = univariate_df[
        (univariate_df["target"] == target) & (univariate_df["scope"] == scope)
    ].copy()
    subset["abs_rho"] = subset["rho"].abs()
    subset = subset.sort_values("abs_rho", ascending=False, na_position="last")
    return subset.head(top_k).drop(columns="abs_rho")
