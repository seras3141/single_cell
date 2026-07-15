"""Regression metrics for the feature_to_mcherry baselines."""

from __future__ import annotations

from typing import Dict, List, Union

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, tau: float) -> float:
    """Mean pinball (quantile) loss at the given tau.

    ``L(y, y_hat) = max(tau * (y - y_hat), (tau - 1) * (y - y_hat))``, averaged.
    """
    diff = y_true - y_pred
    return float(np.mean(np.maximum(tau * diff, (tau - 1) * diff)))


def quantile_crossing_rate(y_pred: np.ndarray) -> float:
    """Fraction of rows where predicted quantiles are not non-decreasing across columns.

    Expects ``y_pred`` columns ordered by increasing tau (e.g. [p75, p90, p95]). Not a
    meaningful diagnostic for a mean predictor (e.g. Ridge) — callers should caption it
    accordingly rather than omit it, for uniform reporting across models.
    """
    if y_pred.shape[1] < 2:
        return 0.0
    violations = np.any(np.diff(y_pred, axis=1) < 0, axis=1)
    return float(np.mean(violations))


def per_target_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    taus: List[float],
    target_names: List[str],
) -> List[Dict[str, Union[str, float]]]:
    """Compute MAE, R2, and pinball loss per target column.

    Parameters
    ----------
    y_true, y_pred : np.ndarray, shape (n_samples, n_targets)
    taus : list[float]
        Quantile level for each target column, same order as ``target_names``.
    target_names : list[str]
        Target column names, same order as the columns of ``y_true``/``y_pred``.

    Returns
    -------
    list[dict]
        One dict per target column with keys ``target``, ``tau``, ``mae``, ``r2``,
        ``pinball_loss``.
    """
    results: List[Dict[str, Union[str, float]]] = []
    for i, (tau, name) in enumerate(zip(taus, target_names)):
        results.append(
            {
                "target": name,
                "tau": tau,
                "mae": float(mean_absolute_error(y_true[:, i], y_pred[:, i])),
                "r2": float(r2_score(y_true[:, i], y_pred[:, i])),
                "pinball_loss": pinball_loss(y_true[:, i], y_pred[:, i], tau),
            }
        )
    return results
