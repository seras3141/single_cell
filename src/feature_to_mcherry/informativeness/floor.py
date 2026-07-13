"""Multivariate performance floor: linear (Ridge) and nonlinear (GBM) models.

Both models are fit under grouped cross-validation using morphology-only features,
to bound the R2/MAE any downstream deep-feature model could improve on. Each target
column is predicted independently, since the floor is about achievable per-target
fit, not joint quantile structure (that is Stage 3's concern, not this gate's).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

from ..evaluation.cv import grouped_kfold_indices
from ..evaluation.metrics import per_target_regression_metrics
from ..models.ridge import RidgeMeanBaseline

logger = logging.getLogger(__name__)


@dataclass
class FloorResult:
    """Grouped-CV out-of-fold floor results for one model."""

    model_name: str
    backend: str
    pooled_metrics: List[Dict[str, Any]]
    oof_predictions: np.ndarray


def _make_nonlinear_regressor(backend: str) -> Tuple[Callable[[], Any], str]:
    """Return (model_factory, backend_used) for the nonlinear floor model.

    ``backend="auto"`` uses LightGBM if importable, else falls back to
    ``sklearn.ensemble.GradientBoostingRegressor``. Never hard-fails on a missing
    optional dependency unless the caller explicitly requested ``"lightgbm"``.
    """
    if backend == "sklearn":
        return (
            lambda: MultiOutputRegressor(GradientBoostingRegressor()),
            "sklearn",
        )

    try:
        import lightgbm as lgb
    except ImportError:
        if backend == "lightgbm":
            raise ValueError(
                "nonlinear_backend='lightgbm' was requested but lightgbm is not "
                "installed. Install the 'informativeness' extra, or set "
                "nonlinear_backend to 'sklearn' or 'auto'."
            )
        logger.warning(
            "lightgbm not importable; falling back to "
            "sklearn.ensemble.GradientBoostingRegressor for the nonlinear floor model"
        )
        return (
            lambda: MultiOutputRegressor(GradientBoostingRegressor()),
            "sklearn",
        )

    return (
        lambda: MultiOutputRegressor(lgb.LGBMRegressor(verbosity=-1)),
        "lightgbm",
    )


def _run_model(
    model_factory: Callable[[], Any],
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    group_by: str,
    taus: List[float],
    target_names: List[str],
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Grouped-CV fit/predict, returning out-of-fold predictions and pooled metrics."""
    oof_predictions = np.full_like(y, np.nan, dtype=float)
    for train_idx, val_idx in grouped_kfold_indices(X, groups, n_splits, group_by):
        model = model_factory()
        model.fit(X[train_idx], y[train_idx])
        oof_predictions[val_idx] = np.asarray(model.predict(X[val_idx]))

    if np.isnan(oof_predictions).any():
        raise RuntimeError(
            "Some cells never landed in a validation fold during the floor CV; "
            "check that grouped_kfold_indices covers every index exactly once."
        )

    pooled_metrics = per_target_regression_metrics(
        y, oof_predictions, taus, target_names
    )
    return oof_predictions, pooled_metrics


def compute_floor(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    target_names: List[str],
    taus: List[float],
    group_by: str,
    n_splits: int,
    ridge_alpha: float,
    nonlinear_backend: str,
) -> Dict[str, FloorResult]:
    """Fit the linear (Ridge) and nonlinear (GBM) floor models under grouped CV.

    Ridge decomposes independently per output column for multi-output ``y`` (its
    scikit-learn implementation solves each column's least-squares problem
    separately), so it is used directly. The nonlinear model has no native
    multi-output support, so it is wrapped in ``MultiOutputRegressor`` to fit one
    independent regressor per target column.

    Returns
    -------
    dict[str, FloorResult]
        Keys ``"linear"`` and ``"nonlinear"``.
    """
    linear_oof, linear_metrics = _run_model(
        lambda: RidgeMeanBaseline(alpha=ridge_alpha),
        X,
        y,
        groups,
        n_splits,
        group_by,
        taus,
        target_names,
    )
    logger.info("Linear (ridge) floor pooled metrics: %s", linear_metrics)

    nonlinear_factory, backend_used = _make_nonlinear_regressor(nonlinear_backend)
    nonlinear_oof, nonlinear_metrics = _run_model(
        nonlinear_factory,
        X,
        y,
        groups,
        n_splits,
        group_by,
        taus,
        target_names,
    )
    logger.info(
        "Nonlinear (%s) floor pooled metrics: %s", backend_used, nonlinear_metrics
    )

    return {
        "linear": FloorResult(
            model_name="ridge",
            backend="sklearn",
            pooled_metrics=linear_metrics,
            oof_predictions=linear_oof,
        ),
        "nonlinear": FloorResult(
            model_name="gradient_boosting",
            backend=backend_used,
            pooled_metrics=nonlinear_metrics,
            oof_predictions=nonlinear_oof,
        ),
    }
