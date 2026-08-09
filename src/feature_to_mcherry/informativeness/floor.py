"""Multivariate performance floor: linear (Ridge) and nonlinear (GBM) models.

Both models are fit under grouped cross-validation using morphology-only features,
to bound the R2/MAE any downstream deep-feature model could improve on. Each target
column is predicted independently, since the floor is about achievable per-target
fit, not joint quantile structure (that is Stage 3's concern, not this gate's).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
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
    feature_importances: Optional[pd.DataFrame] = None
    """Per-(target, feature) importance averaged across CV folds, columns
    ``target, feature, importance_mean, importance_std`` — only populated when the
    fitted estimator exposes ``feature_importances_`` (e.g. LightGBM/GBM); ``None``
    for models that don't (e.g. Ridge)."""


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


def _extract_feature_importances(model: Any, n_targets: int) -> Optional[np.ndarray]:
    """Per-target ``feature_importances_`` from a fitted ``MultiOutputRegressor``.

    Returns an array shaped ``(n_targets, n_features)``, or ``None`` if ``model``
    isn't a ``MultiOutputRegressor`` with one estimator per target, or any of its
    per-target estimators doesn't expose ``feature_importances_`` (e.g. Ridge).
    """
    estimators = getattr(model, "estimators_", None)
    if estimators is None or len(estimators) != n_targets:
        return None
    importances = []
    for estimator in estimators:
        values = getattr(estimator, "feature_importances_", None)
        if values is None:
            return None
        importances.append(np.asarray(values))
    return np.stack(importances)


def _feature_importances_dataframe(
    importances_per_fold: List[np.ndarray],
    target_names: List[str],
    feature_names: List[str],
) -> Optional[pd.DataFrame]:
    """Average per-fold ``(n_targets, n_features)`` importance arrays into a long
    DataFrame, or ``None`` if no fold produced any."""
    if not importances_per_fold:
        return None
    stacked = np.stack(importances_per_fold)  # (n_folds, n_targets, n_features)
    mean_importance = stacked.mean(axis=0)
    std_importance = stacked.std(axis=0)
    rows = []
    for j, target_name in enumerate(target_names):
        for k, feature_name in enumerate(feature_names):
            rows.append(
                {
                    "target": target_name,
                    "feature": feature_name,
                    "importance_mean": mean_importance[j, k],
                    "importance_std": std_importance[j, k],
                }
            )
    return pd.DataFrame(rows)


def _run_model(
    model_factory: Callable[[], Any],
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    group_by: str,
    taus: List[float],
    target_names: List[str],
    feature_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]], Optional[pd.DataFrame]]:
    """Grouped-CV fit/predict, returning out-of-fold predictions, pooled metrics, and
    (when ``feature_names`` is given and the fitted model exposes
    ``feature_importances_``) a fold-averaged feature-importances table."""
    oof_predictions = np.full_like(y, np.nan, dtype=float)
    importances_per_fold: List[np.ndarray] = []
    for train_idx, val_idx in grouped_kfold_indices(X, groups, n_splits, group_by):
        model = model_factory()
        model.fit(X[train_idx], y[train_idx])
        oof_predictions[val_idx] = np.asarray(model.predict(X[val_idx]))

        if feature_names is not None:
            fold_importances = _extract_feature_importances(model, len(target_names))
            if fold_importances is not None:
                importances_per_fold.append(fold_importances)

    if np.isnan(oof_predictions).any():
        raise RuntimeError(
            "Some cells never landed in a validation fold during the floor CV; "
            "check that grouped_kfold_indices covers every index exactly once."
        )

    pooled_metrics = per_target_regression_metrics(
        y, oof_predictions, taus, target_names
    )

    feature_importances_df = None
    if feature_names is not None:
        feature_importances_df = _feature_importances_dataframe(
            importances_per_fold, target_names, feature_names
        )
        if feature_importances_df is None:
            logger.info(
                "Fitted model does not expose feature_importances_; skipping "
                "feature-importance collection."
            )

    return oof_predictions, pooled_metrics, feature_importances_df


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
    feature_names: Optional[List[str]] = None,
) -> Dict[str, FloorResult]:
    """Fit the linear (Ridge) and nonlinear (GBM) floor models under grouped CV.

    Ridge decomposes independently per output column for multi-output ``y`` (its
    scikit-learn implementation solves each column's least-squares problem
    separately), so it is used directly. The nonlinear model has no native
    multi-output support, so it is wrapped in ``MultiOutputRegressor`` to fit one
    independent regressor per target column.

    ``feature_names``, if given, enables fold-averaged feature-importance collection
    on the nonlinear result (``FloorResult.feature_importances``) — not requested for
    the linear (Ridge) result, since Ridge has no ``feature_importances_`` attribute.

    Returns
    -------
    dict[str, FloorResult]
        Keys ``"linear"`` and ``"nonlinear"``.
    """
    linear_oof, linear_metrics, _ = _run_model(
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
    nonlinear_oof, nonlinear_metrics, nonlinear_importances = _run_model(
        nonlinear_factory,
        X,
        y,
        groups,
        n_splits,
        group_by,
        taus,
        target_names,
        feature_names=feature_names,
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
            feature_importances=nonlinear_importances,
        ),
    }
