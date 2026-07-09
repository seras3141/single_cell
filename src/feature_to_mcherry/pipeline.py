"""Orchestration: load, join, grouped-CV fit, evaluate, and report the two baselines."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd

from .config import FeatureToMcherryConfig
from .data.contract import taus_from_target_columns
from .data.join import build_matrix
from .data.loaders import load_features, load_targets
from .evaluation.cv import grouped_kfold_indices
from .evaluation.metrics import per_target_regression_metrics, quantile_crossing_rate
from .models.linear_quantile import LinearQuantileRegressor
from .models.ridge import RidgeMeanBaseline

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    """Evaluation results for one model, across a full grouped-CV run."""

    model_name: str
    per_fold_metrics: List[List[Dict[str, Any]]]
    pooled_metrics: List[Dict[str, Any]]
    pooled_crossing_rate: float
    oof_predictions: np.ndarray


@dataclass
class ResultsBundle:
    """Full output of :func:`run`: both models' results plus dataset shape info."""

    ridge: ModelResult
    linear_quantile: ModelResult
    n_cells: int
    n_features: int
    feature_names: List[str]


def _sort_quantiles(y_pred: np.ndarray) -> np.ndarray:
    """Sort quantiles ascending per row.

    A post-hoc crossing patch, not a modeling fix.
    """
    return np.sort(y_pred, axis=1)


def _run_model(
    model_factory: Callable[[], Any],
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    group_by: str,
    taus: List[float],
    target_names: List[str],
    apply_sort: bool,
    model_name: str,
) -> ModelResult:
    """Run one model through grouped CV, collecting out-of-fold predictions/metrics."""
    oof_predictions = np.full_like(y, np.nan, dtype=float)
    per_fold_metrics: List[List[Dict[str, Any]]] = []

    for train_idx, val_idx in grouped_kfold_indices(X, groups, n_splits, group_by):
        model = model_factory()
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[val_idx])
        if apply_sort:
            preds = _sort_quantiles(preds)
        oof_predictions[val_idx] = preds
        per_fold_metrics.append(
            per_target_regression_metrics(y[val_idx], preds, taus, target_names)
        )

    if np.isnan(oof_predictions).any():
        raise RuntimeError(
            f"{model_name}: some cells never landed in a validation fold; "
            "check that grouped_kfold_indices covers every index exactly once."
        )

    pooled_metrics = per_target_regression_metrics(
        y, oof_predictions, taus, target_names
    )
    pooled_crossing_rate = quantile_crossing_rate(oof_predictions)

    logger.info("%s pooled out-of-fold metrics: %s", model_name, pooled_metrics)

    return ModelResult(
        model_name=model_name,
        per_fold_metrics=per_fold_metrics,
        pooled_metrics=pooled_metrics,
        pooled_crossing_rate=pooled_crossing_rate,
        oof_predictions=oof_predictions,
    )


def _to_markdown_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a markdown table without a tabulate dependency."""
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    separator = "| " + " | ".join("---" for _ in df.columns) + " |"
    body_rows = [
        "| " + " | ".join(str(value) for value in row) + " |" for row in df.to_numpy()
    ]
    return "\n".join([header, separator, *body_rows])


def _write_report(results: ResultsBundle, output_dir: Path) -> None:
    """Write the baseline-ladder comparison table (CSV + markdown) to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for result in (results.ridge, results.linear_quantile):
        for metrics in result.pooled_metrics:
            rows.append(
                {
                    "model": result.model_name,
                    "target": metrics["target"],
                    "tau": metrics["tau"],
                    "mae": metrics["mae"],
                    "r2": metrics["r2"],
                    "pinball_loss": metrics["pinball_loss"],
                    "quantile_crossing_rate": result.pooled_crossing_rate,
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "baseline_ladder.csv", index=False)

    lines = [
        "# feature_to_mcherry baseline ladder",
        "",
        f"n_cells={results.n_cells}, n_features={results.n_features}",
        "",
        "Ridge predicts a mean, not a quantile — its quantile_crossing_rate is not a "
        "meaningful diagnostic; it is reported for table uniformity only.",
        "",
        _to_markdown_table(table),
        "",
    ]
    (output_dir / "baseline_ladder.md").write_text("\n".join(lines))
    logger.info("Wrote baseline ladder report to %s", output_dir)


def run(config: FeatureToMcherryConfig) -> ResultsBundle:
    """Run the full pipeline: load, join, CV-fit, evaluate, and report both models."""
    targets_df = load_targets(
        Path(config.target_csv), target_columns=config.target_columns
    )
    features_df = load_features(
        Path(config.feature_csv),
        id_column=config.id_column,
        sample_id_column=config.sample_id_column,
        timepoint_column=config.timepoint_column,
    )

    X, y, groups, feature_names = build_matrix(
        features_df,
        targets_df,
        target_columns=config.target_columns,
        group_by=config.group_by,
    )

    taus = taus_from_target_columns(config.target_columns)

    ridge_result = _run_model(
        model_factory=lambda: RidgeMeanBaseline(alpha=config.ridge_alpha),
        X=X,
        y=y,
        groups=groups,
        n_splits=config.n_splits,
        group_by=config.group_by,
        taus=taus,
        target_names=config.target_columns,
        apply_sort=False,
        model_name="ridge",
    )

    linear_quantile_result = _run_model(
        model_factory=lambda: LinearQuantileRegressor(
            taus=taus, alpha=config.quantile_alpha, solver=config.quantile_solver
        ),
        X=X,
        y=y,
        groups=groups,
        n_splits=config.n_splits,
        group_by=config.group_by,
        taus=taus,
        target_names=config.target_columns,
        apply_sort=config.sort_quantiles,
        model_name="linear_quantile",
    )

    results = ResultsBundle(
        ridge=ridge_result,
        linear_quantile=linear_quantile_result,
        n_cells=len(y),
        n_features=len(feature_names),
        feature_names=feature_names,
    )

    _write_report(results, Path(config.output_dir))

    return results
