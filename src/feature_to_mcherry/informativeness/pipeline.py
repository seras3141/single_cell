"""Orchestration: load, select morphology features, univariate + floor + noise
ceiling, write the results bundle (CSVs, JSON, report, figures)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ..data.contract import CELL_KEY, taus_from_target_columns
from ..data.join import build_matrix
from ..data.loaders import (
    load_features,
    load_features_from_directory,
    load_targets,
    load_targets_from_directory,
)
from .config import InformativenessConfig
from .features import select_morphology_features
from .floor import FloorResult, compute_floor
from .noise_ceiling import compute_noise_ceiling
from .plots import write_figures
from .report import write_report
from .univariate import compute_univariate_associations

logger = logging.getLogger(__name__)


@dataclass
class ResultsBundle:
    """Full output of :func:`run`."""

    n_cells: int
    n_features_all: int
    n_features_clean: int
    feature_names_all: List[str]
    feature_names_clean: List[str]
    suspect_feature_names: List[str]
    target_columns: List[str]
    univariate: pd.DataFrame
    floor_metrics: pd.DataFrame
    noise_ceiling: pd.DataFrame
    figures: Dict[str, List[Path]] = field(default_factory=dict)


def _floor_metrics_dataframe(
    floor_results: Dict[str, Dict[str, FloorResult]],
) -> pd.DataFrame:
    """Flatten {variant: {model_key: FloorResult}} into one long DataFrame."""
    rows = []
    for variant, models in floor_results.items():
        for result in models.values():
            for metrics in result.pooled_metrics:
                rows.append(
                    {
                        "variant": variant,
                        "model": result.model_name,
                        "backend": result.backend,
                        "target": metrics["target"],
                        "tau": metrics["tau"],
                        "mae": metrics["mae"],
                        "r2": metrics["r2"],
                        "pinball_loss": metrics["pinball_loss"],
                    }
                )
    return pd.DataFrame(rows)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return str(value)


def _write_summary_json(bundle: ResultsBundle, output_dir: Path) -> None:
    summary = {
        "n_cells": bundle.n_cells,
        "n_features_all": bundle.n_features_all,
        "n_features_clean": bundle.n_features_clean,
        "suspect_feature_names": bundle.suspect_feature_names,
        "target_columns": bundle.target_columns,
        "floor_metrics": bundle.floor_metrics.to_dict("records"),
        "noise_ceiling": bundle.noise_ceiling.to_dict("records"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default)
    )


def run(config: InformativenessConfig) -> ResultsBundle:
    """Run the full morphology-informativeness feasibility gate."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_path = Path(config.target_csv)
    if target_path.is_dir():
        targets_df = load_targets_from_directory(
            target_path, target_columns=config.target_columns
        )
    else:
        targets_df = load_targets(target_path, target_columns=config.target_columns)

    feature_path = Path(config.feature_csv)
    if feature_path.is_dir():
        features_df_full = load_features_from_directory(
            feature_path,
            id_column=config.id_column,
            sample_id_column=config.sample_id_column,
            timepoint_column=config.timepoint_column,
            z_index_column=config.z_index_column,
        )
    else:
        features_df_full = load_features(
            feature_path,
            id_column=config.id_column,
            sample_id_column=config.sample_id_column,
            timepoint_column=config.timepoint_column,
            z_index_column=config.z_index_column,
        )

    candidate_columns = [
        column for column in features_df_full.columns if column not in CELL_KEY
    ]
    selection = select_morphology_features(
        candidate_columns,
        config.morphology_feature_patterns,
        config.suspect_feature_patterns,
    )
    features_df_selected = features_df_full[CELL_KEY + selection.all_columns]

    X, y, groups, feature_names = build_matrix(
        features_df_selected,
        targets_df,
        target_columns=config.target_columns,
        group_by=config.group_by,
    )
    taus = taus_from_target_columns(config.target_columns)

    univariate_df = compute_univariate_associations(
        X, y, groups, feature_names, config.target_columns
    )

    floor_results: Dict[str, Dict[str, FloorResult]] = {
        "with_suspect": compute_floor(
            X,
            y,
            groups,
            config.target_columns,
            taus,
            config.group_by,
            config.n_splits,
            config.ridge_alpha,
            config.nonlinear_backend,
        )
    }

    clean_mask = [name in selection.clean_columns for name in feature_names]
    feature_names_clean = [
        name for name, keep in zip(feature_names, clean_mask) if keep
    ]
    if feature_names_clean:
        X_clean = X[:, clean_mask]
        floor_results["without_suspect"] = compute_floor(
            X_clean,
            y,
            groups,
            config.target_columns,
            taus,
            config.group_by,
            config.n_splits,
            config.ridge_alpha,
            config.nonlinear_backend,
        )
    else:
        logger.warning(
            "No 'clean' (non-suspect) morphology features remain; skipping the "
            "without_suspect floor variant."
        )

    floor_metrics_df = _floor_metrics_dataframe(floor_results)

    noise_ceiling_df = compute_noise_ceiling(
        targets_df,
        config.target_columns,
        config.plate_layout_json,
        sample_id_column="sample_id",
    )

    figures = write_figures(
        output_dir,
        univariate_df,
        X,
        y,
        feature_names,
        config.target_columns,
        floor_metrics_df,
        noise_ceiling_df,
        config.top_k_features,
    )

    bundle = ResultsBundle(
        n_cells=len(y),
        n_features_all=len(selection.all_columns),
        n_features_clean=len(selection.clean_columns),
        feature_names_all=feature_names,
        feature_names_clean=feature_names_clean,
        suspect_feature_names=selection.suspect_columns,
        target_columns=config.target_columns,
        univariate=univariate_df,
        floor_metrics=floor_metrics_df,
        noise_ceiling=noise_ceiling_df,
        figures=figures,
    )

    univariate_df.to_csv(output_dir / "univariate_correlations.csv", index=False)
    floor_metrics_df.to_csv(output_dir / "floor_metrics.csv", index=False)
    noise_ceiling_df.to_csv(output_dir / "noise_ceiling.csv", index=False)
    _write_summary_json(bundle, output_dir)
    write_report(bundle, output_dir)

    logger.info(
        "Morphology-informativeness gate complete: n_cells=%d, n_features_all=%d, "
        "n_features_clean=%d. Outputs written to %s",
        bundle.n_cells,
        bundle.n_features_all,
        bundle.n_features_clean,
        output_dir,
    )

    return bundle
