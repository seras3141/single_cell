"""Fold spread and nested R² for a finished baselines run, from its saved files."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from ..config import FeatureToMcherryConfig
from ..data.join import build_matrix_with_metadata
from ..data.loaders import (
    load_features,
    load_features_from_directory,
    load_targets,
    load_targets_from_directory,
)
from ..data.normalize import apply_dmso_normalization
from .nested_r2 import nested_r2, per_fold_r2

logger = logging.getLogger(__name__)


def rebuild_targets_and_metadata(
    config: FeatureToMcherryConfig,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """The run's target matrix and per-row ``CELL_KEY`` metadata, in the run's order."""
    if config.group_by != "sample_id":
        raise ValueError(
            f"group_by={config.group_by!r}: only runs grouped by sample_id (well) "
            "can be decomposed"
        )
    target_path = Path(config.target_csv)
    if target_path.is_dir():
        targets_df = load_targets_from_directory(
            target_path, target_columns=config.target_columns
        )
    else:
        targets_df = load_targets(target_path, target_columns=config.target_columns)
    targets_df, target_columns = apply_dmso_normalization(
        targets_df,
        enabled=config.normalize_to_dmso,
        dmso_well=config.dmso_well,
        target_columns=config.target_columns,
        min_peak_fraction=config.dmso_gate_min_peak_fraction,
        absolute_floor=config.dmso_gate_absolute_floor,
    )
    feature_path = Path(config.feature_csv)
    feature_arguments = dict(
        id_column=config.id_column,
        sample_id_column=config.sample_id_column,
        timepoint_column=config.timepoint_column,
        z_index_column=config.z_index_column,
    )
    if feature_path.is_dir():
        features_df = load_features_from_directory(
            feature_path, pattern=config.feature_pattern, **feature_arguments
        )
    else:
        features_df = load_features(feature_path, **feature_arguments)
    features_df = features_df.drop(
        columns=[c for c in config.exclude_feature_columns if c in features_df.columns]
    )
    _, y, _, _, metadata = build_matrix_with_metadata(
        features_df,
        targets_df,
        target_columns=target_columns,
        group_by=config.group_by,
    )
    return y, metadata


def decompose_oof(
    oof: pd.DataFrame,
    folds: pd.DataFrame,
    y: np.ndarray,
    metadata: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-fold R² and nested R² per target.

    ``oof`` is the long ``oof_predictions.csv`` (``group_id``, ``target_name``,
    ``y_true``, ``y_pred``), ``folds`` the ``fold_assignments.csv``, and ``y`` and
    ``metadata`` the rebuilt targets and per-row cell metadata. Each target's rows must
    hold the same wells in the same order and the same true values as the rebuild, so
    the timepoints belong to the right predictions.
    """
    well_fold = dict(zip(folds["group_id"].astype(str), folds["fold"]))
    metadata_wells = metadata["sample_id"].astype(str).to_numpy()
    spread_rows, nested_rows = [], []
    for j, (target, block) in enumerate(oof.groupby("target_name", sort=False)):
        if len(block) != len(metadata):
            raise ValueError(f"{target}: {len(block)} oof rows, {len(metadata)} cells")
        wells = block["group_id"].astype(str).to_numpy()
        if not np.array_equal(wells, metadata_wells):
            raise ValueError(f"{target}: oof wells do not match the rebuilt cells")
        truth, pred = block["y_true"].to_numpy(), block["y_pred"].to_numpy()
        if not np.allclose(truth, y[:, j]):
            raise ValueError(f"{target}: oof y_true differs from the rebuilt targets")
        for row in per_fold_r2(wells, truth, pred, well_fold).to_dict("records"):
            spread_rows.append({"target": target, **row})
        levels = nested_r2(wells, metadata["timepoint"].to_numpy(), truth, pred)
        nested_rows += [
            {"target": target, "level": level, "r2": value}
            for level, value in levels.items()
        ]
    return pd.DataFrame(spread_rows), pd.DataFrame(nested_rows)


def decompose_run(directory: Path) -> None:
    """Write ``fold_spread.csv`` and ``nested_r2.csv`` into a baselines run."""
    directory = Path(directory)
    record = json.loads((directory / "run_provenance.json").read_text())
    y, metadata = rebuild_targets_and_metadata(
        FeatureToMcherryConfig(**record["config"])
    )
    oof = pd.read_csv(directory / "oof_predictions.csv", dtype={"group_id": str})
    folds = pd.read_csv(directory / "fold_assignments.csv", dtype={"group_id": str})
    spread, nested = decompose_oof(oof, folds, y, metadata)
    spread.to_csv(directory / "fold_spread.csv", index=False)
    nested.to_csv(directory / "nested_r2.csv", index=False)
    logger.info("Wrote fold_spread.csv and nested_r2.csv to %s", directory)
