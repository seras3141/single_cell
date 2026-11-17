"""Fold spread and nested R² for a finished baselines run, from its saved files."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from ..config import FeatureToMcherryConfig
from .nested_r2 import nested_r2, per_fold_r2
from .run_matrix import load_run_matrix

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
    _, y, _, metadata = load_run_matrix(config)
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
