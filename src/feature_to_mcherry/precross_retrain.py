"""Retrain on the pre-confluence window only and compare with whole-window training.

Runs per culture and model (all leave-one-well-out):
  A. train and evaluate on pre-window rows only;
  B. train and evaluate on all timepoints;
  C. the out-of-fold predictions of B, scored on pre-window rows only;
  D. the out-of-fold predictions of B, scored on post-window rows only.

A well's pre-window cutoff is the earlier of the DMSO well's ``t_cross_peak`` and its
own (``dataset_design._earlier_crossing``); a well that never crosses is bounded by
DMSO alone. Targets are raw. Per-well R² is written for runs A, B and C, not D.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import load_config
from .data.contract import taus_from_target_columns
from .dataset_design import _earlier_crossing
from .evaluation.run_matrix import load_run_matrix
from .informativeness.floor import compute_floor

logger = logging.getLogger(__name__)

MIN_TEST_CELLS_PER_FOLD = 50
MODELS = {"ridge": "linear", "lightgbm": "nonlinear"}
CELL_COLUMNS = ["sample_id", "timepoint", "cell_id"]


def top_dose_wells(summary: pd.DataFrame, experiment: str) -> List[str]:
    """Non-DMSO wells at dose rank 1 of one culture."""
    rows = summary[summary["experiment"] == experiment]
    is_dmso = rows["is_dmso"].astype(str).str.lower() == "true"
    top = rows[(rows["dose_rank"] == 1) & ~is_dmso]
    return sorted(top["well"].astype(str))


def window_cutoffs(
    summary: pd.DataFrame, experiment: str, dmso_well: str
) -> Dict[str, Optional[float]]:
    """Last pre-window timepoint per well (``None``: no collapse, nothing to cut)."""
    rows = summary[summary["experiment"] == experiment].set_index("well")
    if rows.index.duplicated().any():
        raise ValueError(f"{experiment}: a well appears more than once in the summary")
    if dmso_well not in rows.index:
        raise ValueError(
            f"{experiment}: DMSO well {dmso_well!r} is not in the summary "
            f"(wells: {sorted(rows.index)})"
        )
    dmso_cross = rows.loc[dmso_well, "t_cross_peak"]
    return {
        str(well): _earlier_crossing(dmso_cross, row["t_cross_peak"])
        for well, row in rows.iterrows()
    }


def pre_window_mask(
    metadata: pd.DataFrame, cutoffs: Dict[str, Optional[float]]
) -> np.ndarray:
    unmapped = sorted(set(metadata["sample_id"].astype(str)) - set(cutoffs))
    if unmapped:
        raise ValueError(f"wells without a window cutoff: {unmapped}")
    timepoint = pd.to_numeric(metadata["timepoint"], errors="raise").to_numpy()
    cutoff = metadata["sample_id"].astype(str).map(cutoffs).astype(float).to_numpy()
    return np.isnan(cutoff) | (timepoint <= cutoff)


def _r2(truth: np.ndarray, pred: np.ndarray) -> float:
    total = float(np.sum((truth - truth.mean()) ** 2))
    if len(truth) < 2 or total == 0.0:
        return float("nan")
    return 1.0 - float(np.sum((truth - pred) ** 2)) / total


def _counts(metadata: pd.DataFrame) -> Dict[str, int]:
    return {
        "n_rows": len(metadata),
        "n_cells": len(metadata[CELL_COLUMNS].drop_duplicates()),
        "n_wells": metadata["sample_id"].nunique(),
        "n_timepoints": metadata["timepoint"].nunique(),
    }


def _pooled_rows(
    base: Dict[str, Any],
    metadata: pd.DataFrame,
    y: np.ndarray,
    oof: np.ndarray,
    target_names: Sequence[str],
) -> List[Dict[str, Any]]:
    counts = _counts(metadata)
    return [
        {
            **base,
            "target": target,
            **counts,
            "r2": _r2(y[:, j], oof[:, j]),
            "mae": float(np.mean(np.abs(y[:, j] - oof[:, j]))),
        }
        for j, target in enumerate(target_names)
    ]


def _per_well_rows(
    base: Dict[str, Any],
    metadata: pd.DataFrame,
    y: np.ndarray,
    oof: np.ndarray,
    target_names: Sequence[str],
) -> List[Dict[str, Any]]:
    rows = []
    wells = metadata["sample_id"].to_numpy()
    for well in sorted(set(wells)):
        index = np.flatnonzero(wells == well)
        for j, target in enumerate(target_names):
            rows.append(
                {
                    **base,
                    "target": target,
                    "held_out_well": well,
                    "n_rows": len(index),
                    "r2": _r2(y[index, j], oof[index, j]),
                }
            )
    return rows


def _fold_rows(base: Dict[str, Any], metadata: pd.DataFrame) -> List[Dict[str, Any]]:
    rows = []
    for well, held_out in metadata.groupby("sample_id"):
        train = metadata[metadata["sample_id"] != well]
        n_cells = len(held_out[CELL_COLUMNS].drop_duplicates())
        rows.append(
            {
                **base,
                "held_out_well": well,
                "n_test_rows": len(held_out),
                "n_test_cells": n_cells,
                "n_train_rows": len(train),
                "n_train_wells": train["sample_id"].nunique(),
                "warning": (
                    f"fewer than {MIN_TEST_CELLS_PER_FOLD} test cells"
                    if n_cells < MIN_TEST_CELLS_PER_FOLD
                    else ""
                ),
            }
        )
    return rows


def _leave_one_well_out(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    target_names: Sequence[str],
    ridge_alpha: float,
) -> Dict[str, np.ndarray]:
    n_wells = len(set(groups.tolist()))
    floor = compute_floor(
        X,
        y,
        groups,
        list(target_names),
        taus_from_target_columns(list(target_names)),
        "sample_id",
        n_wells,
        ridge_alpha,
        "auto",
    )
    return {name: floor[kind].oof_predictions for name, kind in MODELS.items()}


def run_culture(
    culture: str,
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    target_names: Sequence[str],
    cutoffs: Dict[str, Optional[float]],
    ridge_alpha: float = 1.0,
) -> Dict[str, pd.DataFrame]:
    """Runs A to D for one culture; ``metadata`` is row-aligned with ``X`` and ``y``."""
    metadata = metadata.reset_index(drop=True)
    groups = metadata["sample_id"].to_numpy()
    pre = pre_window_mask(metadata, cutoffs)
    pre_wells = metadata.loc[pre, "sample_id"].nunique()
    if pre_wells < 2:
        raise ValueError(
            f"{culture}: {pre_wells} well(s) have pre-window rows; leave-one-well-out "
            "needs at least 2"
        )
    pooled: List[Dict[str, Any]] = []
    per_well: List[Dict[str, Any]] = []
    folds: List[Dict[str, Any]] = []

    runs = {
        "A_pre_only": np.flatnonzero(pre),
        "B_whole_window": np.arange(len(metadata)),
    }
    oof_by_run: Dict[str, Dict[str, np.ndarray]] = {}
    for run, rows in runs.items():
        sub_meta = metadata.iloc[rows]
        oof_by_run[run] = _leave_one_well_out(
            X[rows], y[rows], groups[rows], target_names, ridge_alpha
        )
        folds += _fold_rows({"culture": culture, "run": run}, sub_meta)
        for model, oof in oof_by_run[run].items():
            base = {"culture": culture, "model": model, "run": run}
            pooled += _pooled_rows(base, sub_meta, y[rows], oof, target_names)
            per_well += _per_well_rows(base, sub_meta, y[rows], oof, target_names)

    for run, subset in (("C_B_scored_pre", pre), ("D_B_scored_post", ~pre)):
        rows = np.flatnonzero(subset)
        if len(rows) == 0:
            continue
        for model, oof in oof_by_run["B_whole_window"].items():
            base = {"culture": culture, "model": model, "run": run}
            pooled += _pooled_rows(
                base, metadata.iloc[rows], y[rows], oof[rows], target_names
            )
            if run == "C_B_scored_pre":
                per_well += _per_well_rows(
                    base, metadata.iloc[rows], y[rows], oof[rows], target_names
                )

    per_well_table = pd.DataFrame(per_well)
    spread = (
        per_well_table.groupby(["culture", "model", "run", "target"])["r2"]
        .agg(
            well_r2_mean="mean",
            well_r2_sd="std",
            well_r2_min="min",
            well_r2_max="max",
        )
        .reset_index()
    )
    return {
        "pooled": pd.DataFrame(pooled).merge(
            spread, on=["culture", "model", "run", "target"], how="left"
        ),
        "per_well": per_well_table,
        "folds": pd.DataFrame(folds),
    }


def retrain_culture(
    culture: str,
    experiment_dir: Path,
    dmso_well: str,
    config_path: Path,
    summary_csv: Path,
    output_root: Path,
) -> Dict[str, pd.DataFrame]:
    """Load one culture's radiomics run inputs, run A to D and write the tables.

    Features and targets come from the per-slice tree under ``experiment_dir``;
    everything else (excluded features, folds, ridge alpha) from ``config_path``.
    """
    tree = experiment_dir / "inference_tracked" / "cellpose_sam"
    config = load_config(
        yaml_path=config_path,
        overrides=[
            f"feature_csv={tree / 'features_radiomics'}",
            "target_csv="
            f"{experiment_dir / 'mcherry_metrics/cellpose_sam/instance_metrics.csv'}",
        ],
    )
    if config.normalize_to_dmso:
        raise ValueError(
            "pre-window retraining supports raw targets only: set normalize_to_dmso "
            "to false"
        )
    summary = pd.read_csv(summary_csv)
    dropped = top_dose_wells(summary, culture)
    cutoffs = window_cutoffs(summary, culture, dmso_well)
    no_crossing = sorted(well for well, cutoff in cutoffs.items() if cutoff is None)
    logger.info("%s: top-dose wells dropped: %s", culture, dropped)
    logger.info("%s: wells with no own or DMSO crossing: %s", culture, no_crossing)

    X, y, feature_names, metadata = load_run_matrix(config, dropped)
    logger.info("%s: %d rows, %d features", culture, len(y), len(feature_names))
    tables = run_culture(
        culture, X, y, metadata, config.target_columns, cutoffs, config.ridge_alpha
    )

    output_dir = Path(output_root) / culture
    output_dir.mkdir(parents=True, exist_ok=True)
    tables["pooled"].to_csv(output_dir / "pooled_metrics.csv", index=False)
    tables["per_well"].to_csv(output_dir / "per_well_r2.csv", index=False)
    tables["folds"].to_csv(output_dir / "folds.csv", index=False)
    pd.DataFrame(
        {"dropped_top_dose": pd.Series(dropped), "no_crossing": pd.Series(no_crossing)}
    ).to_csv(output_dir / "excluded_and_unbounded_wells.csv", index=False)
    for row in tables["folds"][tables["folds"]["warning"] != ""].itertuples():
        logger.warning(
            "%s run %s: held-out well %s has %d test cells (< %d)",
            culture,
            row.run,
            row.held_out_well,
            row.n_test_cells,
            MIN_TEST_CELLS_PER_FOLD,
        )
    logger.info("Wrote results to %s", output_dir)
    return tables
