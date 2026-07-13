"""Noise-ceiling estimate: target reproducibility across replicate wells.

A low performance floor next to a healthy noise ceiling is not the same as a low
ceiling (label noise) — the two are kept separate in the results bundle. Uses
one-way random-effects ICC(1) (Shrout & Fleiss variance decomposition) on per-well
aggregates, grouped by drug/dose condition from the plate layout.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from src.dataset_analysis.layout import build_plate_annotation_dataframe

logger = logging.getLogger(__name__)

METHOD = "icc1_variance_decomposition"
NO_LAYOUT_REASON = "plate_layout_json not provided or file not found"
INSUFFICIENT_REPLICATES_REASON = (
    "fewer than 2 conditions have >=2 replicate wells with matched targets"
)
DEGENERATE_REASON = "degenerate variance decomposition (zero denominator)"


def _condition_id(row: pd.Series) -> Optional[str]:
    """Build a condition identifier (drug/control + dose) from one annotation row."""
    if row.get("content") == "drug":
        return f"drug:{row['drug']}@{row['concentration_uM']}"
    if row.get("content") == "control":
        return f"control:{row['control']}"
    return None


def _one_way_icc(values_by_condition: List[np.ndarray]) -> float:
    """One-way random-effects ICC(1) via ANOVA variance decomposition.

    Handles unbalanced group sizes via the standard harmonic-mean-style correction
    (Shrout & Fleiss). ``values_by_condition`` is one array of well-level aggregate
    target values per condition, each with >=2 replicate wells.
    """
    group_sizes = np.array([len(v) for v in values_by_condition], dtype=float)
    n_conditions = len(values_by_condition)
    total_n = int(group_sizes.sum())

    all_values = np.concatenate(values_by_condition)
    grand_mean = all_values.mean()

    ssb = sum(
        n * (v.mean() - grand_mean) ** 2
        for n, v in zip(group_sizes, values_by_condition)
    )
    ssw = sum(((v - v.mean()) ** 2).sum() for v in values_by_condition)

    df_b = n_conditions - 1
    df_w = total_n - n_conditions
    if df_b <= 0 or df_w <= 0:
        return float("nan")

    msb = ssb / df_b
    msw = ssw / df_w
    k0 = (total_n - (group_sizes**2).sum() / total_n) / df_b

    denominator = msb + (k0 - 1) * msw
    if denominator == 0:
        return float("nan")
    return float((msb - msw) / denominator)


def _nan_ceiling_rows(target_columns: List[str], reason: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "target": target,
                "method": METHOD,
                "ceiling": float("nan"),
                "n_conditions": 0,
                "n_replicate_wells": 0,
                "reason": reason,
            }
            for target in target_columns
        ]
    )


def compute_noise_ceiling(
    targets_df: pd.DataFrame,
    target_columns: List[str],
    plate_layout_json: Optional[str],
    sample_id_column: str = "sample_id",
) -> pd.DataFrame:
    """Estimate a per-target noise ceiling from replicate-well agreement.

    Parameters
    ----------
    targets_df : pd.DataFrame
        Must contain ``sample_id_column`` and ``target_columns`` (e.g. the output of
        :func:`feature_to_mcherry.data.loaders.load_targets`).
    target_columns : list[str]
    plate_layout_json : str, optional
        Path to a plate-layout JSON (see ``config/MF5v1_plate_layout.json``). If
        ``None`` or the file does not exist, every target's ceiling is reported as
        ``NaN`` with a reason rather than a fabricated number.
    sample_id_column : str
        Column in ``targets_df`` holding the well id (e.g. ``"C09"``), matched
        against the plate layout's ``well_id``.

    Returns
    -------
    pd.DataFrame
        Columns: ``target``, ``method``, ``ceiling``, ``n_conditions``,
        ``n_replicate_wells``, ``reason`` (empty string when ``ceiling`` is finite).
    """
    if plate_layout_json is None or not Path(plate_layout_json).exists():
        logger.warning(
            "Noise ceiling unavailable: %s (plate_layout_json=%r)",
            NO_LAYOUT_REASON,
            plate_layout_json,
        )
        return _nan_ceiling_rows(target_columns, NO_LAYOUT_REASON)

    annotation = build_plate_annotation_dataframe(plate_layout_json)
    annotation["condition_id"] = annotation.apply(_condition_id, axis=1)

    well_aggregates = (
        targets_df.groupby(sample_id_column)[target_columns].mean().reset_index()
    )
    well_aggregates = well_aggregates.merge(
        annotation[["well_id", "condition_id"]],
        left_on=sample_id_column,
        right_on="well_id",
        how="left",
    )
    well_aggregates = well_aggregates.dropna(subset=["condition_id"])

    rows = []
    for target in target_columns:
        condition_groups: List[np.ndarray] = []
        for _, group in well_aggregates.groupby("condition_id"):
            values = group[target].dropna().to_numpy()
            if len(values) >= 2:
                condition_groups.append(values)

        n_conditions = len(condition_groups)
        n_replicate_wells = sum(len(g) for g in condition_groups)

        if n_conditions < 2:
            rows.append(
                {
                    "target": target,
                    "method": METHOD,
                    "ceiling": float("nan"),
                    "n_conditions": n_conditions,
                    "n_replicate_wells": n_replicate_wells,
                    "reason": INSUFFICIENT_REPLICATES_REASON,
                }
            )
            continue

        ceiling = _one_way_icc(condition_groups)
        rows.append(
            {
                "target": target,
                "method": METHOD,
                "ceiling": ceiling,
                "n_conditions": n_conditions,
                "n_replicate_wells": n_replicate_wells,
                "reason": "" if np.isfinite(ceiling) else DEGENERATE_REASON,
            }
        )

    result = pd.DataFrame(rows)
    logger.info("Noise ceiling estimates: %s", result.to_dict("records"))
    return result
