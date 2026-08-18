"""DMSO control normalization of mCherry targets (robust per-timepoint z-score).

The per-cell mCherry percentiles drift materially over the 72 h time-course, and the
drift is idiosyncratic per experiment/culture. Normalizing each target against the
experiment's DMSO (vehicle) control **at the matched timepoint** removes that shared
baseline, leaving the drug-vs-vehicle effect:

    z = (x - median_DMSO(t)) / MAD_DMSO(t)

using a scaled median-absolute-deviation (normal-consistent, ``x1.4826``) as the robust
scale. This operates on the TARGET table only (morphology features are untouched) and
is a pure DataFrame transform -- no I/O -- so it can be injected between
``load_targets`` and ``build_matrix``.

**Experiment scoping.** ``sample_id`` is the WELL, and distinct cultures reuse the
same DMSO well label (``M11`` / ``N11``). The standard target schema has no experiment
column, so the reference is computed per timepoint within the table passed in: **each
call must cover exactly ONE experiment/culture**, else distinct cultures' DMSO baselines
are pooled. If a single table concatenates several experiments, pass
``experiment_column`` so the reference is computed per (experiment, timepoint).

See ``docs/feature_to_mcherry/plan_dmso_normalization_implementation.md``.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from .contract import TARGET_COLUMNS

logger = logging.getLogger(__name__)

# Scaled MAD: median(|x - median(x)|) * 1.4826 is a normal-consistent estimator of the
# standard deviation (robust to outliers / bright cells).
MAD_SCALE = 1.4826


def _scaled_mad(values: np.ndarray) -> float:
    """Scaled MAD (``x1.4826``) of the finite values; NaN if none are finite."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    median = np.median(values)
    return float(MAD_SCALE * np.median(np.abs(values - median)))


def compute_dmso_reference(
    targets_df: pd.DataFrame,
    dmso_well: str,
    target_columns: Optional[List[str]] = None,
    sample_id_column: str = "sample_id",
    timepoint_column: str = "timepoint",
    experiment_column: Optional[str] = None,
    min_cells: int = 2,
) -> pd.DataFrame:
    """Per-timepoint DMSO reference statistics (median + scaled MAD) for each target.

    Median, scaled MAD and the count all use the **same finite-value mask** per target,
    so a partial-NaN group cannot yield a finite MAD alongside a NaN median.

    Parameters
    ----------
    targets_df : pd.DataFrame
        Target table with at least ``sample_id_column``, ``timepoint_column`` and the
        target columns. ``sample_id`` holds the well label.
    dmso_well : str
        Well label of the DMSO (vehicle) control for this experiment (e.g. ``"M11"``).
    target_columns : list[str], optional
        Columns to summarise. Defaults to ``TARGET_COLUMNS``.
    sample_id_column, timepoint_column : str
        Well and timepoint column names. Defaults match the mcherry_metrics schema.
    experiment_column : str, optional
        If given, the reference is grouped by ``[experiment_column, timepoint_column]``
        so a table spanning several experiments (which reuse DMSO well labels) is scoped
        correctly. If omitted, the whole table is assumed to be a single experiment.
    min_cells : int
        Minimum number of FINITE DMSO values (per target) at a timepoint for its
        reference to be ``valid``.

    Returns
    -------
    pd.DataFrame
        One row per group key, columns: the group-key column(s), ``median_<col>``,
        ``mad_<col>`` (one pair per target), ``n_dmso`` (min finite count over targets)
        and ``valid`` (every target has >= ``min_cells`` finite values, a finite median
        and ``mad_<col> > 0``). Sorted by the group key(s).

    Raises
    ------
    ValueError
        If ``dmso_well`` is not present in ``sample_id_column``, or a required column is
        missing.
    """
    target_columns = list(target_columns) if target_columns else list(TARGET_COLUMNS)
    group_keys = (
        [experiment_column, timepoint_column]
        if experiment_column
        else [timepoint_column]
    )
    required = {sample_id_column, *group_keys, *target_columns}
    missing = required - set(targets_df.columns)
    if missing:
        raise ValueError(f"targets_df is missing required columns: {sorted(missing)}")

    dmso = targets_df[targets_df[sample_id_column] == dmso_well]
    if dmso.empty:
        wells = sorted(targets_df[sample_id_column].unique())
        raise ValueError(
            f"DMSO well {dmso_well!r} not found in {sample_id_column}; "
            f"present wells: {wells}"
        )

    rows = []
    for key, group in dmso.groupby(group_keys, sort=True):
        key_values = key if isinstance(key, tuple) else (key,)
        record = dict(zip(group_keys, key_values))
        finite_counts, all_valid = [], True
        for column in target_columns:
            finite = group[column].to_numpy(float)
            finite = finite[np.isfinite(finite)]
            median = float(np.median(finite)) if finite.size else float("nan")
            mad = _scaled_mad(finite)
            record[f"median_{column}"] = median
            record[f"mad_{column}"] = mad
            finite_counts.append(finite.size)
            if not (finite.size >= min_cells and np.isfinite(median) and mad > 0):
                all_valid = False
        record["n_dmso"] = int(min(finite_counts)) if finite_counts else 0
        record["valid"] = bool(all_valid)
        rows.append(record)

    reference = pd.DataFrame(rows).sort_values(group_keys).reset_index(drop=True)
    n_invalid = int((~reference["valid"]).sum())
    if n_invalid:
        logger.warning(
            "DMSO reference (well %s): %d/%d groups invalid "
            "(finite n<%d, or non-finite median / zero MAD) and will yield NaN z.",
            dmso_well,
            n_invalid,
            len(reference),
            min_cells,
        )
    return reference


def normalize_targets_to_dmso(
    targets_df: pd.DataFrame,
    dmso_well: str,
    target_columns: Optional[List[str]] = None,
    sample_id_column: str = "sample_id",
    timepoint_column: str = "timepoint",
    experiment_column: Optional[str] = None,
    min_cells: int = 2,
    prefix: str = "z_",
) -> pd.DataFrame:
    """Add DMSO-normalized ``<prefix><col>`` columns (robust per-timepoint z-score).

    For every cell, ``z = (x - median_DMSO(t)) / MAD_DMSO(t)`` using the matched
    timepoint's DMSO reference (matched on ``experiment_column`` too when given). Cells
    whose group has an invalid reference (see :func:`compute_dmso_reference`) receive
    ``NaN``. Original target columns are left untouched and row order is preserved.

    Parameters
    ----------
    targets_df : pd.DataFrame
        Target table (see :func:`compute_dmso_reference`).
    dmso_well : str
        DMSO control well label.
    target_columns : list[str], optional
        Columns to normalize. Defaults to ``TARGET_COLUMNS``.
    sample_id_column, timepoint_column : str
        Well and timepoint column names.
    experiment_column : str, optional
        Scope the reference per (experiment, timepoint) when the table spans several
        experiments. If omitted, the whole table is assumed to be one experiment.
    min_cells : int
        Minimum finite DMSO values per timepoint for a valid reference.
    prefix : str
        Prefix for the new normalized columns (default ``"z_"``).

    Returns
    -------
    pd.DataFrame
        A copy of ``targets_df`` with one ``<prefix><col>`` column per target column.
    """
    target_columns = list(target_columns) if target_columns else list(TARGET_COLUMNS)
    group_keys = (
        [experiment_column, timepoint_column]
        if experiment_column
        else [timepoint_column]
    )
    reference = compute_dmso_reference(
        targets_df,
        dmso_well,
        target_columns=target_columns,
        sample_id_column=sample_id_column,
        timepoint_column=timepoint_column,
        experiment_column=experiment_column,
        min_cells=min_cells,
    )

    # Left-merge the valid reference onto the targets (preserves left row order);
    # invalid or unmatched groups leave NaN median/mad -> NaN z.
    valid_ref = reference[reference["valid"]]
    stat_columns = [f"median_{c}" for c in target_columns] + [
        f"mad_{c}" for c in target_columns
    ]
    merged = targets_df.merge(
        valid_ref[group_keys + stat_columns], on=group_keys, how="left"
    )

    result = targets_df.copy()
    for column in target_columns:
        z = (
            merged[column].to_numpy(float) - merged[f"median_{column}"].to_numpy(float)
        ) / (merged[f"mad_{column}"].to_numpy(float))
        result[f"{prefix}{column}"] = z

    n_dropped = int(pd.isna(result[f"{prefix}{target_columns[0]}"]).sum())
    if n_dropped:
        logger.warning(
            "DMSO normalization (well %s): %d/%d cells at invalid-reference groups "
            "received NaN z.",
            dmso_well,
            n_dropped,
            len(result),
        )
    return result
