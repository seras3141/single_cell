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
    """Scaled MAD (``x1.4826``); NaN-safe, returns NaN if no finite values."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
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
    min_cells: int = 2,
) -> pd.DataFrame:
    """Per-timepoint DMSO reference statistics (median + scaled MAD) for each target.

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
    min_cells : int
        Minimum DMSO cell count at a timepoint for its reference to be ``valid``.

    Returns
    -------
    pd.DataFrame
        One row per DMSO timepoint, columns: ``timepoint_column``, ``median_<col>``,
        ``mad_<col>`` (one pair per target), ``n_dmso`` and ``valid``
        (``n_dmso >= min_cells`` AND every ``mad_<col> > 0``). Sorted by timepoint.

    Raises
    ------
    ValueError
        If ``dmso_well`` is not present in ``sample_id_column``, or a required column is
        missing.
    """
    target_columns = list(target_columns) if target_columns else list(TARGET_COLUMNS)
    required = {sample_id_column, timepoint_column, *target_columns}
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
    for timepoint, group in dmso.groupby(timepoint_column, sort=True):
        record = {timepoint_column: timepoint, "n_dmso": int(len(group))}
        mads = []
        for column in target_columns:
            record[f"median_{column}"] = float(np.median(group[column].to_numpy(float)))
            mad = _scaled_mad(group[column].to_numpy())
            record[f"mad_{column}"] = mad
            mads.append(mad)
        record["valid"] = bool(
            record["n_dmso"] >= min_cells
            and all(np.isfinite(m) and m > 0 for m in mads)
        )
        rows.append(record)

    reference = pd.DataFrame(rows).sort_values(timepoint_column).reset_index(drop=True)
    n_invalid = int((~reference["valid"]).sum())
    if n_invalid:
        logger.warning(
            "DMSO reference (well %s): %d/%d timepoints invalid "
            "(n<%d or zero MAD) and will yield NaN z.",
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
    min_cells: int = 2,
    prefix: str = "z_",
) -> pd.DataFrame:
    """Add DMSO-normalized ``<prefix><col>`` columns (robust per-timepoint z-score).

    For every cell, ``z = (x - median_DMSO(t)) / MAD_DMSO(t)`` using the matched
    timepoint's DMSO reference. Cells whose timepoint has an invalid reference (see
    :func:`compute_dmso_reference`) receive ``NaN``. Original target columns are left
    untouched and row order is preserved.

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
    min_cells : int
        Minimum DMSO cells per timepoint for a valid reference.
    prefix : str
        Prefix for the new normalized columns (default ``"z_"``).

    Returns
    -------
    pd.DataFrame
        A copy of ``targets_df`` with one ``<prefix><col>`` column per target column.
    """
    target_columns = list(target_columns) if target_columns else list(TARGET_COLUMNS)
    reference = compute_dmso_reference(
        targets_df,
        dmso_well,
        target_columns=target_columns,
        sample_id_column=sample_id_column,
        timepoint_column=timepoint_column,
        min_cells=min_cells,
    )

    # Map timepoint -> per-column (median, mad); only valid timepoints normalize.
    valid_ref = reference[reference["valid"]].set_index(timepoint_column)
    result = targets_df.copy()
    timepoints = result[timepoint_column]
    for column in target_columns:
        medians = timepoints.map(valid_ref[f"median_{column}"])
        mads = timepoints.map(valid_ref[f"mad_{column}"])
        result[f"{prefix}{column}"] = (result[column] - medians) / mads

    n_dropped = int(result[f"{prefix}{target_columns[0]}"].isna().sum())
    if n_dropped:
        logger.warning(
            "DMSO normalization (well %s): %d/%d cells at invalid-reference timepoints "
            "received NaN z.",
            dmso_well,
            n_dropped,
            len(result),
        )
    return result
