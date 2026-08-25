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

from src.dataset_analysis.gate_survival import (
    DEFAULT_ABSOLUTE_FLOOR,
    per_timepoint_flags,
    ratcheted_flags,
)

from .contract import TARGET_COLUMNS

logger = logging.getLogger(__name__)

#: Relative confidence-gate threshold, as a fraction of each well's own peak cell count.
#: Picked from data in Step 2 (``gate_threshold_survival_summary.md``): 0.5 merely
#: reproduces ``t_cross_peak`` (it is the verdict definition, not a gate) and 0.05 fails
#: to flag Ew2-2's DMSO well even though it ends at 9.1% of peak. 0.10 lands within one
#: sample of independent hand-derived windows for the three undistorted cultures.
DEFAULT_MIN_PEAK_FRACTION = 0.10

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


#: Column added by :func:`compute_confidence_flags` naming which condition(s) fired.
CONFIDENCE_REASON_COLUMN = "low_confidence_reason"


def _numeric_order(values: pd.Series) -> np.ndarray:
    """Sort positions for ``values`` treated as numbers, not strings.

    The frame labels in this dataset are ``1, 11, 21, ... 351``, and
    :func:`contract.normalize_cell_key` casts key columns to ``str``, so a lexical sort
    puts ``11`` before ``2`` and the ratchet would trip at the wrong frame. Values that
    are not numeric sort last, in stable input order, so a malformed label degrades to
    "unordered" rather than silently reordering the trajectory.
    """
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any():
        logger.warning(
            "%d/%d timepoint labels are not numeric; they cannot be ordered in "
            "time and the ratcheted condition is unreliable for them.",
            int(numeric.isna().sum()),
            len(numeric),
        )
    return np.argsort(numeric.to_numpy(dtype=float), kind="stable")


def compute_confidence_flags(
    targets_df: pd.DataFrame,
    dmso_well: str,
    *,
    sample_id_column: str = "sample_id",
    timepoint_column: str = "timepoint",
    cell_id_column: str = "cell_id",
    experiment_column: Optional[str] = None,
    min_peak_fraction: Optional[float] = DEFAULT_MIN_PEAK_FRACTION,
    absolute_floor: Optional[int] = DEFAULT_ABSOLUTE_FLOOR,
) -> pd.DataFrame:
    """Per-(well, timepoint) confidence flags: has this well's population collapsed?

    Implements the three-condition gate designed in Step 2
    (``src/dataset_analysis/gate_survival.py``'s module docstring), reusing that
    module's ``ratcheted_flags`` / ``per_timepoint_flags`` primitives rather than
    reimplementing them:

    1. **relative, ratcheted** -- the well's own distinct-cell count falls below
       ``min_peak_fraction x peak``. Once crossed it stays flagged: a re-fragmenting
       spheroid does not restore trustworthy per-cell statistics.
    2. **absolute floor, per timepoint** -- fewer than ``absolute_floor`` cells *here*.
       Deliberately NOT ratcheted; it asks whether a percentile is computable at this
       timepoint, which a later timepoint may legitimately answer differently.
    3. **DMSO reference degraded, inherited** -- the experiment's DMSO well is itself
       flagged by (1) or (2) at this timepoint, so there is no trustworthy baseline to
       normalise against however healthy the target well looks. This is the condition
       that binds in practice: Step 2 measured it discarding a further 18.3% of all
       well-timepoints beyond each well's own condition.

    **Counts are DISTINCT ``cell_id``, not rows.** The target table's row unit is
    ``(cell, z_slice)`` (``contract.CELL_KEY``), with 3-13 slices per cell, so counting
    rows would inflate every count ~5x and reduce a 30-cell floor to roughly 5 cells.
    Distinct cells also matches how Step 2's threshold was derived
    (``cell_population.csv`` counts ``cell_id.nunique()``).

    Parameters
    ----------
    targets_df : pd.DataFrame
        Target table carrying ``sample_id_column``, ``timepoint_column`` and
        ``cell_id_column``.
    dmso_well : str
        Well label of the DMSO (vehicle) reference.
    sample_id_column, timepoint_column, cell_id_column : str
        Column names; defaults match the mcherry_metrics schema.
    experiment_column : str, optional
        Scope peaks and the DMSO reference per experiment when the table spans several
        (cultures reuse DMSO well labels). If omitted the table is assumed to be one
        experiment.
    min_peak_fraction : float, optional
        Relative threshold as a fraction of the well's own peak. ``None`` disables
        condition (1).
    absolute_floor : int, optional
        Minimum distinct cells at a timepoint. ``None`` disables condition (2).

    Returns
    -------
    pd.DataFrame
        One row per (experiment?, well, timepoint) with ``n_cells``, ``peak``,
        ``relative_threshold``, the three boolean conditions
        (``flag_relative``, ``flag_absolute_floor``, ``flag_dmso_reference``),
        ``low_confidence`` (their OR) and ``low_confidence_reason``.

    Raises
    ------
    ValueError
        If a required column is missing, ``dmso_well`` is absent, or
        ``min_peak_fraction`` is outside ``(0, 1]``.
    """
    if min_peak_fraction is not None and not 0 < min_peak_fraction <= 1:
        raise ValueError(
            f"min_peak_fraction must be in (0, 1]; got {min_peak_fraction!r}. It is a "
            f"fraction of the well's peak, not a cell count."
        )
    if absolute_floor is not None and absolute_floor < 1:
        # `< 1` would make the condition vacuous for every non-negative count, which is
        # indistinguishable from an active-but-useless floor. Disabling the condition
        # has its own spelling: pass None.
        raise ValueError(
            f"absolute_floor must be at least 1; got {absolute_floor!r}. Pass None to "
            f"disable the floor condition instead of a value no count can fall below."
        )

    keys = [sample_id_column, timepoint_column, cell_id_column]
    if experiment_column:
        keys.append(experiment_column)
    missing = [c for c in keys if c not in targets_df.columns]
    if missing:
        raise ValueError(
            f"targets_df is missing {missing!r}; the confidence gate counts distinct "
            f"cells per (well, timepoint) and cannot fall back to row counts -- "
            f"the row unit is (cell, z_slice), so that would overcount ~5x."
        )
    if dmso_well not in set(targets_df[sample_id_column]):
        wells = sorted(map(str, targets_df[sample_id_column].unique()))
        raise ValueError(
            f"DMSO well {dmso_well!r} not found in {sample_id_column}; a silently "
            f"absent reference would make condition (3) all-False and understate the "
            f"gate's cost. Present wells: {wells}"
        )

    well_keys = ([experiment_column] if experiment_column else []) + [sample_id_column]
    counts = (
        targets_df.groupby(well_keys + [timepoint_column], sort=False)[cell_id_column]
        .nunique()
        .reset_index(name="n_cells")
    )

    frames = []
    for _, group in counts.groupby(well_keys, sort=False):
        ordered = group.iloc[_numeric_order(group[timepoint_column])].copy()
        values = ordered["n_cells"].to_numpy(dtype=float)
        peak = float(values.max()) if values.size else float("nan")
        ordered["peak"] = peak

        if min_peak_fraction is not None and np.isfinite(peak) and peak > 0:
            threshold = min_peak_fraction * peak
            ordered["relative_threshold"] = threshold
            ordered["flag_relative"] = ratcheted_flags(values, threshold)
        else:
            # A zero/NaN peak cannot anchor a relative threshold: `fraction x 0` is
            # never exceeded, so the well would read "never flagged" rather than
            # "unknown". Mirrors gate_survival._usable_peak.
            ordered["relative_threshold"] = float("nan")
            ordered["flag_relative"] = False

        if absolute_floor is not None:
            ordered["flag_absolute_floor"] = per_timepoint_flags(values, absolute_floor)
        else:
            ordered["flag_absolute_floor"] = False
        frames.append(ordered)

    flags = pd.concat(frames, ignore_index=True)
    own = flags["flag_relative"] | flags["flag_absolute_floor"]

    # Condition 3: inherit the DMSO well's own flags at the matched timepoint (matched
    # on experiment too, when scoped). Timepoints the DMSO well never reached have no
    # reference at all, so they are flagged rather than defaulted to False.
    dmso_keys = ([experiment_column] if experiment_column else []) + [timepoint_column]
    dmso = flags[flags[sample_id_column] == dmso_well][dmso_keys].assign(
        _dmso_flagged=own[flags[sample_id_column] == dmso_well].to_numpy()
    )
    merged = flags[dmso_keys].merge(dmso, on=dmso_keys, how="left")
    # np.where rather than .fillna: the merged column is object dtype (bool + NaN) and
    # pandas deprecated the silent downcast that .fillna would perform here.
    raw = merged["_dmso_flagged"]
    flags["flag_dmso_reference"] = np.where(
        raw.isna().to_numpy(), True, raw.to_numpy()
    ).astype(bool)

    flags["low_confidence"] = own | flags["flag_dmso_reference"]

    reasons = {
        "flag_relative": (
            f"below {min_peak_fraction:g}x peak (ratcheted)"
            if min_peak_fraction is not None
            else ""
        ),
        "flag_absolute_floor": f"fewer than {absolute_floor} cells",
        "flag_dmso_reference": "DMSO reference degraded",
    }
    flags[CONFIDENCE_REASON_COLUMN] = [
        "; ".join(label for column, label in reasons.items() if row[column] and label)
        for _, row in flags.iterrows()
    ]
    return flags.reset_index(drop=True)


def apply_dmso_normalization(
    targets_df: pd.DataFrame,
    *,
    enabled: bool,
    dmso_well: Optional[str],
    target_columns: List[str],
    experiment_column: Optional[str] = None,
    sample_id_column: str = "sample_id",
    timepoint_column: str = "timepoint",
    cell_id_column: str = "cell_id",
    min_cells: int = 2,
    min_peak_fraction: Optional[float] = None,
    absolute_floor: Optional[int] = None,
    prefix: str = "z_",
) -> "tuple[pd.DataFrame, List[str]]":
    """Pipeline hook: optionally DMSO-normalize the targets, returning target columns.

    A thin wrapper for use between ``load_targets`` and ``build_matrix``. When
    ``enabled`` is False this is a no-op returning ``(targets_df, target_columns)``.
    When True it adds ``<prefix><col>`` columns, **drops cells whose z is undefined**
    (invalid-reference timepoints -- so downstream models never see NaN targets), and
    returns the ``<prefix>`` names as the effective target columns to model.

    Parameters
    ----------
    targets_df : pd.DataFrame
        Loaded target table (see :func:`normalize_targets_to_dmso`).
    enabled : bool
        Whether to normalize. If False, returns the inputs unchanged.
    dmso_well : str, optional
        DMSO control well label. Required when ``enabled`` is True.
    target_columns : list[str]
        Original target columns to normalize.
    experiment_column, sample_id_column, timepoint_column, min_cells, prefix
        Forwarded to :func:`normalize_targets_to_dmso`.
    cell_id_column : str
        Cell identifier, used by the confidence gate to count DISTINCT cells.
    min_peak_fraction : float, optional
        Confidence-gate relative threshold, a fraction of each well's own peak
        distinct-cell count (Step 2's recommendation: ``0.10``, available as
        :data:`DEFAULT_MIN_PEAK_FRACTION`). ``None`` -- the default -- leaves the gate
        OFF, preserving the historical behaviour of this function.
    absolute_floor : int, optional
        Confidence-gate degenerate-statistics floor in distinct cells (Step 2 keeps
        ``30``, :data:`DEFAULT_ABSOLUTE_FLOOR`). ``None`` disables that condition. The
        gate is active when *either* threshold is set.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        ``(targets_df, effective_target_columns)``. When enabled, the columns are the
        ``<prefix><col>`` names and the frame has undefined-z rows dropped -- plus
        low-confidence rows when the gate is active.

    Raises
    ------
    ValueError
        If ``enabled`` is True but ``dmso_well`` is not set, or the gate is active and
        discards every row.

    Notes
    -----
    **Two distinct filters run here; do not conflate them.** The long-standing one drops
    cells whose z is *undefined* (zero MAD or fewer than ``min_cells`` finite DMSO
    values) -- a degenerate-arithmetic guard that fires rarely. The confidence gate is
    the separate, opt-in filter that drops cells at (well, timepoint)s whose population
    has collapsed.

    **The gate is results-changing, not a safety rail.** Step 2 measured its cost at the
    recommended ``min_peak_fraction=0.10``: only **47% of all 1620 well-timepoints
    survive** the relative + DMSO-reference conditions, and the loss is severely
    culture-asymmetric -- median retention is 1.00 for HD1509 and HD1883 but 0.11, 0.17
    and 0.11 for Ew2-1, Ew2-2 and SA110. Enabling it therefore narrows z-target analyses
    towards two of the five cultures. Record before/after counts per (experiment, well)
    when turning it on; a passing test suite is not evidence.
    """
    if not enabled:
        return targets_df, list(target_columns)
    if not dmso_well:
        raise ValueError("normalize_to_dmso is enabled but dmso_well is not set")
    if experiment_column is None:
        # No experiment column to scope by: the reference is pooled per timepoint over
        # every row with sample_id == dmso_well. Correct only for a single-experiment
        # table; a table concatenating cultures (which reuse DMSO well labels) would
        # pool their baselines. See TODO in plan_dmso_normalization_implementation.md.
        logger.warning(
            "DMSO normalization (well %s) assumes a SINGLE-experiment target table; "
            "pass experiment_column to scope per (experiment, timepoint) if the input "
            "concatenates cultures.",
            dmso_well,
        )

    normalized = normalize_targets_to_dmso(
        targets_df,
        dmso_well,
        target_columns=target_columns,
        sample_id_column=sample_id_column,
        timepoint_column=timepoint_column,
        experiment_column=experiment_column,
        min_cells=min_cells,
        prefix=prefix,
    )
    z_columns = [f"{prefix}{column}" for column in target_columns]
    before = len(normalized)
    normalized = normalized.dropna(subset=z_columns).reset_index(drop=True)
    dropped = before - len(normalized)
    if dropped:
        logger.info(
            "DMSO normalization: dropped %d/%d cells with undefined z "
            "(invalid-reference timepoints) before modeling.",
            dropped,
            before,
        )

    gate_active = min_peak_fraction is not None or absolute_floor is not None
    if not gate_active:
        return normalized, z_columns

    # Flags come from `targets_df`, the UNFILTERED input -- not from `normalized`,
    # which has already had its undefined-z rows dropped above. Each well's relative
    # threshold is a fraction of its own peak, so computing the peak from the filtered
    # frame would understate it whenever an invalid-reference timepoint happened to be
    # that well's peak, lowering the threshold and letting genuinely collapsed
    # timepoints survive. It also keeps this identical to the peaks
    # `scripts/report_dmso_gate_impact.py` reports, which reads the raw target table.
    # `normalize_targets_to_dmso` only adds columns, so the two frames' rows correspond.
    flags = compute_confidence_flags(
        targets_df,
        dmso_well,
        sample_id_column=sample_id_column,
        timepoint_column=timepoint_column,
        cell_id_column=cell_id_column,
        experiment_column=experiment_column,
        min_peak_fraction=min_peak_fraction,
        absolute_floor=absolute_floor,
    )
    join_keys = ([experiment_column] if experiment_column else []) + [
        sample_id_column,
        timepoint_column,
    ]
    gate_columns = join_keys + ["low_confidence", CONFIDENCE_REASON_COLUMN]
    annotated = normalized.merge(flags[gate_columns], on=join_keys, how="left")
    if len(annotated) != len(normalized):
        raise ValueError(
            f"confidence-gate join changed the row count "
            f"({len(normalized)} -> {len(annotated)}); the flag frame must have "
            f"exactly one row per {join_keys!r}."
        )
    # An unmatched row has no flag verdict at all, which must not read as "confident".
    keep = ~annotated["low_confidence"].fillna(True).to_numpy(dtype=bool)

    gated = normalized.loc[keep].reset_index(drop=True)
    n_gated = len(normalized) - len(gated)
    if n_gated:
        reason_counts = (
            annotated.loc[~keep, CONFIDENCE_REASON_COLUMN].value_counts().to_dict()
        )
        logger.warning(
            "Confidence gate (min_peak_fraction=%s, absolute_floor=%s): dropped "
            "%d/%d cells (%.1f%%) at low-confidence (well, timepoint)s. This is "
            "results-changing, not a safety rail. Reasons: %s",
            min_peak_fraction,
            absolute_floor,
            n_gated,
            len(normalized),
            100.0 * n_gated / len(normalized) if len(normalized) else 0.0,
            reason_counts,
        )
    if gated.empty:
        raise ValueError(
            f"the confidence gate discarded every row "
            f"(min_peak_fraction={min_peak_fraction!r}, "
            f"absolute_floor={absolute_floor!r}). This means the DMSO reference well "
            f"{dmso_well!r} is flagged at every timepoint, which flags every other "
            f"well through the inherited condition. Note the relative condition alone "
            f"cannot cause this -- a well's peak timepoint is never below a fraction "
            f"of its own peak -- so the reference well's count is below absolute_floor "
            f"across its whole trajectory. Lower absolute_floor or exclude this "
            f"experiment rather than modelling an empty table."
        )
    return gated, z_columns
