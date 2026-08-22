"""Assembly analytics for the dataset-design assessment.

Three questions, each answered from artefacts the earlier steps already produced:

- **(a)** what pre-collapse keep-window does each culture have?
- **(b)** does seeding density predict time-to-collapse across the 45 wells?
- **(c')** is a drug response visible — inside the trustworthy window, and over the
  full timecourse?

**(c') deliberately reports two windows.** Restricting to the pre-confluence window
alone would miss the largest drug effects in this dataset: the T4/T5 scratch work found
the mCherry divergence is *late* and grows over time, and warned explicitly against
blanket-truncating the timecourse. Per-cell *morphology* is only valid pre-collapse, but
the *response* readout is not bound by that, so both are reported and labelled.

**The pre-confluence window is bounded on both sides.** It ends at the earlier of the
DMSO reference's ``t_cross`` and the drug well's own, because feature validity is a
property of each well: drug wells commonly collapse *before* their reference (15 of 40
here), so cutting only at the reference would admit post-collapse spheroid-fragment rows
on the drug side of a comparison labelled pre-confluence.

**One observation per physical cell, not per z-slice.** ``instance_metrics.csv``
holds one row per (cell, z_slice) -- ~5.4 rows per cell, ranging 3..13. Comparing raw
rows would inflate ``n`` *and* weight each cell by how many slices it spans; because
treatment changes morphology, that weighting correlates with the effect being measured,
biasing the estimate rather than merely overstating its precision. Rows are reduced to
one value per ``(well, timepoint, cell_id)`` -- the median over that cell's slices --
before any comparison, and reported counts are distinct cells.

**Effect sizes here are descriptive for one imaged well.** The design has exactly one
well per (drug, concentration) — no within-condition replicates — so a cell-level rank
effect describes that well, and cannot support a condition-level "no effect" claim. The
labels and the mandatory ``n_wells`` column encode that limit rather than leaving it to
the reader.

See ``docs/feature_to_mcherry/plan_dataset_design_assessment.md`` §Step 7.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .data.collapse import collapse_slices_to_cells as _collapse_to_cells

logger = logging.getLogger(__name__)

#: Cliff's delta magnitude cut-points (Romano et al.), fixed up front rather than
#: eyeballed. Only these two are used: :func:`classify_shift` reports three bands, so
#: everything below ``SMALL_MAX`` is one band. Romano's 0.147 negligible/small cut-point
#: is deliberately NOT defined here -- it was previously present but unused, which
#: invited write-ups to describe a four-band scheme the code does not implement.
SMALL_MAX = 0.33
MEDIUM_MAX = 0.474

LABEL_LARGE = "large shift observed"
LABEL_MODERATE = "moderate"
LABEL_NO_LARGE = "no large shift observed in this imaged well"
LABEL_NOT_ESTIMABLE = "not estimable (window too narrow)"

#: Minimum *distinct cells* (not slice rows) per side for a per-timepoint delta.
MIN_CELLS_PER_SIDE = 10

#: Minimum matched timepoints before a window's aggregate delta is reported. This is a
#: floor on *this well's* usable comparisons; the *culture's* window width is judged by
#: :func:`~src.feature_to_mcherry.pre_collapse.classify_estimability`, the single
#: source of truth for that verdict. Both must pass. Holding a second, looser threshold
#: here previously let Ew2-2 and SA110 (3-timepoint windows) receive confident labels in
#: Step 7 while Step 5' called the same cultures not estimable.
MIN_MATCHED_TIMEPOINTS = 3

WINDOW_PRE = "pre_confluence"
WINDOW_FULL = "full_timecourse"


def cliffs_delta(x: Sequence[float], y: Sequence[float]) -> float:
    """Cliff's delta for ``x`` vs ``y``: P(x>y) - P(x<y), in [-1, 1].

    Computed from ranks rather than a pairwise sweep (O(n log n) not O(n*m)), and from
    :func:`scipy.stats.rankdata` rather than a ``mannwhitneyu`` result object — the
    latter's attribute names have shifted across the scipy versions this project
    supports, and average-ranking credits ties at the half weight Cliff's delta wants.

    Positive means ``x`` tends to exceed ``y``. Returns NaN if either side is empty.
    """
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    n_a, n_b = a.size, b.size
    if n_a == 0 or n_b == 0:
        return float("nan")
    combined = np.concatenate([a, b])
    ranks = rankdata(combined)
    rank_sum_a = float(ranks[:n_a].sum())
    # Mann-Whitney U for `a`, then map U -> delta on [-1, 1].
    u_a = rank_sum_a - n_a * (n_a + 1) / 2.0
    return float(2.0 * u_a / (n_a * n_b) - 1.0)


def classify_shift(delta: Optional[float], *, estimable: bool = True) -> str:
    """Band a delta, or mark it not estimable.

    ``LABEL_NO_LARGE`` deliberately says "no large shift observed in this imaged well"
    rather than "not distinguishable": with a single well, a small delta is not evidence
    of no drug effect, and the shorter phrasing invites exactly that misreading.
    """
    if not estimable or delta is None or pd.isna(delta):
        return LABEL_NOT_ESTIMABLE
    magnitude = abs(float(delta))
    if magnitude >= MEDIUM_MAX:
        return LABEL_LARGE
    if magnitude >= SMALL_MAX:
        return LABEL_MODERATE
    return LABEL_NO_LARGE


def _direction(delta: Optional[float]) -> Optional[str]:
    """Signed direction, or ``None`` for missing *and for exactly zero*.

    Zero means the two distributions are interchangeable, which is neither elevation nor
    suppression; letting it fall through to an ``else`` branch would stamp a drug
    direction onto a no-change row.
    """
    if delta is None or pd.isna(delta) or float(delta) == 0.0:
        return None
    return "elevates" if float(delta) > 0 else "suppresses"


def _earlier_crossing(
    dmso_cross: Optional[float], well_cross: Optional[float]
) -> Optional[float]:
    """The earlier of the two collapse times, treating NaN as "never collapsed".

    A well that never falls below the threshold imposes no bound of its own, so the
    other side governs; if neither collapses there is nothing to truncate and the result
    is ``None`` (no truncation) rather than NaN.
    """
    bounds = [
        float(value)
        for value in (dmso_cross, well_cross)
        if value is not None and not pd.isna(value)
    ]
    return min(bounds) if bounds else None


def collapse_slices_to_cells(
    targets: pd.DataFrame,
    target_column: str,
    *,
    well_column: str = "sample_id",
    time_column: str = "timepoint",
    cell_id_column: str = "cell_id",
) -> pd.DataFrame:
    """One row per ``(well, timepoint, cell_id)``, the target median over its z-slices.

    A single-column wrapper over the shared
    :func:`src.feature_to_mcherry.data.collapse_slices_to_cells`. Both this module and
    Step 5' delegate to that one definition, so the two analyses cannot drift apart on
    what one observation is.

    Raises:
        ValueError: If any key column or ``target_column`` is absent. Falling back to
            raw rows would silently reinstate slice-weighted comparisons -- the bug
            this guards against.
    """
    return _collapse_to_cells(
        targets,
        [target_column],
        well_column=well_column,
        time_column=time_column,
        cell_id_column=cell_id_column,
    )


def timepoint_matched_delta(
    targets: pd.DataFrame,
    drug_well: str,
    dmso_well: str,
    target_column: str,
    *,
    max_ti: Optional[float] = None,
    well_column: str = "sample_id",
    time_column: str = "timepoint",
    cell_id_column: str = "cell_id",
    min_cells_per_side: int = MIN_CELLS_PER_SIDE,
    targets_are_per_cell: bool = False,
) -> Dict[str, Any]:
    """Median per-timepoint Cliff's delta of ``drug_well`` against ``dmso_well``.

    Matching *within* timepoint before aggregating is what keeps the comparison honest:
    a pooled delta would confound the drug effect with how many cells each well
    contributes at each time, and both wells thin out over time at different rates.

    Args:
        max_ti: Keep only timepoints ``<= max_ti`` (the pre-confluence window). ``None``
            uses the full timecourse.
        targets_are_per_cell: Set when ``targets`` has already been through
            :func:`collapse_slices_to_cells`. Callers looping over many wells should
            collapse once and pass ``True`` rather than re-collapsing the same table per
            well -- the reduction is the expensive step on a 1.4M-row experiment.

    Returns:
        ``delta`` (median across matched timepoints), ``n_timepoints_matched``,
        ``n_cells_drug``, ``n_cells_dmso``, and ``per_timepoint`` (the raw deltas).
    """
    n_slice_rows = len(targets)
    if targets_are_per_cell:
        per_cell = targets
    else:
        per_cell = collapse_slices_to_cells(
            targets,
            target_column,
            well_column=well_column,
            time_column=time_column,
            cell_id_column=cell_id_column,
        )
    times = pd.to_numeric(per_cell[time_column], errors="coerce")
    frame = per_cell.assign(_ti=times)
    if max_ti is not None:
        # A NaN bound would make every ``<=`` comparison False and silently return an
        # empty window, which surfaces as a spurious "not estimable" rather than an
        # error. Callers must pass None to mean "no truncation".
        if pd.isna(max_ti):
            raise ValueError(
                "max_ti is NaN; pass None to mean 'no truncation' rather than a "
                "missing bound, which would silently empty the window"
            )
        frame = frame[frame["_ti"] <= float(max_ti)]

    drug = frame[frame[well_column].astype(str) == str(drug_well)]
    dmso = frame[frame[well_column].astype(str) == str(dmso_well)]

    deltas = []
    for ti, drug_rows in drug.groupby("_ti", sort=True):
        dmso_rows = dmso[dmso["_ti"] == ti]
        # Count FINITE targets, not rows. cliffs_delta drops non-finite values itself,
        # so a row-count floor would admit a timepoint where (say) 10 of 12 drug cells
        # carry NaN, reporting a confident delta computed from the surviving 2. The CLI
        # path happens to be safe because load_targets drops NaN target rows, but this
        # function is exported and must not depend on that.
        drug_values = drug_rows[target_column].to_numpy(dtype=float)
        dmso_values = dmso_rows[target_column].to_numpy(dtype=float)
        if (
            np.isfinite(drug_values).sum() < min_cells_per_side
            or np.isfinite(dmso_values).sum() < min_cells_per_side
        ):
            continue
        deltas.append(cliffs_delta(drug_values, dmso_values))

    finite = [d for d in deltas if np.isfinite(d)]
    return {
        "delta": float(np.median(finite)) if finite else float("nan"),
        "n_timepoints_matched": len(finite),
        "n_cells_drug": int(len(drug)),
        "n_cells_dmso": int(len(dmso)),
        "n_slice_rows_in": n_slice_rows,
        "per_timepoint": finite,
    }


def build_effect_table(
    targets: pd.DataFrame,
    summary: pd.DataFrame,
    experiment: str,
    dmso_well: str,
    target_column: str,
    *,
    min_matched_timepoints: int = MIN_MATCHED_TIMEPOINTS,
) -> pd.DataFrame:
    """Both windows x every drug well, one row each.

    The ``pre_confluence`` row is cut at the **earlier** of the DMSO reference's
    ``t_cross_peak`` and the drug well's own, since per-cell features must be valid on
    both sides of the comparison; ``window_max_ti`` records the bound applied. The
    ``full_timecourse`` row is never truncated.

    Raises:
        ValueError: If ``experiment`` is absent from ``summary``, or its DMSO row is
            missing — that row is one of the two bounds on the pre-confluence window, so
            a silent skip would report a wider window mislabelled as pre-confluence.
    """
    # Validate the summary BEFORE collapsing: the checks are O(1) on a 45-row table
    # while the reduction is the expensive step on a 1.4M-row experiment, so a bad
    # experiment label should fail without paying for it.
    wells = summary[summary["experiment"] == experiment]
    if wells.empty:
        raise ValueError(f"{experiment}: no rows in the Step 1 summary")
    dmso_rows = wells[wells["well"].astype(str) == str(dmso_well)]
    if dmso_rows.empty:
        raise ValueError(
            f"{experiment}: DMSO well {dmso_well!r} absent from the summary; its "
            f"t_cross_peak is one of the two bounds on the pre-confluence window"
        )
    dmso_cross = dmso_rows.iloc[0]["t_cross_peak"]

    # Collapse slices to cells ONCE for this experiment/target: the loop below runs
    # 8 wells x 2 windows, and the caller loops 3 targets, so collapsing per call
    # repeated the same reduction 48x per experiment (240x overall).
    n_slice_rows = len(targets)
    per_cell = collapse_slices_to_cells(targets, target_column)

    # The culture-level window verdict comes from pre_collapse -- the same function Step
    # 5' uses -- so the two sections of the deliverable cannot disagree about whether a
    # culture's pre-confluence window is wide enough to interpret.
    from src.feature_to_mcherry.pre_collapse import (
        VERDICT_NOT_ESTIMABLE,
        classify_estimability,
    )

    window_verdict = classify_estimability(dmso_rows.iloc[0]["n_timepoints_pre_cross"])
    pre_window_estimable = window_verdict != VERDICT_NOT_ESTIMABLE

    rows = []
    for _, well_row in wells.iterrows():
        well = str(well_row["well"])
        if well == str(dmso_well):
            continue
        # The pre-confluence window has to hold for BOTH sides of the comparison: it is
        # the span over which per-cell features describe real single cells. Cutting only
        # at the DMSO reference's t_cross is not enough, because drug wells usually
        # collapse EARLIER than their reference -- in this dataset 15 of 40 do -- so a
        # reference-only cut silently admits post-collapse spheroid-fragment rows on the
        # drug side. Take whichever side collapses first.
        pre_cut = _earlier_crossing(dmso_cross, well_row["t_cross_peak"])
        for window, max_ti in ((WINDOW_PRE, pre_cut), (WINDOW_FULL, None)):
            result = timepoint_matched_delta(
                per_cell,
                well,
                dmso_well,
                target_column,
                max_ti=max_ti,
                targets_are_per_cell=True,
            )
            enough_matched = result["n_timepoints_matched"] >= min_matched_timepoints
            # A narrow culture window disqualifies the pre-confluence row regardless of
            # how many timepoints happened to match inside it.
            estimable = enough_matched and (
                pre_window_estimable or window != WINDOW_PRE
            )
            # Report no aggregate delta when the window is not interpretable: a number
            # printed beside "not estimable" invites being read as the effect anyway.
            delta = result["delta"] if estimable else float("nan")
            rows.append(
                {
                    "experiment": experiment,
                    "well": well,
                    "drug": well_row["drug"],
                    "dose_rank": well_row["dose_rank"],
                    "concentration_uM": well_row["concentration_uM"],
                    "target": target_column,
                    "window": window,
                    "window_max_ti": max_ti,
                    "n_wells": 1,  # no within-condition replicates by design (B1)
                    "cliffs_delta": delta,
                    "direction": _direction(delta),
                    "window_verdict": window_verdict,
                    "n_timepoints_matched": result["n_timepoints_matched"],
                    "n_cells_drug": result["n_cells_drug"],
                    "n_cells_dmso": result["n_cells_dmso"],
                    "n_slice_rows_in": n_slice_rows,
                    "label": classify_shift(delta, estimable=estimable),
                }
            )
    return pd.DataFrame(rows)


def build_cutoff_table(
    summary: pd.DataFrame,
    *,
    gate_threshold: Optional[float] = 0.10,
) -> pd.DataFrame:
    """Per-culture pre-collapse keep-window, from each culture's own DMSO reference.

    ``gate_threshold`` is Step 2's recommended relative gate. It is carried only to note
    where the *verdict* cutoff (50 % of peak, which defines ``t_cross_peak``) and the
    *gate* cutoff diverge; pass ``None`` if Step 2 was cut.
    """
    rows = []
    for experiment, group in summary.groupby("experiment", sort=True):
        dmso = group[group["is_dmso"].astype(bool)]
        if dmso.empty:
            raise ValueError(f"{experiment}: no DMSO well in the summary")
        row = dmso.iloc[0]
        rows.append(
            {
                "experiment": experiment,
                "dmso_well": row["well"],
                "denominator": "peak",
                "t_cross_peak": row["t_cross_peak"],
                "n_timepoints_pre_cross": row["n_timepoints_pre_cross"],
                "end_pct_of_peak": row["end_pct_of_peak"],
                "collapse_shape": row["collapse_shape"],
                "verdict_cutoff_fraction": 0.5,
                "gate_cutoff_fraction": gate_threshold,
            }
        )
    return pd.DataFrame(rows)


def density_vs_collapse(summary: pd.DataFrame) -> pd.DataFrame:
    """Spearman of peak density against time-to-collapse, pooled and per culture.

    Wells that never collapse are **right-censored**, not missing: their true
    ``t_cross`` exceeds the timecourse. Spearman cannot represent that, so they are
    excluded from the correlation and counted in ``n_censored_excluded`` — reporting the
    correlation without that count would overstate what the 45 wells support.
    """
    from src.feature_to_mcherry.informativeness.univariate import _safe_spearmanr

    def _one(frame: pd.DataFrame, label: str) -> Dict[str, Any]:
        crossed = frame[frame["t_cross_peak"].notna()]
        rho, pvalue = _safe_spearmanr(
            crossed["peak_n_cells"].to_numpy(dtype=float),
            crossed["t_cross_peak"].to_numpy(dtype=float),
        )
        return {
            "scope": label,
            "n_wells": len(frame),
            "n_used": len(crossed),
            "n_censored_excluded": len(frame) - len(crossed),
            "rho": rho,
            "pvalue": pvalue,
        }

    rows = [_one(summary, "pooled (all cultures)")]
    for experiment, group in summary.groupby("experiment", sort=True):
        rows.append(_one(group, experiment))
    return pd.DataFrame(rows)
