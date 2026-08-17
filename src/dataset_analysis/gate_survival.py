"""How much data each candidate confidence-gate threshold would discard.

A confidence gate flags well-timepoints whose per-cell statistics are no longer
trustworthy because the well's population has collapsed. This module quantifies, for a
set of candidate thresholds, how much of each well's trajectory survives — so the
default threshold is picked from the data rather than guessed.

Three flag conditions, matching the gate design in
``docs_local/feature_to_mcherry/plan_dmso_normalization_implementation.md``:

1. **relative, ratcheted** — the well's own count falls below ``fraction x peak``. Once
   it crosses, every later timepoint stays flagged regardless of an uptick, since a
   re-fragmenting spheroid does not restore trustworthy per-cell statistics.
2. **absolute floor** — fewer than ``absolute_floor`` cells at that timepoint. This is
   a degenerate-statistics safety net, so it is evaluated **per timepoint, not
   ratcheted**: it asks "are there enough cells to compute a percentile here", which a
   later timepoint can legitimately answer differently.
3. **DMSO-reference degraded** — the experiment's DMSO reference well is itself flagged
   at that timepoint, so there is no trustworthy baseline to normalise against, however
   healthy the target well looks.

Thresholds are expressed against **peak**, not the mean of the first three samples: with
~2 h sampling several wells fall severalfold across their first three timepoints, so a
first-3 "baseline" already sits partway down the collapse (see the plan's §B3).

See ``docs/feature_to_mcherry/plan_fatima_deliverable_pipeline.md`` §Step 2.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Candidate relative thresholds, as a fraction of a well's peak count.
DEFAULT_FRACTIONS: Sequence[float] = (0.5, 0.25, 0.1, 0.05)

#: Existing absolute floor, kept for comparison.
DEFAULT_ABSOLUTE_FLOOR = 30

#: Label used for the absolute-floor condition in threshold columns.
ABSOLUTE_FLOOR_LABEL = "absolute_floor"


def ratcheted_flags(counts: Sequence[float], threshold: float) -> np.ndarray:
    """Flag the first timepoint below ``threshold`` and every timepoint after it.

    Args:
        counts: Per-timepoint counts, already ordered by time.
        threshold: Absolute count below which a timepoint is flagged.

    Returns:
        Boolean array, one entry per timepoint. All ``False`` if the well never crosses.
    """
    values = np.asarray(counts, dtype=float)
    below = values < threshold
    if not below.any():
        return np.zeros(values.shape, dtype=bool)
    first = int(np.argmax(below))
    flags = np.zeros(values.shape, dtype=bool)
    flags[first:] = True
    return flags


def per_timepoint_flags(counts: Sequence[float], threshold: float) -> np.ndarray:
    """Flag each timepoint below ``threshold`` independently (no ratchet)."""
    return np.asarray(counts, dtype=float) < threshold


def _threshold_label(fraction: float) -> str:
    """Stable column label for a relative threshold, e.g. ``0.25`` -> ``rel_0.25``."""
    return f"rel_{fraction:g}"


def build_flag_frame(
    cell_population: pd.DataFrame,
    peaks: Mapping[str, float],
    *,
    dmso_well: str,
    fractions: Iterable[float] = DEFAULT_FRACTIONS,
    absolute_floor: int = DEFAULT_ABSOLUTE_FLOOR,
    well_column: str = "sample_id",
    count_column: str = "n_cells",
    time_column: str = "ti",
) -> pd.DataFrame:
    """Long-form per-(well, timepoint) flags for every candidate threshold.

    Args:
        cell_population: One experiment's ``cell_population.csv``.
        peaks: ``{well: peak_n_cells}``, from the Step 1 summary.
        dmso_well: Well id of the experiment's DMSO reference.
        fractions: Candidate relative thresholds.
        absolute_floor: Cell count below which statistics are treated as degenerate.
        well_column, count_column, time_column: Column names in ``cell_population``.

    Returns:
        One row per (well, timepoint, threshold) with ``own_flagged`` (the well's own
        relative/floor condition) and ``dmso_flagged`` (its experiment's DMSO reference
        under the same threshold). ``any_flagged`` is the OR of the two.

    Raises:
        KeyError: If ``dmso_well`` is absent from the table — a silent all-False DMSO
            column would understate the gate's cost.
    """
    wells = list(cell_population[well_column].unique())
    if dmso_well not in wells:
        raise KeyError(
            f"DMSO well {dmso_well!r} not present in cell_population "
            f"(wells: {sorted(wells)})"
        )

    labels: Dict[str, Any] = {_threshold_label(f): f for f in fractions}
    per_well: Dict[str, pd.DataFrame] = {}

    for well, group in cell_population.groupby(well_column, sort=True):
        ordered = group.sort_values(time_column)
        counts = ordered[count_column].to_numpy(dtype=float)
        times = ordered[time_column].to_numpy(dtype=int)
        peak = float(peaks.get(str(well), np.nan))

        frame = pd.DataFrame({"ti": times})
        for label, fraction in labels.items():
            if np.isnan(peak):
                frame[label] = False
            else:
                frame[label] = ratcheted_flags(counts, fraction * peak)
        frame[ABSOLUTE_FLOOR_LABEL] = per_timepoint_flags(counts, absolute_floor)
        per_well[str(well)] = frame

    threshold_labels = list(labels) + [ABSOLUTE_FLOOR_LABEL]
    dmso_frame = per_well[dmso_well].set_index("ti")

    records = []
    for well, frame in per_well.items():
        for label in threshold_labels:
            own = frame[label].to_numpy(dtype=bool)
            dmso = (
                dmso_frame[label]
                .reindex(frame["ti"])
                .fillna(False)
                .to_numpy(dtype=bool)
            )
            records.append(
                pd.DataFrame(
                    {
                        "well": well,
                        "ti": frame["ti"].to_numpy(),
                        "threshold": label,
                        "own_flagged": own,
                        "dmso_flagged": dmso,
                        "any_flagged": own | dmso,
                    }
                )
            )

    return pd.concat(records, ignore_index=True)


def summarise_survival(flags: pd.DataFrame) -> pd.DataFrame:
    """Per (well, threshold): how many timepoints survive each condition.

    Columns: ``n_timepoints``; ``n_flagged_own``/``frac_kept_own`` (the well's own
    condition alone); ``n_flagged_any``/``frac_kept_any`` (including the DMSO-reference
    condition); ``n_dmso_only`` (flagged *purely* because the DMSO reference degraded —
    the marginal cost of condition 3); and ``first_flagged_ti`` for the own condition.
    """
    rows = []
    for (well, threshold), group in flags.groupby(["well", "threshold"], sort=True):
        own = group["own_flagged"].to_numpy(dtype=bool)
        dmso = group["dmso_flagged"].to_numpy(dtype=bool)
        n = len(group)
        flagged_ti = group.loc[group["own_flagged"], "ti"]
        rows.append(
            {
                "well": well,
                "threshold": threshold,
                "n_timepoints": n,
                "n_flagged_own": int(own.sum()),
                "frac_kept_own": float(1.0 - own.sum() / n) if n else np.nan,
                "n_flagged_any": int((own | dmso).sum()),
                "frac_kept_any": float(1.0 - (own | dmso).sum() / n) if n else np.nan,
                "n_dmso_only": int((~own & dmso).sum()),
                "first_flagged_ti": (
                    int(flagged_ti.iloc[0]) if not flagged_ti.empty else None
                ),
            }
        )
    return pd.DataFrame(rows)


def aggregate_by_threshold(survival: pd.DataFrame) -> pd.DataFrame:
    """Across all wells, the spread of "fraction of timepoints kept" per threshold.

    Reports the own-condition and the with-DMSO figures side by side; the gap between
    them is what condition 3 costs.
    """
    rows = []
    for threshold, group in survival.groupby("threshold", sort=True):
        rows.append(
            {
                "threshold": threshold,
                "n_wells": len(group),
                "kept_own_min": group["frac_kept_own"].min(),
                "kept_own_median": group["frac_kept_own"].median(),
                "kept_own_max": group["frac_kept_own"].max(),
                "kept_any_min": group["frac_kept_any"].min(),
                "kept_any_median": group["frac_kept_any"].median(),
                "kept_any_max": group["frac_kept_any"].max(),
                "n_wells_fully_flagged": int((group["frac_kept_any"] <= 0).sum()),
                "total_timepoints": int(group["n_timepoints"].sum()),
                "total_flagged_any": int(group["n_flagged_any"].sum()),
                "total_dmso_only": int(group["n_dmso_only"].sum()),
            }
        )
    out = pd.DataFrame(rows)
    out["pct_lost_to_dmso_condition"] = (
        100.0 * out["total_dmso_only"] / out["total_timepoints"]
    )
    return out


def crosscheck_against_t_cross(
    survival: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    fraction: float = 0.5,
) -> pd.DataFrame:
    """Confirm the ``fraction`` gate reproduces Step 1's ``t_cross_peak`` per well.

    At 0.5 the two are the same computation by construction (first timepoint below half
    the peak), so any mismatch means the survival table and the Step 1 summary have
    drifted apart. Returns one row per well with both values and an ``agrees`` flag.
    """
    label = _threshold_label(fraction)
    gate = survival[survival["threshold"] == label].set_index("well")
    merged = summary.set_index("well")[["t_cross_peak", "collapse_shape"]].join(
        gate[["first_flagged_ti"]], how="left"
    )
    merged = merged.reset_index()
    merged["agrees"] = merged.apply(
        lambda r: (
            (pd.isna(r["t_cross_peak"]) and pd.isna(r["first_flagged_ti"]))
            or r["t_cross_peak"] == r["first_flagged_ti"]
        ),
        axis=1,
    )
    return merged


def keep_window(
    survival: pd.DataFrame, flags: pd.DataFrame, well: str, threshold: str
) -> Optional[int]:
    """Last timepoint kept for ``well`` under ``threshold`` (``None`` if none kept)."""
    subset = flags[(flags["well"] == well) & (flags["threshold"] == threshold)]
    kept = subset.loc[~subset["own_flagged"], "ti"]
    return int(kept.iloc[-1]) if not kept.empty else None
