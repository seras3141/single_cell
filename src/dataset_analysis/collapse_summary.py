"""Per-well collapse metrics derived from a cell-population trajectory table.

Consumes the output of
:func:`src.dataset_analysis.cell_population.compute_cell_population` (one row per
``(sample_id, timepoint)``) and reduces each well's trajectory to a single row of
collapse metrics: peak/edge counts, both end-percentage denominators, both
``t_cross`` variants, the width of the usable pre-collapse window, and a shape label.

**Two references, deliberately.** ``peak`` (the trajectory maximum) is the primary
reference; ``base_first3`` (mean of the first three sampled timepoints) is reported
alongside it only for continuity with the deprecated ``docs/_phase6_scratch`` tables.
``base_first3`` is *not* a plateau for this dataset — sampling is ~2 h apart and several
wells drop severalfold across their first three samples, so a "baseline" measured there
already sits partway down the collapse. Prefer the ``*_peak`` columns for anything
reported.

The shape rule and the 50 %-of-reference collapse criterion are carried over unchanged
from ``docs/_phase6_scratch/dmso_vs_drug_cell_count/analyze_dmso_vs_drug.py``
(``characterize()``), so the two remain comparable; the reference against which they are
evaluated is ``peak`` here rather than that script's first-3 mean.

See ``docs/feature_to_mcherry/plan_fatima_deliverable_pipeline.md`` §Step 1.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from src.dataset_analysis.layout import get_well_annotation

logger = logging.getLogger(__name__)

#: Fraction of the reference count below which a well counts as collapsed.
#: Matches ``DISTINCT_DROP_FRAC`` in the deprecated scratch script.
DEFAULT_COLLAPSE_FRACTION = 0.5

#: Number of leading/trailing samples averaged for the edge counts.
#: Matches ``EDGE_K`` in the deprecated scratch script.
DEFAULT_EDGE_N = 3

#: Maximum gap (in samples) between the peak and the crossing for a "sharp step".
DEFAULT_SHARP_STEP_GAP = 3

#: Columns emitted by :func:`summarize_experiment`, in order.
SUMMARY_COLUMNS = (
    "experiment",
    "well",
    "drug",
    "dose_rank",
    "concentration_uM",
    "is_dmso",
    "peak_n_cells",
    "base_first3_n_cells",
    "end_n_cells",
    "end_pct_of_peak",
    "end_pct_of_first3",
    "t_cross_peak",
    "t_cross_first3",
    "n_timepoints_pre_cross",
    "collapse_shape",
    "distinct_collapses",
)

_SHAPE_NO_COLLAPSE = "no material collapse"
_SHAPE_SHARP = "sharp step"
_SHAPE_GRADUAL = "gradual"


def _first_crossing(
    counts: "pd.Series[Any]", times: "pd.Series[Any]", threshold: float
) -> Optional[int]:
    """First value in ``times`` whose paired count is strictly below ``threshold``."""
    below = times[counts < threshold]
    if below.empty:
        return None
    return int(below.iloc[0])


def _classify_shape(
    counts: "pd.Series[Any]",
    times: "pd.Series[Any]",
    t_cross: Optional[int],
    collapsed: bool,
    sharp_step_gap: int,
) -> str:
    """Shape label, following ``analyze_dmso_vs_drug.py``'s ``characterize()``.

    The peak is located within the *first half* of the series (so a late rebound cannot
    masquerade as the peak), and the crossing is called a sharp step when it follows
    that peak within ``sharp_step_gap`` samples. Positional indices are used throughout;
    the original relied on a fresh ``RangeIndex`` making labels and positions coincide.
    """
    if not collapsed or t_cross is None:
        return _SHAPE_NO_COLLAPSE
    half = max(1, len(counts) // 2)
    peak_pos = int(np.argmax(counts.to_numpy()[:half]))
    cross_positions = np.flatnonzero(times.to_numpy() == t_cross)
    if cross_positions.size == 0:  # pragma: no cover - t_cross comes from `times`
        return _SHAPE_NO_COLLAPSE
    cross_pos = int(cross_positions[0])
    return _SHAPE_SHARP if (cross_pos - peak_pos) <= sharp_step_gap else _SHAPE_GRADUAL


def compute_collapse_metrics(
    trajectory: pd.DataFrame,
    *,
    count_column: str = "n_cells",
    time_column: str = "ti",
    collapse_fraction: float = DEFAULT_COLLAPSE_FRACTION,
    edge_n: int = DEFAULT_EDGE_N,
    sharp_step_gap: int = DEFAULT_SHARP_STEP_GAP,
) -> Dict[str, Any]:
    """Reduce one well's trajectory to its collapse metrics.

    Args:
        trajectory: Rows for a single well. Sorted by ``time_column`` internally, so the
            caller need not pre-sort — but the column must be numeric, not a string
            (``"11"`` sorts before ``"2"`` lexically, which would silently scramble the
            time axis).
        count_column: Per-timepoint cell count. Defaults to ``cell_population.csv``'s
            ``n_cells``.
        time_column: Numeric frame index. Defaults to ``cell_population.csv``'s ``ti``.
        collapse_fraction: A well is collapsed when its end count falls below this
            fraction of its peak.
        edge_n: Samples averaged at each end for ``base_first3``/``end``.
        sharp_step_gap: Peak-to-crossing gap, in samples, still counted as a sharp step.

    Returns:
        A dict with the metric subset of :data:`SUMMARY_COLUMNS`. ``t_cross_*`` are
        ``None`` when the well never crosses. ``n_timepoints_pre_cross`` is the number
        of samples at or before ``t_cross_peak`` — and the **full** sample count when
        there is no crossing, since every timepoint is usable in that case. Distinguish
        the two cases via ``t_cross_peak``, not by testing this count against zero.

    Raises:
        ValueError: If ``trajectory`` is empty or a required column is missing.
    """
    missing = [c for c in (count_column, time_column) if c not in trajectory.columns]
    if missing:
        raise ValueError(f"trajectory missing required columns: {missing}")
    if trajectory.empty:
        raise ValueError("trajectory is empty; cannot compute collapse metrics")

    ordered = trajectory.sort_values(time_column).reset_index(drop=True)
    counts = ordered[count_column].astype(float)
    times = ordered[time_column].astype(int)

    peak = float(counts.max())
    base_first3 = float(counts.head(edge_n).mean())
    end = float(counts.tail(edge_n).mean())

    pct_of_peak = 100.0 * end / peak if peak else np.nan
    pct_of_first3 = 100.0 * end / base_first3 if base_first3 else np.nan

    t_cross_peak = (
        _first_crossing(counts, times, collapse_fraction * peak) if peak else None
    )
    t_cross_first3 = (
        _first_crossing(counts, times, collapse_fraction * base_first3)
        if base_first3
        else None
    )

    collapsed = bool(peak) and (end / peak) < collapse_fraction
    shape = _classify_shape(counts, times, t_cross_peak, collapsed, sharp_step_gap)

    if t_cross_peak is None:
        n_pre_cross = int(len(times))
    else:
        n_pre_cross = int((times <= t_cross_peak).sum())

    return {
        "peak_n_cells": peak,
        "base_first3_n_cells": base_first3,
        "end_n_cells": end,
        "end_pct_of_peak": pct_of_peak,
        "end_pct_of_first3": pct_of_first3,
        "t_cross_peak": t_cross_peak,
        "t_cross_first3": t_cross_first3,
        "n_timepoints_pre_cross": n_pre_cross,
        "collapse_shape": shape,
        "distinct_collapses": collapsed,
    }


def _dose_rank(
    drug: Optional[str], concentration_uM: Optional[float], layout: Mapping[str, Any]
) -> Optional[int]:
    """Concentration tier (1 = highest) for a drug well, by matching the layout ladder.

    Derived from ``concentration_uM`` rather than from the well's plate column, so this
    never re-implements the plate's irregular column geometry — the source of the
    column-7 mislabelling bug fixed in ``b35396f``.
    """
    if not drug or concentration_uM is None:
        return None
    ladder = layout.get("drugs", {}).get(drug, {}).get("concentrations_uM", {})
    for key, value in ladder.items():
        if value == concentration_uM:
            return int(str(key).rsplit("_", 1)[-1])
    return None


def _well_annotation(well: str, layout: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Public-API annotation for one well id, tolerant of unparseable ids."""
    if layout is None:
        return {"content": None, "drug": None, "concentration_uM": None}
    row_letters = "".join(c for c in str(well) if c.isalpha())
    digits = "".join(c for c in str(well) if c.isdigit())
    if not row_letters or not digits:
        return {"content": None, "drug": None, "concentration_uM": None}
    try:
        return dict(get_well_annotation(row_letters.upper(), int(digits), layout))
    except Exception:  # unknown row/column - leave unannotated rather than crash
        logger.warning("could not annotate well %r from the plate layout", well)
        return {"content": None, "drug": None, "concentration_uM": None}


def summarize_experiment(
    cell_population: pd.DataFrame,
    experiment: str,
    *,
    layout: Optional[Mapping[str, Any]] = None,
    dmso_well: Optional[str] = None,
    well_column: str = "sample_id",
    **metric_kwargs: Any,
) -> pd.DataFrame:
    """One row of collapse metrics per well in ``cell_population``.

    Args:
        cell_population: A ``cell_population.csv`` table for a single experiment.
        experiment: Label written to the ``experiment`` column.
        layout: Loaded plate layout, for ``drug``/``concentration_uM``/``dose_rank``.
            When omitted, those columns fall back to the table's own ``drug`` column
            (if present) and ``None``.
        dmso_well: Well id of the DMSO reference. Falls back to the table's ``is_dmso``
            column when not given.
        well_column: Column holding the well id.
        **metric_kwargs: Forwarded to :func:`compute_collapse_metrics`.

    Returns:
        A DataFrame with :data:`SUMMARY_COLUMNS`, one row per well, sorted by well id.
    """
    if well_column not in cell_population.columns:
        raise ValueError(f"cell_population missing well column {well_column!r}")

    rows = []
    for well, group in cell_population.groupby(well_column, sort=True):
        annotation = _well_annotation(str(well), layout)
        drug = annotation.get("drug")
        concentration = annotation.get("concentration_uM")

        if drug is None and annotation.get("content") == "control":
            # Control wells carry no `drug`; their identity is in `control`
            # ("DMSO", "Benzethonium Chloride"). Label them with that rather than
            # leaving `drug` null, which the composition guard would read as an
            # unannotated well.
            drug = annotation.get("control")

        if layout is None and drug is None and "drug" in group.columns:
            # Only when there is no layout to consult: the table's own resolved value is
            # then the sole source. Deliberately NOT a fallback when a layout *is*
            # supplied -- cell_population.csv's own `drug` column is itself produced by
            # get_well_annotation, so borrowing it would paper over a failed layout
            # lookup and leave `dose_rank`/`concentration_uM` silently empty while
            # `drug` still looked right. That would defeat assert_well_composition,
            # which exists to catch exactly the column-7 regression fixed in b35396f.
            fallback = group["drug"].dropna()
            drug = str(fallback.iloc[0]) if not fallback.empty else None

        if dmso_well is not None:
            is_dmso = str(well).upper() == str(dmso_well).upper()
        elif "is_dmso" in group.columns:
            is_dmso = bool(group["is_dmso"].iloc[0])
        else:
            is_dmso = False

        record: Dict[str, Any] = {
            "experiment": experiment,
            "well": str(well),
            "drug": drug,
            "dose_rank": (
                _dose_rank(drug, concentration, layout) if layout is not None else None
            ),
            "concentration_uM": concentration,
            "is_dmso": is_dmso,
        }
        record.update(compute_collapse_metrics(group, **metric_kwargs))
        rows.append(record)

    return pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS))


def assert_well_composition(
    summary: pd.DataFrame,
    *,
    expected_drug_wells: int = 8,
    expected_dmso_wells: int = 1,
) -> None:
    """Guard against the plate-layout regression fixed in ``b35396f``.

    Before that fix, ``get_well_annotation`` returned ``empty`` for every column-7 well
    — two of each experiment's nine imaged wells, and precisely the top-dose ones. This
    check is cheap and is what would have caught it, so it runs on every summary.

    The check covers the **layout-derived dose fields**, not just ``drug``. Checking
    ``drug`` alone is insufficient: it can be populated from a source other than the
    layout, in which case a failed layout lookup would leave ``dose_rank`` and
    ``concentration_uM`` empty while ``drug`` still looked correct.

    Raises:
        AssertionError: If any experiment has a well annotated ``empty``/unannotated, a
            drug/DMSO well count other than the expected one, or a drug well missing its
            layout-derived ``dose_rank``/``concentration_uM``.
    """
    for experiment, group in summary.groupby("experiment", sort=True):
        # Control wells are exempt from the drug check: they have no drug by design.
        drug_rows = group[~group["is_dmso"].astype(bool)]
        empty = drug_rows[
            drug_rows["drug"].isin([None, "empty"]) | drug_rows["drug"].isna()
        ]
        if not empty.empty:
            raise AssertionError(
                f"{experiment}: {len(empty)} well(s) unannotated or 'empty' "
                f"({sorted(empty['well'])}) — plate-layout regression, see plan §B1"
            )
        n_dmso = int(group["is_dmso"].sum())
        n_drug = len(group) - n_dmso
        if n_dmso != expected_dmso_wells or n_drug != expected_drug_wells:
            raise AssertionError(
                f"{experiment}: expected {expected_drug_wells} drug + "
                f"{expected_dmso_wells} DMSO wells, got {n_drug} + {n_dmso}"
            )

        dose_columns = [
            c for c in ("dose_rank", "concentration_uM") if c in group.columns
        ]
        if dose_columns:
            drug_wells = group[~group["is_dmso"].astype(bool)]
            incomplete = drug_wells[drug_wells[dose_columns].isna().any(axis=1)]
            if not incomplete.empty:
                raise AssertionError(
                    f"{experiment}: {len(incomplete)} drug well(s) "
                    f"({sorted(incomplete['well'])}) have a drug but no layout-derived "
                    f"{'/'.join(dose_columns)} — the layout lookup failed even though "
                    f"'drug' is set; plate-layout regression, see plan §B1"
                )

        logger.info(
            "%s: well composition OK (%d drug + %d DMSO, dose fields complete)",
            experiment,
            n_drug,
            n_dmso,
        )


def summarize_all_experiments(
    cell_population_csvs: Mapping[str, Union[str, Path]],
    *,
    layout: Optional[Mapping[str, Any]] = None,
    dmso_wells: Optional[Mapping[str, str]] = None,
    **metric_kwargs: Any,
) -> pd.DataFrame:
    """Concatenate per-experiment summaries into one table.

    Args:
        cell_population_csvs: ``{experiment_label: path to cell_population.csv}``.
        layout: Loaded plate layout, shared across experiments.
        dmso_wells: Optional ``{experiment_label: dmso well id}`` override.
        **metric_kwargs: Forwarded to :func:`compute_collapse_metrics`.

    Returns:
        A DataFrame with :data:`SUMMARY_COLUMNS`, sorted by experiment then well.
    """
    frames = []
    for experiment, csv_path in cell_population_csvs.items():
        path = Path(csv_path)
        if not path.is_file():
            raise FileNotFoundError(f"{experiment}: no cell_population.csv at {path}")
        table = pd.read_csv(path)
        logger.info("%s: read %d rows from %s", experiment, len(table), path)
        frames.append(
            summarize_experiment(
                table,
                experiment,
                layout=layout,
                dmso_well=(dmso_wells or {}).get(experiment),
                **metric_kwargs,
            )
        )

    if not frames:
        return pd.DataFrame(columns=list(SUMMARY_COLUMNS))
    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values(["experiment", "well"]).reset_index(drop=True)


def dmso_reference_table(summary: pd.DataFrame) -> pd.DataFrame:
    """The DMSO-well rows only, with both end-percentage denominators side by side.

    This is the table quoted in the Step 1 markdown summary; keeping it here means the
    numbers in the write-up and the CSV cannot drift apart.
    """
    columns: Sequence[str] = (
        "experiment",
        "well",
        "peak_n_cells",
        "base_first3_n_cells",
        "end_n_cells",
        "end_pct_of_peak",
        "end_pct_of_first3",
        "t_cross_peak",
        "t_cross_first3",
        "n_timepoints_pre_cross",
        "collapse_shape",
    )
    dmso = summary[summary["is_dmso"]].copy()
    return dmso[list(columns)].sort_values("experiment").reset_index(drop=True)
