"""Is a feature-vs-time trend estimable inside a culture's pre-collapse window?

Step 1 established that each experiment's DMSO reference collapses at its own pace,
and that for three of the five cultures only 2-3 mCherry timepoints precede that
collapse. A trend fitted to 2-3 points is not a trend, so this module decides *per
culture* whether a within-window feature-vs-time analysis is estimable at all, and
only then selects features and truncates the data for it.

The estimability verdict is the deliverable, not a gate on the rest: "this design cannot
resolve a pre-confluence trajectory for these cultures" is itself the answer to the
dataset-design question.

See ``docs/feature_to_mcherry/plan_fatima_deliverable_pipeline.md`` §Step 5'.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Pre-collapse timepoints at or above which a within-window trend is estimable.
ESTIMABLE_MIN_TIMEPOINTS = 10

#: Below :data:`ESTIMABLE_MIN_TIMEPOINTS` but at or above this, report with n stated.
BORDERLINE_MIN_TIMEPOINTS = 5

VERDICT_ESTIMABLE = "estimable"
VERDICT_BORDERLINE = "borderline"
VERDICT_NOT_ESTIMABLE = "not estimable (window too narrow)"

#: Default |rho| above which a (feature, target) pair is considered informative.
DEFAULT_RHO_THRESHOLD = 0.3

#: Scope value identifying pooled (not per-well) rows in univariate_correlations.csv.
POOLED_SCOPE = "pooled"


def classify_estimability(
    n_timepoints_pre_cross: Optional[int],
    *,
    estimable_min: int = ESTIMABLE_MIN_TIMEPOINTS,
    borderline_min: int = BORDERLINE_MIN_TIMEPOINTS,
) -> str:
    """Verdict for a pre-collapse window of ``n_timepoints_pre_cross`` samples.

    ``None`` is treated as not estimable rather than as unlimited: an unknown window is
    not evidence of a wide one.
    """
    if n_timepoints_pre_cross is None or pd.isna(n_timepoints_pre_cross):
        return VERDICT_NOT_ESTIMABLE
    n = int(n_timepoints_pre_cross)
    if n >= estimable_min:
        return VERDICT_ESTIMABLE
    if n >= borderline_min:
        return VERDICT_BORDERLINE
    return VERDICT_NOT_ESTIMABLE


def build_estimability_table(
    summary: pd.DataFrame,
    **classify_kwargs: Any,
) -> pd.DataFrame:
    """One row per experiment: its DMSO window width and the resulting verdict.

    Args:
        summary: Step 1's ``all_experiments_cell_population_summary.csv``.
        **classify_kwargs: Forwarded to :func:`classify_estimability`.

    Returns:
        Columns ``experiment``, ``dmso_well``, ``t_cross_peak``,
        ``n_timepoints_pre_cross``, ``end_pct_of_peak``, ``collapse_shape``,
        ``verdict``.

    Raises:
        ValueError: If an experiment has no row flagged ``is_dmso`` — the window is
            defined by the DMSO reference, so a missing one cannot be silently skipped.
    """
    rows = []
    for experiment, group in summary.groupby("experiment", sort=True):
        dmso = group[group["is_dmso"].astype(bool)]
        if dmso.empty:
            raise ValueError(
                f"{experiment}: no DMSO well in the summary. The window is defined by "
                f"the DMSO reference, so this experiment cannot be classified."
            )
        if len(dmso) > 1:
            raise ValueError(
                f"{experiment}: {len(dmso)} wells flagged is_dmso "
                f"({sorted(dmso['well'])}); expected exactly one reference."
            )
        row = dmso.iloc[0]
        n_pre = row["n_timepoints_pre_cross"]
        rows.append(
            {
                "experiment": experiment,
                "dmso_well": row["well"],
                "t_cross_peak": row["t_cross_peak"],
                "n_timepoints_pre_cross": n_pre,
                "end_pct_of_peak": row["end_pct_of_peak"],
                "collapse_shape": row["collapse_shape"],
                "verdict": classify_estimability(n_pre, **classify_kwargs),
            }
        )
    table = pd.DataFrame(rows)
    for verdict, count in table["verdict"].value_counts().items():
        logger.info("estimability: %d experiment(s) %s", count, verdict)
    return table


def select_informative_features(
    univariate: pd.DataFrame,
    *,
    rho_threshold: float = DEFAULT_RHO_THRESHOLD,
    scope: str = POOLED_SCOPE,
) -> pd.DataFrame:
    """Pooled (feature, target) pairs whose ``|rho|`` exceeds ``rho_threshold``.

    Filters ``scope`` explicitly. ``univariate_correlations.csv`` holds two scopes —
    ``pooled`` and ``per_group`` (per well) — and only ~1 row in 10 is pooled, so a
    selection that forgets to filter silently mixes per-well rows into a pooled claim.

    Returns:
        The surviving rows with an added ``abs_rho``, sorted strongest first.

    Raises:
        ValueError: If the table has no rows at the requested scope, which means the
            input is not the expected univariate output.
    """
    missing = [c for c in ("feature", "target", "scope", "rho") if c not in univariate]
    if missing:
        raise ValueError(f"univariate table missing columns: {missing}")

    pooled = univariate[univariate["scope"] == scope].copy()
    if pooled.empty:
        raise ValueError(
            f"no rows with scope=={scope!r} (scopes present: "
            f"{sorted(univariate['scope'].dropna().unique())})"
        )

    pooled["abs_rho"] = pooled["rho"].abs()
    selected = pooled[pooled["abs_rho"] > rho_threshold]
    logger.info(
        "selected %d/%d pooled pairs at |rho| > %g",
        len(selected),
        len(pooled),
        rho_threshold,
    )
    return selected.sort_values("abs_rho", ascending=False).reset_index(drop=True)


def top_features(selected: pd.DataFrame, *, limit: int = 6) -> list:
    """The ``limit`` strongest distinct features, deduplicated across targets.

    A feature-vs-time plot shows the feature alone; the target only decides *which*
    features are worth plotting, so the same feature selected via three targets is one
    panel, not three.
    """
    return list(selected["feature"].drop_duplicates().head(limit))


def truncate_to_pre_collapse(
    metadata: pd.DataFrame,
    summary: pd.DataFrame,
    experiment: str,
    *,
    sample_id_column: str = "sample_id",
    timepoint_column: str = "timepoint",
) -> "pd.Series[bool]":
    """Boolean mask keeping rows at or before each well's own ``t_cross_peak``.

    Wells that never cross keep **every** timepoint — a null ``t_cross_peak`` means "no
    collapse", not "no data". Wells absent from the summary also keep everything, with a
    warning: dropping them silently would understate the surviving data.

    Args:
        metadata: Per-cell rows carrying well id and timepoint (from
            ``build_matrix_with_metadata``).
        summary: Step 1's summary table.
        experiment: Which experiment's rows to look up in ``summary``.
    """
    wells = summary[summary["experiment"] == experiment]
    if wells.empty:
        raise ValueError(f"{experiment}: no rows in the Step 1 summary")
    cross = wells.set_index("well")["t_cross_peak"].to_dict()

    timepoints = pd.to_numeric(metadata[timepoint_column], errors="coerce")
    keep = pd.Series(True, index=metadata.index)

    unknown = set()
    for well, well_rows in metadata.groupby(metadata[sample_id_column].astype(str)):
        if well not in cross:
            unknown.add(well)
            continue
        limit = cross[well]
        if pd.isna(limit):
            continue  # never collapses -> every timepoint is usable
        keep.loc[well_rows.index] = timepoints.loc[well_rows.index] <= float(limit)

    if unknown:
        logger.warning(
            "%s: %d well(s) absent from the summary, keeping all their timepoints: %s",
            experiment,
            len(unknown),
            sorted(unknown),
        )
    logger.info(
        "%s: kept %d/%d cell-rows after pre-collapse truncation",
        experiment,
        int(keep.sum()),
        len(keep),
    )
    return keep


def within_window_trend(
    metadata: pd.DataFrame,
    features: pd.DataFrame,
    feature_names: Sequence[str],
    dmso_t_cross: Optional[float],
    *,
    timepoint_column: str = "timepoint",
) -> pd.DataFrame:
    """Does each feature actually vary with time *inside* the pre-collapse window?

    The plots answer this by eye; this answers it numerically, which is what the
    deliverable needs. Splits at the DMSO reference's own crossing and reports, per
    feature, the Spearman correlation with time **within** the window (the question that
    matters) alongside the median on each side.

    A near-zero within-window rho means the design cannot see morphology change before
    confluency, however clean the culture looks — a materially different conclusion from
    "no trend was plotted".

    Returns:
        One row per feature: ``n_within``, ``n_post``, ``rho_within``, ``p_within``,
        ``median_within``, ``median_post``, ``pct_change_post_vs_within``.
    """
    # Reuse the informativeness module's own guarded wrapper rather than calling
    # scipy directly: it tuple-unpacks (``result.statistic`` only exists from scipy
    # 1.9, but pyproject allows >=1.7.0) and returns (nan, nan) for a constant or
    # too-small input instead of warning. Every other spearmanr call site in this
    # package unpacks the same way.
    from src.feature_to_mcherry.informativeness.univariate import _safe_spearmanr

    times = pd.to_numeric(metadata[timepoint_column], errors="coerce").to_numpy(
        dtype=float
    )
    within = (
        np.isfinite(times)
        if dmso_t_cross is None or pd.isna(dmso_t_cross)
        else np.isfinite(times) & (times <= float(dmso_t_cross))
    )
    post = np.isfinite(times) & ~within

    rows = []
    for name in feature_names:
        values = features[name].to_numpy(dtype=float)
        finite = np.isfinite(values)
        w = within & finite
        p = post & finite
        rho: Optional[float] = None
        pval: Optional[float] = None
        if w.any():
            # _safe_spearmanr itself guards n < 3 and constant input, returning nan.
            rho, pval = _safe_spearmanr(times[w], values[w])
        median_within = float(np.median(values[w])) if w.any() else None
        median_post = float(np.median(values[p])) if p.any() else None
        rows.append(
            {
                "feature": name,
                "n_within": int(w.sum()),
                "n_post": int(p.sum()),
                "rho_within": rho,
                "p_within": pval,
                "median_within": median_within,
                "median_post": median_post,
                "pct_change_post_vs_within": (
                    None
                    if not median_within or median_post is None
                    else 100.0 * (median_post - median_within) / median_within
                ),
            }
        )
    return pd.DataFrame(rows)


def summarise_selection(
    selection_by_experiment: Mapping[str, pd.DataFrame],
    rho_threshold: float,
) -> pd.DataFrame:
    """Per experiment: how many pooled pairs cleared the threshold, and the strongest.

    Surfaces a thin selection rather than letting it pass silently — the plan flags
    SA110 as the binding case.
    """
    rows: list = []
    for experiment, selected in selection_by_experiment.items():
        strongest = selected.iloc[0] if not selected.empty else None
        rows.append(
            {
                "experiment": experiment,
                "rho_threshold": rho_threshold,
                "n_pairs_selected": len(selected),
                "n_distinct_features": selected["feature"].nunique(),
                "max_abs_rho": None if strongest is None else strongest["abs_rho"],
                "strongest_pair": (
                    None
                    if strongest is None
                    else f"{strongest['feature']} ~ {strongest['target']}"
                ),
            }
        )
    return pd.DataFrame(rows)
