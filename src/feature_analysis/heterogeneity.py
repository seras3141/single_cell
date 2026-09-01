"""Phase 2 #5 — within-well heterogeneity over time (mCherry-free).

Across-cell spread of each morphology feature within a (well, timepoint), on the
per-cell-collapsed values (median-over-z), and its trend over time. A rising trend =
subpopulation emergence. See docs/feature_analysis/plan_feature_variation_over_time.md.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import theilslopes

from src.feature_analysis.feature_trajectories import (
    BIOLOGICAL_FEATURES,
    FEATURE_GROUP,
    N_FLOOR,
    grouped_feature_stats,
)
from src.feature_to_mcherry.pre_collapse import classify_estimability

logger = logging.getLogger(__name__)

#: Denominator floor for rIQR = IQR/|median|; below this the ratio is meaningless
#: (features whose median rides near 0, e.g. skewness/kurtosis).
RIQR_EPS = 1e-9


def compute_heterogeneity(
    cells: pd.DataFrame,
    features: Sequence[str] = BIOLOGICAL_FEATURES,
    n_floor: int = N_FLOOR,
    eps: float = RIQR_EPS,
) -> pd.DataFrame:
    """Per (well, timepoint, feature) across-cell dispersion, plus a per-(well,timepoint)
    ``feature='__overall__'`` mean rIQR over valid, confident biological features.

    ``cells`` must be the per-cell-collapsed frame (one row per (well, timepoint, cell_id)).
    """
    het = grouped_feature_stats(cells, features, with_mad=True)
    het["riqr_valid"] = het["median"].abs() >= eps
    het["rIQR"] = np.where(het["riqr_valid"], het["iqr"] / het["median"].abs(), np.nan)
    het["low_confidence"] = het["n_cells"] < n_floor
    het["group"] = het["feature"].map(FEATURE_GROUP)
    het = het.drop(columns=["q25", "q75"])

    # __overall__: mean rIQR over valid + confident biological features per (well, timepoint)
    ok = het[het["riqr_valid"] & ~het["low_confidence"]]
    ov = (
        ok.groupby(["sample_id", "timepoint", "ti"])["rIQR"].mean().reset_index()
        .rename(columns={"rIQR": "rIQR_mean"})
    )
    ov_rows = ov.assign(
        feature="__overall__", group="overall", median=np.nan, mad=np.nan, iqr=np.nan,
        riqr_valid=ov["rIQR_mean"].notna(),
        rIQR=ov["rIQR_mean"], n_cells=-1, low_confidence=ov["rIQR_mean"].isna(),
    ).drop(columns=["rIQR_mean"])

    out = pd.concat([het, ov_rows[het.columns]], ignore_index=True)
    return out.sort_values(["feature", "sample_id", "ti"]).reset_index(drop=True)


def compute_heterogeneity_trend(
    het: pd.DataFrame,
    dmso_n_timepoints_pre_cross: Optional[int] = None,
) -> pd.DataFrame:
    """Per (well, feature) Theil-Sen slope of rIQR over time (valid rows only).

    A positive slope = heterogeneity increasing over time. ``estimable`` gates biological reads.
    """
    verdict = classify_estimability(dmso_n_timepoints_pre_cross)
    sub = het[(het["feature"] != "__overall__") & het["riqr_valid"] & ~het["low_confidence"]]
    rows = []
    for (well, feat), g in sub.groupby(["sample_id", "feature"]):
        g = g.sort_values("ti")
        ti = g["ti"].to_numpy(float)
        y = g["rIQR"].to_numpy(float)
        n = len(g)
        slope = np.nan
        if n >= 3 and np.nanstd(ti) > 0 and np.nanstd(y) > 0:
            slope = float(theilslopes(y, ti)[0])
        elif n >= 3 and np.nanstd(ti) > 0:
            slope = 0.0  # confidently flat
        rows.append(dict(sample_id=well, feature=feat, group=FEATURE_GROUP.get(feat),
                         slope=slope, n_timepoints=n, estimable=verdict))
    return pd.DataFrame(rows).sort_values(["feature", "sample_id"]).reset_index(drop=True)
