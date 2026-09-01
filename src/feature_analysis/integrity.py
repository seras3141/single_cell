"""Phase 2 #6 — pipeline-integrity / discontinuity monitor (mCherry-free).

Flags adjacent-timepoint distribution jumps per (well, feature): a median jump AND a shape
jump (both must fire). Would have caught the uint8 mask wrap.

METRIC NOTE (approved 2026-08-25; the plan now specs this): an earlier draft z-scored the median
step by the *delta-series* MAD. That fails this monitor's own
calibration target — a single clean jump in an otherwise-flat series gives a zero delta-series
MAD (undefined z), and a *bounce* (the uint8-wrap signature) has many large deltas so each
step's z stays ~1. So `median_jump` here normalizes the median step by the **within-timepoint
cell spread** (max of the two timepoints' scaled MADs): a discontinuity is a median moving many
cell-MADs. `shape_jump` is the KS between the two adjacent per-cell distributions. Degenerate
cases (too few cells, zero within-timepoint spread) are recorded via `status`, never silent NaN.

See docs/feature_analysis/plan_feature_variation_over_time.md.
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.feature_analysis.feature_trajectories import (
    BIOLOGICAL_FEATURES,
    FEATURE_GROUP,
    N_FLOOR,
    _resolve_features,
    _ti,
)
from src.feature_to_mcherry.data.normalize import _scaled_mad

logger = logging.getLogger(__name__)

ROBUST_Z_THRESH = 3.5   # median must move >= this many within-timepoint cell-MADs
KS_THRESH = 0.3

STATUS_OK = "ok"
STATUS_SKIPPED_LOW_N = "skipped_low_n"
STATUS_SKIPPED_ZERO_MAD = "skipped_zero_mad"


def compute_integrity_flags(
    cells: pd.DataFrame,
    features: Sequence[str] = BIOLOGICAL_FEATURES,
    n_floor: int = N_FLOOR,
    robust_z_thresh: float = ROBUST_Z_THRESH,
    ks_thresh: float = KS_THRESH,
) -> pd.DataFrame:
    """One row per (well, feature, adjacent-timepoint-pair). ``cells`` = per-cell-collapsed."""
    feats = _resolve_features(cells.columns, features)
    rows = []
    for well, gw in cells.groupby("sample_id"):
        by_ti = {}  # feature-agnostic: store the sub-frame per timepoint
        for tv, gt in gw.groupby("timepoint"):
            by_ti[tv] = gt
        tps = sorted(by_ti, key=_ti)
        for feat in feats:
            vals = {}
            for t in tps:
                v = by_ti[t][feat].to_numpy(float)
                vals[t] = v[np.isfinite(v)]
            for i in range(len(tps) - 1):
                t, tn = tps[i], tps[i + 1]
                vt, vn = vals[t], vals[tn]
                n_t, n_n = int(vt.size), int(vn.size)
                median_delta = mad_t = mad_n = scale_mad = robust_z = ks_stat = np.nan
                median_jump = shape_jump = flagged = False

                if n_t < n_floor or n_n < n_floor:
                    status = STATUS_SKIPPED_LOW_N
                else:
                    median_delta = float(np.median(vn) - np.median(vt))
                    mad_t, mad_n = _scaled_mad(vt), _scaled_mad(vn)
                    scale_mad = float(np.nanmax([mad_t, mad_n]))
                    ks_stat = float(ks_2samp(vn, vt).statistic)
                    shape_jump = bool(ks_stat >= ks_thresh)
                    if not (np.isfinite(scale_mad) and scale_mad > 0):
                        status = STATUS_SKIPPED_ZERO_MAD  # within-timepoint spread ~0
                    else:
                        robust_z = float(median_delta / scale_mad)
                        median_jump = bool(abs(robust_z) >= robust_z_thresh)
                        status = STATUS_OK
                    flagged = bool(median_jump and shape_jump)

                rows.append(dict(
                    sample_id=well, feature=feat, group=FEATURE_GROUP.get(feat),
                    ti=_ti(t), ti_next=_ti(tn), median_delta=median_delta,
                    mad_t=mad_t, mad_n=mad_n, scale_mad=scale_mad,
                    robust_z=robust_z, ks_stat=ks_stat,
                    n_cells_t=n_t, n_cells_tnext=n_n,
                    median_jump=median_jump, shape_jump=shape_jump, flagged=flagged,
                    status=status, robust_z_thresh=robust_z_thresh, ks_thresh=ks_thresh,
                ))
    return pd.DataFrame(rows).sort_values(["sample_id", "feature", "ti"]).reset_index(drop=True)
