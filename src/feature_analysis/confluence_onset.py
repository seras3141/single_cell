"""Phase 2 #4 — confluence-onset / healthy-window boundary (mCherry-free).

Per well, three onset signals reported side-by-side plus a consensus:
  - population onset: ``t_cross_peak`` CONSUMED from the shipped 45-well summary CSV
    (``dataset_analysis.collapse_summary``); first ti where n_cells < 0.5 * global peak.
  - coverage onset: first ti where coverage_fraction < 0.5 * max coverage, sustained >=2 frames
    (read from per-experiment cell_population.csv).
  - feature-drift onset (control-chart): first ti (after the early window) where >=50% of
    biological features have their median departed > K * the early *temporal* scale (scaled-MAD
    of the early per-timepoint medians, floored at 2%*|center|) from the early-window median,
    sustained >=2 frames. Advisory/noisy at 3 early points; the consensus is robust to it.
The consensus is the median of the available onsets; not a single hard-coded signal.
See docs/feature_analysis/plan_feature_variation_over_time.md.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.feature_analysis.feature_trajectories import BIOLOGICAL_FEATURES
from src.feature_to_mcherry.data.normalize import _scaled_mad
from src.feature_to_mcherry.pre_collapse import classify_estimability

logger = logging.getLogger(__name__)

DROP_FRACTION = 0.5           # onset = first crossing below this fraction of the peak
SUSTAIN = 2                   # consecutive sampled frames required to count a crossing
DRIFT_FEATURE_FRACTION = 0.5  # >= this fraction of features must have moved...
EARLY_WINDOW = 3              # first N sampled timepoints define the early baseline
# ... a feature "has moved" (control-chart style) when its median departs from the early-window
# median-center by more than DRIFT_ONSET_K * early *temporal* scale, where the temporal scale is
# the scaled-MAD of the early per-timepoint medians floored at DRIFT_ONSET_REL_FLOOR*|center|.
# (Normalizing by the early temporal variability of the MEDIAN, NOT the cross-cell population
# IQR — the latter is far too large, so nothing ever crossed. Recalibrated 2026-09-01.)
DRIFT_ONSET_K = 3.0
DRIFT_ONSET_REL_FLOOR = 0.02


def _sustained_first_crossing(ti: np.ndarray, below: np.ndarray, sustain: int = SUSTAIN) -> Optional[int]:
    """First ti where ``below`` is True for ``sustain`` consecutive sampled frames; else None."""
    run = 0
    for i in range(len(ti)):
        run = run + 1 if below[i] else 0
        if run >= sustain:
            return int(ti[i - sustain + 1])
    return None


def _coverage_onset(cov: pd.DataFrame) -> Optional[int]:
    """First sustained drop below DROP_FRACTION*peak, searched only AFTER the coverage peak.

    Coverage RAMPS UP from seeding to a late peak, so an unrestricted search would flag the
    early growth ramp (every early frame is < 0.5*late-peak). Onset is a post-peak *collapse*.
    """
    g = cov.dropna(subset=["coverage_fraction"]).sort_values("ti").reset_index(drop=True)
    if g.empty:
        return None
    peak = g["coverage_fraction"].max()
    if not (peak > 0):
        return None
    peak_idx = int(g["coverage_fraction"].to_numpy().argmax())
    gp = g.iloc[peak_idx:]  # only frames at/after the coverage peak
    below = gp["coverage_fraction"].to_numpy(float) < DROP_FRACTION * peak
    return _sustained_first_crossing(gp["ti"].to_numpy(float), below)


def _drift_onset(traj_well: pd.DataFrame, features: Sequence[str]) -> Optional[int]:
    """One well's feature-drift onset: first ti where >= DRIFT_FEATURE_FRACTION of features have
    their median departed > DRIFT_ONSET_K * early-temporal-scale from the early median, sustained.

    traj_well: rows (feature, ti, median). The scale is the scaled-MAD of each feature's early
    per-timepoint medians (temporal variability of the median), floored relative to the center.
    """
    feats = [f for f in features if f in set(traj_well["feature"])]
    if not feats:
        return None
    tis = sorted(traj_well["ti"].unique())
    if len(tis) < EARLY_WINDOW + SUSTAIN:
        return None
    early_tis = tis[:EARLY_WINDOW]
    base = {}  # feature -> (center, scale, {ti: median})
    for f in feats:
        gf = traj_well[traj_well["feature"] == f]
        med_by_ti = dict(zip(gf["ti"], gf["median"]))
        early_meds = np.array([med_by_ti[t] for t in early_tis if t in med_by_ti], float)
        if early_meds.size < 2:
            continue
        center = float(np.median(early_meds))
        scale = float(_scaled_mad(early_meds))
        floor = max(DRIFT_ONSET_REL_FLOOR * abs(center), 1e-9)
        scale = max(scale, floor) if np.isfinite(scale) else floor
        base[f] = (center, scale, med_by_ti)
    if not base:
        return None
    frac_moved = []
    for t in tis:
        moved = tot = 0
        for f, (center, scale, mbt) in base.items():
            if t not in mbt:
                continue
            tot += 1
            if abs(float(mbt[t]) - center) > DRIFT_ONSET_K * scale:
                moved += 1
        frac_moved.append((moved / tot) if tot else 0.0)
    moved_flag = np.array(frac_moved) >= DRIFT_FEATURE_FRACTION
    # onset must be OUTSIDE the early baseline window it is measured against -> search from EARLY_WINDOW
    return _sustained_first_crossing(
        np.array(tis[EARLY_WINDOW:], float), moved_flag[EARLY_WINDOW:])


def compute_confluence_onset(
    traj: pd.DataFrame,
    cell_population: pd.DataFrame,
    summary: pd.DataFrame,
    experiment_label: str,
    dmso_well: str,
    features: Sequence[str] = BIOLOGICAL_FEATURES,
) -> pd.DataFrame:
    """Per-well onset table (population / coverage / drift) + consensus.

    Args:
        traj: this experiment's trajectory table (from compute_trajectories).
        cell_population: this experiment's cell_population.csv (sample_id, ti, coverage_fraction).
        summary: all_experiments_cell_population_summary.csv (has t_cross_peak, is_dmso, …).
        experiment_label: e.g. "HD1509" — matched against summary.experiment.
        dmso_well: the DMSO well id, for the estimability verdict + is_dmso flag.
    """
    # population onset from the summary CSV (matched to this experiment)
    s = summary.copy()
    if "experiment" in s.columns:
        s = s[s["experiment"].astype(str).str.contains(experiment_label, case=False, na=False, regex=False)]
    pop = {str(r["well"]).upper(): r.get("t_cross_peak") for _, r in s.iterrows()}
    dmso_pre = None
    if len(s):
        dm = s[s["well"].astype(str).str.upper() == str(dmso_well).upper()]
        if len(dm) and "n_timepoints_pre_cross" in dm.columns:
            v = dm.iloc[0]["n_timepoints_pre_cross"]
            dmso_pre = None if pd.isna(v) else int(v)
    verdict = classify_estimability(dmso_pre)

    rows = []
    for well in sorted(traj["sample_id"].unique()):
        onset_pop = pop.get(str(well).upper())
        onset_pop = None if onset_pop is None or pd.isna(onset_pop) else int(onset_pop)
        cov = cell_population[cell_population["sample_id"].astype(str).str.upper() == str(well).upper()]
        onset_cov = _coverage_onset(cov) if len(cov) else None
        onset_drift = _drift_onset(traj[traj["sample_id"] == well], features)
        avail = [x for x in (onset_pop, onset_cov, onset_drift) if x is not None]
        # round-half (not truncate): with 3 signals median is a real sampled ti; with 2 it is a
        # midpoint estimate between two frames (documented — not necessarily a sampled ti).
        consensus = int(round(float(np.median(avail)))) if avail else None
        span = (max(avail) - min(avail)) if len(avail) >= 2 else 0
        rows.append(dict(
            sample_id=well, is_dmso=str(well).upper() == str(dmso_well).upper(),
            onset_pop_ti=onset_pop, onset_cov_ti=onset_cov, onset_drift_ti=onset_drift,
            onset_consensus_ti=consensus, n_signals=len(avail), disagreement_span=span,
            estimable=verdict,
        ))
    return pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)
