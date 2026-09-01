"""Feature variation over time — Phase 1: trajectories, drift, divergence-from-DMSO.

mCherry-free analysis of how morphology features change across the 72 h timelapse, per
well. See ``docs/feature_analysis/plan_feature_variation_over_time.md`` (narrowed +
codex-reviewed 2026-08-25).

Design decisions folded from the codex cross-check:
- Observation unit is the **cell**: per-z detections are collapsed to one value per
  (well, timepoint, cell) by median-over-z before any statistic (matches the project's
  per-cell observation-unit convention). Heterogeneity (Phase 2) then measures the
  across-cell spread.
- Feature groups: **shape (12)** + **intensity (4)** are the biological set; **spatial (5)**
  are excluded and consumed by no metric; **gabor (2)** dropped (aliased/unreliable).
- Divergence: KS is the PRIMARY distance when DMSO ``n < 30`` or MAD≈0; the MAD-standardized
  median shift (reusing ``normalize._scaled_mad``) is carried as an interpretable effect size
  with per-feature n/MAD validity and a ``low_confidence`` flag.
- Feasibility gate: ``pre_collapse.classify_estimability`` on the DMSO well's pre-collapse
  timepoint count decides whether a *biological* trend is estimable (HD1509 estimable,
  HD1883 borderline, the other three not).
"""
from __future__ import annotations

import glob
import logging
import re
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr, theilslopes

from src.dataset_analysis.layout import get_well_annotation
from src.feature_to_mcherry.data.collapse import collapse_slices_to_cells
from src.feature_to_mcherry.data.normalize import _scaled_mad
from src.feature_to_mcherry.dataset_design import cliffs_delta as _cliffs_delta
from src.feature_to_mcherry.pre_collapse import classify_estimability

_WELL_RE = re.compile(r"^([A-Za-z]+)(\d+)$")

logger = logging.getLogger(__name__)

KEYS = ("cell_id", "sample_id", "timepoint", "z_index")

SHAPE_FEATURES = (
    "area", "perimeter", "elongation", "compactness", "circularity", "feret_diameter",
    "radius_of_gyration", "major_axis", "minor_axis", "skewness", "kurtosis", "entropy",
)
INTENSITY_FEATURES = ("mean_intensity", "std_intensity", "cv_intensity", "total_intensity")
SPATIAL_FEATURES = (
    "centroid_x", "centroid_y", "center_of_mass_x", "center_of_mass_y", "mass_displacement",
)
GABOR_FEATURES = ("gabor_mean", "gabor_std")

#: The biological feature set (shape + intensity). Spatial is excluded/unused; gabor is dropped.
BIOLOGICAL_FEATURES = SHAPE_FEATURES + INTENSITY_FEATURES

FEATURE_GROUP = {
    **{f: "shape" for f in SHAPE_FEATURES},
    **{f: "intensity" for f in INTENSITY_FEATURES},
    **{f: "spatial" for f in SPATIAL_FEATURES},
    **{f: "gabor" for f in GABOR_FEATURES},
}

#: Minimum cell count for a statistic to be trusted (matches the n_DMSO<30 low-confidence
#: threshold used elsewhere in feature_to_mcherry).
N_FLOOR = 30


def _ti(timepoint) -> int:
    s = str(timepoint)
    return int(s) if s.isdigit() else -1


def _resolve_features(columns: Sequence[str], requested: Sequence[str]) -> list:
    present = [f for f in requested if f in set(columns)]
    missing = [f for f in requested if f not in set(columns)]
    if missing:
        logger.warning("requested features absent from data: %s", missing)
    return present


def load_features(
    features_dir: Union[str, Path],
    features: Sequence[str] = BIOLOGICAL_FEATURES,
) -> pd.DataFrame:
    """Concatenate the per-image incarta feature CSVs, keeping keys + requested features."""
    files = sorted(glob.glob(str(Path(features_dir) / "*.csv")))
    if not files:
        raise FileNotFoundError(f"no *.csv under {features_dir}")
    usecols = set(KEYS) | set(features)
    frames = [pd.read_csv(f, usecols=lambda c: c in usecols) for f in files]
    df = pd.concat(frames, ignore_index=True)
    for k in ("cell_id", "sample_id", "timepoint", "z_index"):
        if k in df.columns:
            df[k] = df[k].astype(str)
    df["ti"] = df["timepoint"].map(_ti)
    return df


def collapse_to_cell(df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    """One row per (sample_id, timepoint, cell_id): median of each feature over z-slices.

    Delegates to the canonical guarded ``collapse_slices_to_cells`` (which RAISES if any key
    or requested value column is absent — no silent slice-weighting or shrunken feature set).
    Keys are coerced to ``str`` first so behaviour is identical whether called on
    ``load_features`` output or a raw DataFrame.
    """
    feats = list(features)
    df = df.assign(**{
        k: df[k].astype(str) for k in ("sample_id", "timepoint", "cell_id") if k in df.columns
    })
    cells = collapse_slices_to_cells(df, feats)
    cells["ti"] = cells["timepoint"].map(_ti)
    return cells


def _annotate(df: pd.DataFrame, dmso_well: Optional[str], layout: Optional[Mapping[str, Any]]) -> pd.DataFrame:
    """Return a copy of ``df`` with ``is_dmso``/``drug`` columns added (no in-place mutation)."""
    df = df.copy()
    if dmso_well is not None:
        df["is_dmso"] = df["sample_id"].str.upper() == str(dmso_well).upper()
    if layout is not None:
        def _drug(well: str) -> Optional[str]:
            m = _WELL_RE.match(str(well))
            if not m:
                return None
            try:
                ann = get_well_annotation(m.group(1).upper(), int(m.group(2)), layout)
            except Exception:
                return None
            return ann.get("drug") or ann.get("content")

        drug_map = {w: _drug(w) for w in df["sample_id"].unique()}
        df["drug"] = df["sample_id"].map(drug_map)
    return df


def grouped_feature_stats(
    cells: pd.DataFrame,
    features: Sequence[str] = BIOLOGICAL_FEATURES,
    with_mad: bool = False,
) -> pd.DataFrame:
    """Long-form per (sample_id, timepoint, ti, feature): median, q25, q75, n_cells, iqr (+ mad).

    Shared reduction for compute_trajectories (#1) and compute_heterogeneity (#5) so the
    per-(well,timepoint,feature) IQR is defined in exactly one place.
    """
    feats = _resolve_features(cells.columns, features)
    long = cells.melt(
        id_vars=["sample_id", "timepoint", "ti"],
        value_vars=list(feats),
        var_name="feature",
        value_name="val",
    ).dropna(subset=["val"])
    aggs = dict(
        median="median",
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75),
        n_cells="count",
    )
    if with_mad:
        aggs["mad"] = lambda x: _scaled_mad(x.to_numpy(float))
    out = long.groupby(["sample_id", "timepoint", "ti", "feature"])["val"].agg(**aggs).reset_index()
    out["iqr"] = out["q75"] - out["q25"]
    return out


def compute_trajectories(
    cells: pd.DataFrame,
    features: Sequence[str] = BIOLOGICAL_FEATURES,
    dmso_well: Optional[str] = None,
    layout: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    """Per (well, timepoint, feature): median + IQR across cells (long form)."""
    traj = grouped_feature_stats(cells, features)
    traj["group"] = traj["feature"].map(FEATURE_GROUP)
    traj = _annotate(traj, dmso_well, layout)
    return traj.sort_values(["feature", "sample_id", "ti"]).reset_index(drop=True)


def compute_drift(
    traj: pd.DataFrame,
    dmso_n_timepoints_pre_cross: Optional[int] = None,
) -> pd.DataFrame:
    """Per (well, feature) drift over the full time-course.

    Spearman(median vs ti), Theil-Sen slope, and a normalized early→late median shift.
    ``estimable`` is the experiment-level feasibility verdict (from the DMSO well's
    pre-collapse timepoint count) — biological reads should be restricted to it.
    """
    verdict = classify_estimability(dmso_n_timepoints_pre_cross)
    rows = []
    for (well, feat), g in traj.groupby(["sample_id", "feature"]):
        g = g.sort_values("ti")
        ti = g["ti"].to_numpy(float)
        med = g["median"].to_numpy(float)
        n = len(g)
        rho = pval = slope = np.nan
        if n >= 3 and np.nanstd(ti) > 0:
            if np.nanstd(med) > 0:
                rho, pval = spearmanr(ti, med)
                slope = float(theilslopes(med, ti)[0])
            else:
                # confidently flat over time — 0 drift, distinct from NaN (n<3 / undefined)
                rho, pval, slope = 0.0, 1.0, 0.0
        k = max(1, n // 3)
        early = float(np.nanmedian(med[:k]))
        late = float(np.nanmedian(med[-k:]))
        scale = float(np.nanmedian(g["iqr"].to_numpy(float)))
        norm_shift = (late - early) / scale if scale and np.isfinite(scale) and scale > 0 else np.nan
        rows.append(dict(
            sample_id=well, feature=feat, group=FEATURE_GROUP.get(feat),
            spearman_rho=rho, spearman_p=pval, theil_slope=slope,
            early_median=early, late_median=late, norm_shift=norm_shift,
            n_timepoints=n, estimable=verdict,
        ))
    return pd.DataFrame(rows).sort_values(["feature", "sample_id"]).reset_index(drop=True)


def compute_divergence_from_dmso(
    cells: pd.DataFrame,
    dmso_well: str,
    features: Sequence[str] = BIOLOGICAL_FEATURES,
    n_floor: int = N_FLOOR,
    layout: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    """Per (drug well, feature, timepoint) distance from the DMSO distribution.

    KS statistic (primary when DMSO ``n < n_floor`` or MAD≈0) + the MAD-standardized median
    shift ``(median_well − median_DMSO)/MAD_DMSO`` (effect size), each row carrying per-feature
    n/MAD validity and a ``low_confidence`` flag. A ``feature='__overall__'`` row per (well,
    timepoint) holds mean |std_shift| across confident biological features.
    """
    feats = _resolve_features(cells.columns, features)
    dmso_up = str(dmso_well).upper()
    is_dmso = cells["sample_id"].str.upper() == dmso_up
    dmso = cells[is_dmso]
    drug = cells[~is_dmso]

    # Iterate the UNION of timepoints across all wells (not just DMSO's): if the DMSO well
    # has no cells at a timepoint (e.g. an acquisition dropout), the drug-well points must
    # still be emitted — flagged low_confidence with NaN DMSO stats — not silently dropped.
    dmso_by_t = {tv: sub for tv, sub in dmso.groupby("timepoint")}
    all_tp = sorted(set(cells["timepoint"]), key=_ti)

    rows = []
    for tv in all_tp:
        ti = _ti(tv)
        dmso_t = dmso_by_t.get(tv)
        drug_t = drug[drug["timepoint"] == tv]
        for well, well_t in drug_t.groupby("sample_id"):
            per_feat_shift = []
            for f in feats:
                dv = dmso_t[f].to_numpy(float) if dmso_t is not None else np.empty(0)
                dv = dv[np.isfinite(dv)]
                wv = well_t[f].to_numpy(float)
                wv = wv[np.isfinite(wv)]
                n_dmso, n_well = int(dv.size), int(wv.size)
                mad = _scaled_mad(dv) if n_dmso else np.nan
                med_d = float(np.median(dv)) if n_dmso else np.nan
                med_w = float(np.median(wv)) if n_well else np.nan
                std_shift = (med_w - med_d) / mad if mad and np.isfinite(mad) and mad > 0 else np.nan
                if n_dmso >= 2 and n_well >= 2:
                    ks_stat, ks_p = ks_2samp(wv, dv)
                    ks_stat, ks_p = float(ks_stat), float(ks_p)
                    cliffs_delta = _cliffs_delta(wv, dv)  # rank effect size (robust at low n)
                else:
                    ks_stat = ks_p = cliffs_delta = np.nan
                low_conf = (n_dmso < n_floor) or (n_well < n_floor) or not (np.isfinite(mad) and mad > 0)
                rows.append(dict(
                    sample_id=well, timepoint=tv, ti=ti, feature=f, group=FEATURE_GROUP.get(f),
                    ks_stat=ks_stat, ks_p=ks_p, cliffs_delta=cliffs_delta, std_shift=std_shift,
                    median_well=med_w, median_dmso=med_d, mad_dmso=mad,
                    n_well=n_well, n_dmso=n_dmso, low_confidence=bool(low_conf),
                ))
                if (not low_conf) and np.isfinite(std_shift):
                    per_feat_shift.append(abs(std_shift))
            rows.append(dict(
                sample_id=well, timepoint=tv, ti=ti, feature="__overall__", group="overall",
                ks_stat=np.nan, ks_p=np.nan, cliffs_delta=np.nan,
                std_shift=float(np.mean(per_feat_shift)) if per_feat_shift else np.nan,
                median_well=np.nan, median_dmso=np.nan, mad_dmso=np.nan,
                n_well=len(well_t), n_dmso=(len(dmso_t) if dmso_t is not None else 0),
                low_confidence=len(per_feat_shift) == 0,
            ))
    out = pd.DataFrame(rows)
    if layout is not None and len(out):
        out = _annotate(out, dmso_well=None, layout=layout)
    return out.sort_values(["sample_id", "feature", "ti"]).reset_index(drop=True)
