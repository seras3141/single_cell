#!/usr/bin/env python
"""CLI: feature variation over time (Phase 1) for ONE experiment — mCherry-free.

Loads incarta features, collapses per-z to per-cell, and writes trajectories, drift, and
divergence-from-DMSO tables + plots. The drift `estimable` verdict comes from the DMSO well's
pre-collapse timepoint count read from the shared cell-population summary CSV.

See docs/feature_analysis/plan_feature_variation_over_time.md.
"""
import argparse
import logging
from pathlib import Path

import pandas as pd

from src.dataset_analysis.layout import load_plate_layout
from src.feature_analysis import (
    BIOLOGICAL_FEATURES,
    collapse_to_cell,
    compute_confluence_onset,
    compute_divergence_from_dmso,
    compute_drift,
    compute_heterogeneity,
    compute_heterogeneity_trend,
    compute_integrity_flags,
    compute_trajectories,
    load_features,
    plot_overall_divergence,
    plot_trajectories_grid,
)
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def _dmso_pre_cross(summary_csv: Path, label: str, dmso_well: str):
    """DMSO well's n_timepoints_pre_cross for this experiment (None if unavailable)."""
    if not summary_csv.exists():
        logger.warning("summary CSV %s missing — drift estimability -> None", summary_csv)
        return None
    s = pd.read_csv(summary_csv)
    m = s["well"].astype(str).str.upper() == str(dmso_well).upper()
    if "experiment" in s.columns:
        m = m & s["experiment"].astype(str).str.contains(label, case=False, na=False, regex=False)
    hit = s[m]
    if not len(hit) and "is_dmso" in s.columns:  # fall back to the flagged DMSO row
        hit = s[s["is_dmso"].astype(bool) & s["experiment"].astype(str).str.contains(label, case=False, na=False, regex=False)]
    if not len(hit) or "n_timepoints_pre_cross" not in hit.columns:
        return None
    val = hit.iloc[0]["n_timepoints_pre_cross"]
    return None if pd.isna(val) else int(val)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment-dir", required=True)
    ap.add_argument("--model", default="cellpose_sam")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dmso-well", required=True, help="e.g. N11 (HD/SA) or M11 (Ew2)")
    ap.add_argument("--layout", default="config/MF5v1_plate_layout.json")
    ap.add_argument("--summary-csv",
                    default="results/dataset_analysis/all_experiments_cell_population_summary.csv")
    ap.add_argument("--cell-population-csv", default=None,
                    help="per-experiment cell_population.csv (for #4 coverage onset); "
                         "default results/dataset_analysis/<label>/cell_population/cell_population.csv")
    args = ap.parse_args()

    setup_logging()
    exp = Path(args.experiment_dir)
    label = exp.name.split(" ")[0]
    feat_dir = exp / "inference_tracked" / args.model / "features_incarta" / "split_data"
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    layout = load_plate_layout(args.layout) if Path(args.layout).exists() else None

    logger.info("[%s] loading features from %s", label, feat_dir)
    df = load_features(feat_dir, features=BIOLOGICAL_FEATURES)
    cells = collapse_to_cell(df, BIOLOGICAL_FEATURES)
    logger.info("[%s] %d rows -> %d per-cell observations, %d wells",
                label, len(df), len(cells), cells["sample_id"].nunique())

    # --- trajectories ---
    traj = compute_trajectories(cells, BIOLOGICAL_FEATURES, dmso_well=args.dmso_well, layout=layout)
    traj.to_csv(out / "feature_trajectories.csv", index=False)

    # --- drift (with feasibility verdict) ---
    n_pre = _dmso_pre_cross(Path(args.summary_csv), label, args.dmso_well)
    drift = compute_drift(traj, dmso_n_timepoints_pre_cross=n_pre)
    drift.to_csv(out / "feature_drift.csv", index=False)
    logger.info("[%s] drift estimability (DMSO pre-cross n=%s): %s",
                label, n_pre, drift["estimable"].iloc[0] if len(drift) else "n/a")

    # --- divergence from DMSO ---
    div = compute_divergence_from_dmso(cells, args.dmso_well, BIOLOGICAL_FEATURES, layout=layout)
    div.to_csv(out / "divergence_from_dmso.csv", index=False)

    # --- plots ---
    plot_trajectories_grid(traj, out / "trajectories_grid.png", dmso_well=args.dmso_well,
                           title=f"{label} — feature trajectories (median vs t)")
    plot_overall_divergence(div, out / "divergence_overall.png",
                            title=f"{label} — divergence from DMSO over time")
    logger.info("[%s] wrote Phase-1 CSVs + PNGs -> %s", label, out)

    # --- Phase 2: heterogeneity (#5), integrity (#6), confluence-onset (#4) ---
    het = compute_heterogeneity(cells, BIOLOGICAL_FEATURES)
    het.to_csv(out / "heterogeneity.csv", index=False)
    compute_heterogeneity_trend(het, dmso_n_timepoints_pre_cross=n_pre).to_csv(
        out / "heterogeneity_trend.csv", index=False)

    integ = compute_integrity_flags(cells, BIOLOGICAL_FEATURES)
    integ.to_csv(out / "integrity_flags.csv", index=False)
    n_flag = int(integ["flagged"].sum()) if len(integ) else 0
    logger.info("[%s] integrity: %d flagged of %d adjacent-timepoint pairs (expect ~0 on clean data)",
                label, n_flag, len(integ))

    cp_path = Path(args.cell_population_csv) if args.cell_population_csv else Path(
        f"results/dataset_analysis/{label}/cell_population/cell_population.csv")
    if cp_path.exists() and Path(args.summary_csv).exists():
        onset = compute_confluence_onset(
            traj, pd.read_csv(cp_path), pd.read_csv(args.summary_csv),
            experiment_label=label, dmso_well=args.dmso_well, features=BIOLOGICAL_FEATURES)
        onset.to_csv(out / "confluence_onset.csv", index=False)
        logger.info("[%s] confluence onset written (%d wells)", label, len(onset))
    else:
        logger.warning("[%s] confluence onset SKIPPED — missing %s or %s",
                       label, cp_path, args.summary_csv)
    logger.info("[%s] wrote Phase-2 CSVs -> %s", label, out)


if __name__ == "__main__":
    main()
