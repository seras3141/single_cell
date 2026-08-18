#!/usr/bin/env python
"""CLI: pre-collapse estimability table, plus feature-vs-time plots where estimable.

For each culture, decides from Step 1's summary whether a within-window feature-vs-time
trend is estimable at all. Where it is, plots the strongest features against timepoint
inside that culture's pre-collapse window. Where it is not, plots the DMSO population
collapse with ``t_cross`` marked -- showing *why* no trend figure is offered, the
honest figure rather than a trend fitted to three points.

See docs/feature_to_mcherry/plan_fatima_deliverable_pipeline.md, Step 5'.

Example::

    python scripts/run_pre_collapse_figures.py \\
        --summary results/dataset_analysis/all_experiments_summary.csv \\
        --output-dir docs/feature_to_mcherry/figures/feature_vs_time_for_fatima
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.feature_to_mcherry.data.join import build_matrix_with_metadata
from src.feature_to_mcherry.data.loaders import (
    load_features_from_directory,
    load_targets,
)

# Reuse the informativeness module's own trend helper rather than reimplementing it:
# the plan asks for its visual conventions, and a second implementation would drift.
from src.feature_to_mcherry.informativeness.plots import (
    _rolling_trend,
    _subsample_indices,
)
from src.feature_to_mcherry.pre_collapse import (
    DEFAULT_RHO_THRESHOLD,
    VERDICT_NOT_ESTIMABLE,
    build_estimability_table,
    select_informative_features,
    summarise_selection,
    top_features,
    truncate_to_pre_collapse,
    within_window_trend,
)
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

DATA_ROOT = Path(
    "/ictstr01/home/haicu/serena.sritharan/projects/single_cell/data/"
    "MF5V1_processed Timelapse samples 19.03.2024"
)
MODEL = "cellpose_sam"

#: label -> on-disk experiment dir (mirrors slurm/feature_informativeness_qc.sbatch).
EXP_DIR: Dict[str, str] = {
    "Ew2-1": "Ew2-1 MF5V1 0-72h 06-03-26",
    "Ew2-2": "Ew2-2 MF5V1 072h 17-04-26",
    "HD1509": "HD1509 MF5V1 0-72h 23-02-26",
    "HD1883": "HD1883 MF5V1 0-72h 20-03-26",
    "SA110": "SA110 MF5V1 0-72h 13-02-26",
}

TARGET_COLUMNS = ["percentile_75", "percentile_90", "percentile_95"]


def _univariate_path(results_root: Path, experiment: str) -> Path:
    """Canonical per-experiment univariate output.

    Deliberately NOT any ``*_corrupted_pre_uint8fix`` sibling or
    ``feature_to_mcherry_*`` ablation dir -- those are pre-fix or non-canonical runs.
    """
    return (
        results_root
        / "feature_to_mcherry"
        / experiment
        / "morphology_informativeness"
        / "univariate_correlations.csv"
    )


def _plot_feature_vs_time(
    metadata: pd.DataFrame,
    features: pd.DataFrame,
    feature_names: List[str],
    experiment: str,
    n_timepoints: int,
    dmso_t_cross: Optional[float],
    out_png: Path,
    *,
    point_cap: int = 60_000,
    seed: int = 0,
) -> None:
    """Feature value vs timepoint, coloured by well, with a rolling-median trend.

    Each well is truncated at **its own** ``t_cross``, so wells that never collapse
    legitimately extend to the end of the timecourse. The DMSO reference's own crossing
    is drawn as a dashed line and the region beyond it shaded, because that is the point
    past which there is no usable baseline to normalise against — without it the x-range
    reads as though the whole span were inside the pre-collapse window.

    Points are subsampled to ``point_cap`` (deterministically) purely for legibility;
    the rolling-median trend is computed on the **full** data first, so the line is not
    an artefact of the subsample.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(feature_names)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.2 * ncols, 3.8 * nrows), squeeze=False
    )

    times = pd.to_numeric(metadata["timepoint"], errors="coerce").to_numpy(dtype=float)
    wells = metadata["sample_id"].astype(str).to_numpy()
    palette = plt.get_cmap("tab10")
    well_ids = sorted(set(wells))
    rng = np.random.default_rng(seed)

    for index, name in enumerate(feature_names):
        ax = axes[index // ncols][index % ncols]
        values = features[name].to_numpy(dtype=float)
        finite = np.isfinite(times) & np.isfinite(values)

        # Trend on the full data, before any subsampling.
        if finite.sum() >= 5:
            x_trend, y_trend = _rolling_trend(times[finite], values[finite])
        else:
            x_trend = y_trend = None

        for i, well in enumerate(well_ids):
            mask = np.flatnonzero(finite & (wells == well))
            if mask.size == 0:
                continue
            share = max(1, int(point_cap * mask.size / max(1, finite.sum())))
            mask = _subsample_indices(mask, share, rng)
            ax.scatter(
                times[mask],
                values[mask],
                s=5,
                alpha=0.35,
                color=palette(i % 10),
                label=well if index == 0 else None,
                linewidths=0,
            )

        if x_trend is not None:
            ax.plot(x_trend, y_trend, color="black", linewidth=2.2, zorder=6)
        if dmso_t_cross is not None and pd.notna(dmso_t_cross):
            ax.axvline(
                float(dmso_t_cross), color="tab:red", linestyle="--", linewidth=1.6
            )
            ax.axvspan(
                float(dmso_t_cross), float(np.nanmax(times)), color="grey", alpha=0.12
            )
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("timepoint (frame index)")
        ax.grid(alpha=0.25)

    for index in range(n, nrows * ncols):
        axes[index // ncols][index % ncols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels, loc="lower center", ncol=min(9, len(labels)), fontsize=8
        )
    cross_text = (
        "no DMSO crossing"
        if dmso_t_cross is None or pd.isna(dmso_t_cross)
        else f"DMSO reference collapses at t={int(dmso_t_cross)} (red line)"
    )
    fig.suptitle(
        f"{experiment} — features vs time, each well truncated at its own t_cross.\n"
        f"{cross_text}; only {n_timepoints} mCherry timepoints precede it. "
        f"Shaded region has no usable DMSO baseline.",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _plot_collapse_reason(
    cell_population: pd.DataFrame,
    row: pd.Series,
    out_png: Path,
) -> None:
    """DMSO population trajectory with t_cross marked — why no trend figure exists."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dmso = cell_population[cell_population["sample_id"] == row["dmso_well"]]
    dmso = dmso.sort_values("ti")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(dmso["ti"], dmso["n_cells"], marker="o", markersize=4, color="black")
    limit = row["t_cross_peak"]
    n_pre = int(row["n_timepoints_pre_cross"])
    if pd.notna(limit):
        ax.axvline(float(limit), color="tab:red", linestyle="--", linewidth=2)
        ax.axvspan(
            float(dmso["ti"].min()),
            float(limit),
            color="tab:green",
            alpha=0.12,
        )
        ax.annotate(
            f"t_cross = {int(limit)}\nonly {n_pre} usable timepoint(s)",
            xy=(float(limit), float(dmso["n_cells"].max()) * 0.75),
            xytext=(float(limit) * 1.6 + 20, float(dmso["n_cells"].max()) * 0.75),
            arrowprops={"arrowstyle": "->", "color": "tab:red"},
            fontsize=11,
            color="tab:red",
        )
    ax.set_xlabel("timepoint (frame index)")
    ax.set_ylabel("distinct cells in the DMSO well")
    ax.set_title(
        f"{row['experiment']} — DMSO ({row['dmso_well']}) collapses at "
        f"t={int(limit) if pd.notna(limit) else 'never'}; "
        f"no feature-vs-time trend is estimable",
        fontsize=11,
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _load_experiment(experiment: str, id_column: str) -> tuple:
    """Features + targets for one experiment, from the canonical incarta/mCherry
    paths."""
    exp_dir = DATA_ROOT / EXP_DIR[experiment]
    feature_dir = (
        exp_dir / "inference_tracked" / MODEL / "features_incarta" / "split_data"
    )
    target_csv = exp_dir / "mcherry_metrics" / MODEL / "instance_metrics.csv"
    if not feature_dir.is_dir():
        raise FileNotFoundError(f"{experiment}: no feature dir at {feature_dir}")
    if not target_csv.is_file():
        raise FileNotFoundError(f"{experiment}: no target csv at {target_csv}")
    logger.info("%s: loading features from %s", experiment, feature_dir)
    features = load_features_from_directory(feature_dir, id_column=id_column)
    targets = load_targets(target_csv, target_columns=TARGET_COLUMNS)
    logger.info(
        "%s: %d feature rows, %d target rows", experiment, len(features), len(targets)
    )
    return features, targets


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--summary",
        default="results/dataset_analysis/all_experiments_cell_population_summary.csv",
    )
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--cell-population-root", default="results/dataset_analysis")
    ap.add_argument(
        "--output-dir",
        default="docs/feature_to_mcherry/figures/feature_vs_time_for_fatima",
    )
    ap.add_argument("--rho-threshold", type=float, default=DEFAULT_RHO_THRESHOLD)
    ap.add_argument("--max-features", type=int, default=6)
    ap.add_argument("--id-column", default="cell_id")
    args = ap.parse_args()

    setup_logging()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_root = Path(args.results_root)

    summary = pd.read_csv(args.summary)
    estimability = build_estimability_table(summary)
    estimability.to_csv(out_dir / "estimability_table.csv", index=False)
    logger.info("estimability table:\n%s", estimability.to_string(index=False))

    selections: Dict[str, pd.DataFrame] = {}
    for experiment in EXP_DIR:
        path = _univariate_path(results_root, experiment)
        if not path.is_file():
            raise FileNotFoundError(f"{experiment}: no univariate output at {path}")
        selections[experiment] = select_informative_features(
            pd.read_csv(path), rho_threshold=args.rho_threshold
        )
    selection_summary = summarise_selection(selections, args.rho_threshold)
    selection_summary.to_csv(out_dir / "feature_selection_summary.csv", index=False)
    logger.info("feature selection:\n%s", selection_summary.to_string(index=False))

    thin = selection_summary[selection_summary["n_pairs_selected"] == 0]
    if not thin.empty:
        raise ValueError(
            f"no pooled pairs cleared |rho| > {args.rho_threshold} for "
            f"{sorted(thin['experiment'])}; lower the threshold explicitly rather than "
            f"shipping an empty selection"
        )

    for _, row in estimability.iterrows():
        experiment = row["experiment"]
        exp_dir = out_dir / experiment
        exp_dir.mkdir(parents=True, exist_ok=True)

        if row["verdict"] == VERDICT_NOT_ESTIMABLE:
            pop = pd.read_csv(
                Path(args.cell_population_root)
                / experiment
                / "cell_population"
                / "cell_population.csv"
            )
            _plot_collapse_reason(pop, row, exp_dir / "why_no_trend.png")
            logger.info("%s: not estimable — wrote why_no_trend.png", experiment)
            continue

        names = top_features(selections[experiment], limit=args.max_features)
        features, targets = _load_experiment(experiment, args.id_column)
        X, _, _, feature_names, metadata = build_matrix_with_metadata(
            features, targets, target_columns=TARGET_COLUMNS, group_by="sample_id"
        )
        usable = [n for n in names if n in set(feature_names)]
        if not usable:
            raise ValueError(
                f"{experiment}: none of the selected features {names} are in the "
                f"joined matrix ({len(feature_names)} columns) — selection and feature "
                f"table have diverged"
            )

        # X is already the joined feature matrix, row-aligned with `metadata`. Use it
        # directly rather than re-merging `features` onto `metadata`: the pipeline has
        # done the join once, and a second one would risk diverging from its key
        # handling (metadata's CELL_KEY columns are normalised to str, the raw feature
        # frame's are not, so a naive merge does not even typecheck).
        feature_frame = pd.DataFrame(X, columns=feature_names)
        keep = truncate_to_pre_collapse(metadata, summary, experiment)
        mask = keep.to_numpy()
        _plot_feature_vs_time(
            metadata.loc[mask].reset_index(drop=True),
            feature_frame.loc[mask].reset_index(drop=True),
            usable,
            experiment,
            int(row["n_timepoints_pre_cross"]),
            row["t_cross_peak"],
            exp_dir / "features_vs_time.png",
        )
        trend = within_window_trend(
            metadata.loc[mask].reset_index(drop=True),
            feature_frame.loc[mask].reset_index(drop=True),
            usable,
            row["t_cross_peak"],
        )
        trend.insert(0, "experiment", experiment)
        trend.to_csv(exp_dir / "within_window_trend.csv", index=False)
        logger.info(
            "%s: %s — plotted %d features on %d pre-collapse rows",
            experiment,
            row["verdict"],
            len(usable),
            int(keep.sum()),
        )
        logger.info(
            "%s within-window trend:\n%s", experiment, trend.to_string(index=False)
        )

    logger.info("wrote outputs to %s", out_dir)


if __name__ == "__main__":
    main()
