#!/usr/bin/env python
"""CLI: how much data each candidate confidence-gate threshold would discard.

Reads the five per-experiment ``cell_population.csv`` files plus the Step 1 summary
(for each well's peak), applies the candidate gate thresholds, and writes the per-well
survival table, the aggregate distribution, the DMSO-reference-degraded breakdown, and
plots.

Read-only with respect to ``results/`` and every config/source file: outputs go to the
scratch directory only.

See docs/feature_to_mcherry/plan_fatima_deliverable_pipeline.md, Step 2.

Example::

    SUMMARY=results/dataset_analysis/all_experiments_cell_population_summary.csv
    python scripts/run_gate_survival.py \\
        --results-root results/dataset_analysis \\
        --summary "$SUMMARY" \\
        --output-dir docs/_phase6_scratch/dmso_gate_threshold_survival_table
"""

import argparse
import logging
from pathlib import Path
from typing import Dict

import pandas as pd

from src.dataset_analysis.gate_survival import (
    ABSOLUTE_FLOOR_LABEL,
    DEFAULT_ABSOLUTE_FLOOR,
    DEFAULT_FRACTIONS,
    aggregate_by_threshold,
    build_flag_frame,
    crosscheck_against_t_cross,
    summarise_survival,
)
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

#: DMSO reference well per experiment.
DMSO_WELLS: Dict[str, str] = {
    "Ew2-1": "M11",
    "Ew2-2": "M11",
    "HD1509": "N11",
    "HD1883": "N11",
    "SA110": "N11",
}


def _plot_survival(survival: pd.DataFrame, out_png: Path) -> None:
    """Per-threshold spread of 'fraction of timepoints kept' across all wells."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = [f"rel_{f:g}" for f in DEFAULT_FRACTIONS] + [ABSOLUTE_FLOOR_LABEL]
    order = [t for t in order if t in set(survival["threshold"])]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, column, title in (
        (axes[0], "frac_kept_own", "Own-well condition only"),
        (axes[1], "frac_kept_any", "Including DMSO-reference condition"),
    ):
        data = [
            survival.loc[survival["threshold"] == t, column].to_numpy() for t in order
        ]
        ax.boxplot(data, labels=order, showmeans=True)
        for i, values in enumerate(data, start=1):
            ax.scatter(
                [i] * len(values), values, s=12, alpha=0.45, zorder=3, color="tab:blue"
            )
        ax.set_title(title)
        ax.set_xlabel("threshold (fraction of the well's peak)")
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("fraction of timepoints kept")
    fig.suptitle("Gate-threshold survival across all wells (reference = peak)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _plot_dmso_cost(aggregate: pd.DataFrame, out_png: Path) -> None:
    """Share of all well-timepoints lost purely to the DMSO-reference condition."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(aggregate["threshold"], aggregate["pct_lost_to_dmso_condition"])
    ax.set_ylabel("% of all well-timepoints flagged ONLY by the DMSO condition")
    ax.set_xlabel("threshold")
    ax.set_title("Marginal cost of the DMSO-reference-degraded condition")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", default="results/dataset_analysis")
    ap.add_argument(
        "--summary",
        default="results/dataset_analysis/all_experiments_cell_population_summary.csv",
    )
    ap.add_argument(
        "--output-dir",
        default="docs/_phase6_scratch/dmso_gate_threshold_survival_table",
    )
    ap.add_argument("--absolute-floor", type=int, default=DEFAULT_ABSOLUTE_FLOOR)
    args = ap.parse_args()

    setup_logging()
    results_root = Path(args.results_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(args.summary)
    logger.info("read Step 1 summary: %d rows", len(summary))

    survival_frames = []
    flag_frames = []
    for experiment, dmso_well in DMSO_WELLS.items():
        csv_path = results_root / experiment / "cell_population" / "cell_population.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(
                f"{experiment}: no cell_population.csv at {csv_path}"
            )
        table = pd.read_csv(csv_path)
        peaks = (
            summary[summary["experiment"] == experiment]
            .set_index("well")["peak_n_cells"]
            .to_dict()
        )
        flags = build_flag_frame(
            table,
            peaks,
            dmso_well=dmso_well,
            absolute_floor=args.absolute_floor,
        )
        flags.insert(0, "experiment", experiment)
        flag_frames.append(flags)

        well_survival = summarise_survival(flags)
        well_survival.insert(0, "experiment", experiment)
        survival_frames.append(well_survival)
        logger.info("%s: %d wells flagged across thresholds", experiment, len(peaks))

    all_flags = pd.concat(flag_frames, ignore_index=True)
    survival = pd.concat(survival_frames, ignore_index=True)
    aggregate = aggregate_by_threshold(survival)

    survival.to_csv(out_dir / "per_well_survival.csv", index=False)
    aggregate.to_csv(out_dir / "aggregate_by_threshold.csv", index=False)
    all_flags.to_csv(out_dir / "per_timepoint_flags.csv.gz", index=False)

    crosscheck = pd.concat(
        [
            crosscheck_against_t_cross(
                survival[survival["experiment"] == experiment],
                summary[summary["experiment"] == experiment],
            ).assign(experiment=experiment)
            for experiment in DMSO_WELLS
        ],
        ignore_index=True,
    )
    crosscheck.to_csv(out_dir / "t_cross_crosscheck.csv", index=False)
    n_disagree = int((~crosscheck["agrees"]).sum())
    if n_disagree:
        logger.error(
            "%d/%d wells disagree with Step 1's t_cross_peak at the 0.5 gate — the "
            "survival table and the Step 1 summary have drifted apart",
            n_disagree,
            len(crosscheck),
        )
    else:
        logger.info(
            "crosscheck OK: the 0.5 gate reproduces t_cross_peak for all %d wells",
            len(crosscheck),
        )

    _plot_survival(survival, out_dir / "survival_by_threshold.png")
    _plot_dmso_cost(aggregate, out_dir / "dmso_condition_cost.png")

    with pd.option_context("display.width", 220, "display.max_columns", 40):
        logger.info("aggregate by threshold:\n%s", aggregate.to_string(index=False))

    logger.info("wrote outputs to %s", out_dir)


if __name__ == "__main__":
    main()
