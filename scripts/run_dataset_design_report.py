#!/usr/bin/env python
"""CLI: assemble the dataset-design assessment (Step 7 parts a, b, c').

Reads Step 1's 45-well summary and each experiment's mCherry targets, then writes:

- ``cutoff_table.csv``          (7a) per-culture pre-collapse keep-window
- ``density_vs_collapse.csv``   (7b) Spearman of peak density vs time-to-collapse
- ``density_vs_collapse.png``   (7b) the scatter, censored wells kept as their own
  marker
- ``drug_effect_two_windows.csv`` (7c') rank-based effect per well, BOTH windows
- ``drug_effect_summary.md``    (7c') the readable roll-up

7d (feature-vs-time figures) comes from Step 5' output; no computation needed here.

Targets come from ``mcherry_metrics/<model>/instance_metrics.csv`` — one CSV per
experiment, so this is cheap; it does NOT load the per-slice feature CSVs.

See docs/feature_to_mcherry/plan_dataset_design_assessment.md, Step 7.

Example::

    SUMMARY=results/dataset_analysis/all_experiments_cell_population_summary.csv
    python scripts/run_dataset_design_report.py --summary "$SUMMARY" \\
        --output-dir docs/feature_to_mcherry/dataset_design_report
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.feature_to_mcherry.data.loaders import load_targets
from src.feature_to_mcherry.dataset_design import (
    WINDOW_FULL,
    WINDOW_PRE,
    build_cutoff_table,
    build_effect_table,
    density_vs_collapse,
)
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

DATA_ROOT = Path(
    "/ictstr01/home/haicu/serena.sritharan/projects/single_cell/data/"
    "MF5V1_processed Timelapse samples 19.03.2024"
)
MODEL = "cellpose_sam"

#: label -> (on-disk experiment dir, DMSO reference well).
EXPERIMENTS: Dict[str, tuple] = {
    "Ew2-1": ("Ew2-1 MF5V1 0-72h 06-03-26", "M11"),
    "Ew2-2": ("Ew2-2 MF5V1 072h 17-04-26", "M11"),
    "HD1509": ("HD1509 MF5V1 0-72h 23-02-26", "N11"),
    "HD1883": ("HD1883 MF5V1 0-72h 20-03-26", "N11"),
    "SA110": ("SA110 MF5V1 0-72h 13-02-26", "N11"),
}

TARGET_COLUMNS = ["percentile_75", "percentile_90", "percentile_95"]

#: Last frame index in the assay. Right-censored wells are drawn beyond this, not merely
#: beyond the largest observed crossing -- at ti=191 that would place them ~214, reading
#: as "collapsed slightly later" rather than "never collapsed".
ASSAY_END_TI = 351

#: The target reported in the markdown roll-up; the CSV carries all of TARGET_COLUMNS.
HEADLINE_TARGET = "percentile_90"


def _plot_density_vs_collapse(summary: pd.DataFrame, out_png: Path) -> None:
    """Peak density vs time-to-collapse, one point per well.

    Non-collapsing wells are drawn at the top of the axis with their own marker rather
    than dropped: they are right-censored (`t_cross` past the timecourse), and omitting
    them would make the surviving wells look like the whole story.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))
    palette = plt.get_cmap("tab10")
    experiments = sorted(summary["experiment"].unique())
    markers = {
        "Navitoclax": "o",
        "Venetoclax": "s",
        "Selinexor": "^",
        "Doxorubicin": "D",
    }
    # `nan or 0.0` evaluates to nan (NaN is truthy) and `max(nan, x)` returns nan, so
    # an all-censored summary would place every censored marker at NaN. Fall back to 0.
    observed_max = float(summary["t_cross_peak"].max(skipna=True))
    if pd.isna(observed_max):
        observed_max = 0.0
    ceiling = max(observed_max, float(ASSAY_END_TI)) * 1.06

    for i, experiment in enumerate(experiments):
        group = summary[summary["experiment"] == experiment]
        for _, row in group.iterrows():
            crossed = pd.notna(row["t_cross_peak"])
            ax.scatter(
                row["peak_n_cells"],
                row["t_cross_peak"] if crossed else ceiling,
                s=90 if crossed else 130,
                marker=markers.get(str(row["drug"]), "P") if crossed else "X",
                facecolor=palette(i % 10) if crossed else "none",
                edgecolor=palette(i % 10),
                linewidths=1.8,
                alpha=0.85,
                label=experiment if _ == group.index[0] else None,
            )
    ax.axhline(ceiling, color="grey", linestyle=":", linewidth=1)
    ax.text(
        ax.get_xlim()[1],
        ceiling,
        "  never collapsed\n  (right-censored)",
        va="center",
        fontsize=9,
        color="grey",
    )
    ax.set_xlabel("peak distinct cells in the well (seeding-density proxy)")
    ax.set_ylabel("t_cross_peak  (first frame below 50 % of peak)")
    ax.set_title(
        "Does seeding density predict time-to-collapse?  45 wells, 5 cultures\n"
        "marker = drug, colour = culture, hollow X = never collapsed",
        fontsize=11,
    )
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=9, title="culture")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _load_targets(experiment: str) -> pd.DataFrame:
    exp_dir, _ = EXPERIMENTS[experiment]
    csv = DATA_ROOT / exp_dir / "mcherry_metrics" / MODEL / "instance_metrics.csv"
    if not csv.is_file():
        raise FileNotFoundError(f"{experiment}: no target csv at {csv}")
    targets = load_targets(csv, target_columns=TARGET_COLUMNS)
    logger.info("%s: %d target rows", experiment, len(targets))
    return targets


def _summarise_effects(effects: pd.DataFrame) -> str:
    """Readable roll-up of the two-window comparison for the headline target."""
    lines: List[str] = []
    subset = effects[effects["target"] == HEADLINE_TARGET]
    for experiment in sorted(subset["experiment"].unique()):
        exp_rows = subset[subset["experiment"] == experiment]
        lines.append(f"\n### {experiment}\n")
        lines.append(
            "| well | drug | dose | window | Cliff's δ | direction | tp | label |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for _, r in exp_rows.sort_values(["well", "window"]).iterrows():
            delta = "n/a" if pd.isna(r["cliffs_delta"]) else f"{r['cliffs_delta']:+.3f}"
            conc = (
                "" if pd.isna(r["concentration_uM"]) else f"{r['concentration_uM']:g}"
            )
            lines.append(
                f"| {r['well']} | {r['drug']} | {conc} µM | {r['window']} | {delta} | "
                f"{r['direction'] or '-'} | {r['n_timepoints_matched']} | "
                f"{r['label']} |"
            )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--summary",
        default="results/dataset_analysis/all_experiments_cell_population_summary.csv",
    )
    ap.add_argument(
        "--output-dir", default="docs/feature_to_mcherry/dataset_design_report"
    )
    ap.add_argument(
        "--gate-threshold",
        type=float,
        default=0.10,
        help="Step 2's recommended relative gate; pass a negative value to omit it",
    )
    args = ap.parse_args()

    setup_logging()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(args.summary)
    logger.info("read Step 1 summary: %d rows", len(summary))

    gate = None if args.gate_threshold < 0 else args.gate_threshold

    # --- 7a ---------------------------------------------------------------------
    cutoffs = build_cutoff_table(summary, gate_threshold=gate)
    cutoffs.to_csv(out_dir / "cutoff_table.csv", index=False)
    logger.info("7a cutoff table:\n%s", cutoffs.to_string(index=False))

    # --- 7b ---------------------------------------------------------------------
    density = density_vs_collapse(summary)
    density.to_csv(out_dir / "density_vs_collapse.csv", index=False)
    logger.info("7b density vs collapse:\n%s", density.to_string(index=False))
    _plot_density_vs_collapse(summary, out_dir / "density_vs_collapse.png")

    # --- 7c' -------------------------------------------------------------------
    frames = []
    for experiment, (_, dmso_well) in EXPERIMENTS.items():
        targets = _load_targets(experiment)
        for target_column in TARGET_COLUMNS:
            frames.append(
                build_effect_table(
                    targets, summary, experiment, dmso_well, target_column
                )
            )
    effects = pd.concat(frames, ignore_index=True)
    effects.to_csv(out_dir / "drug_effect_two_windows.csv", index=False)

    headline = effects[effects["target"] == HEADLINE_TARGET]
    for window in (WINDOW_PRE, WINDOW_FULL):
        rows = headline[headline["window"] == window]
        counts = rows["label"].value_counts().to_dict()
        logger.info("7c' %s (%s): %s", window, HEADLINE_TARGET, counts)

    (out_dir / "drug_effect_summary.md").write_text(
        "# 7c' — is a drug response visible? Two windows\n\n"
        f"Rank-based (Cliff's δ), timepoint-matched vs each culture's DMSO well, "
        f"target `{HEADLINE_TARGET}`. **Every row is one imaged well "
        "(`n_wells = 1`)** — no within-condition replicates exist, so these are "
        "descriptive for that well and cannot support a condition-level 'no effect' "
        "claim.\n" + _summarise_effects(effects) + "\n"
    )
    logger.info("wrote outputs to %s", out_dir)


if __name__ == "__main__":
    main()
