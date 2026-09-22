#!/usr/bin/env python
"""CLI: z0 collapse metrics, per-well signatures, and per-drug confluence figures.

Step 2 of the z0 projection check. Recomputes t_cross on the z0 object counts through
the shipped collapse metric, reduces every well to a merging-vs-loss signature, compares
against the published tracked summary, and writes one dose-panel figure per drug.

Absolute counts are NOT comparable between the two trees (z0 projection objects from raw
inference vs 3D-linked tracked identities); the comparison reports the normalised shape
and the t_cross difference, not a difference of counts.

See docs/dataset_analysis/plan_z0_projection_count_check.md.
"""
import argparse
import logging
from pathlib import Path

import matplotlib
import pandas as pd

from src.dataset_analysis.collapse_summary import assert_well_composition
from src.dataset_analysis.layout import load_plate_layout
from src.dataset_analysis.z0_population import drop_unmeasurable
from src.dataset_analysis.z0_collapse import (
    summarize_z0_collapse,
    summarize_z0_signature,
    write_drug_dose_figures,
)
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

#: Experiment label -> DMSO well. Matches slurm/cell_population.sbatch.
DMSO_WELLS = {
    "HD1509": "N11",
    "HD1883": "N11",
    "SA110": "N11",
    "Ew2-1": "M11",
    "Ew2-2": "M11",
}

TRACKED_SUMMARY = "results/dataset_analysis/all_experiments_cell_population_summary.csv"


def _compare_to_tracked(z0: pd.DataFrame, tracked_csv: Path) -> pd.DataFrame:
    """Side-by-side t_cross and end-percentage, z0 vs the published tracked summary."""
    tracked = pd.read_csv(tracked_csv)
    keep = [
        "experiment",
        "well",
        "t_cross_peak",
        "end_pct_of_peak",
        "n_timepoints_pre_cross",
        "collapse_shape",
        "distinct_collapses",
    ]
    merged = z0[keep].merge(
        tracked[keep],
        on=["experiment", "well"],
        how="outer",
        suffixes=("_z0", "_tracked"),
    )
    merged["t_cross_delta"] = merged["t_cross_peak_z0"] - merged["t_cross_peak_tracked"]
    merged["collapses_in_both"] = merged["distinct_collapses_z0"].fillna(False).astype(
        bool
    ) & merged["distinct_collapses_tracked"].fillna(False).astype(bool)
    return merged.sort_values(["experiment", "well"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", default="results/dataset_analysis")
    ap.add_argument("--output-dir", default="results/dataset_analysis/z0_collapse")
    ap.add_argument("--layout", default="config/MF5v1_plate_layout.json")
    ap.add_argument("--tracked-summary", default=TRACKED_SUMMARY)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument(
        "--skip-composition-check",
        action="store_true",
        help="skip the plate-layout guard (see collapse_summary)",
    )
    args = ap.parse_args()

    # The plotting helpers import pyplot lazily, so the entry point can still
    # own the backend here — and this keeps it out of the import block.
    matplotlib.use("Agg")
    setup_logging()
    root = Path(args.results_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    layout_path = Path(args.layout)
    if layout_path.exists():
        layout = load_plate_layout(layout_path)
    else:
        layout = None
        logger.warning(
            "layout %s not found — wells will not be dose-annotated, so the "
            "plate-composition guard is skipped too",
            layout_path,
        )

    csvs = {}
    for label in DMSO_WELLS:
        path = root / label / "z0_population" / "z0_population.csv"
        if path.is_file():
            csvs[label] = path
        else:
            logger.warning("%s: no z0_population.csv at %s — skipped", label, path)
    if not csvs:
        raise FileNotFoundError(f"no z0_population.csv found under {root}")

    summary = summarize_z0_collapse(csvs, layout=layout, dmso_wells=DMSO_WELLS)
    # The guard checks layout-derived dose columns, so it can only run when a
    # layout was loaded — without one it would fail for the wrong reason.
    if layout is not None and not args.skip_composition_check:
        assert_well_composition(summary)
    summary.to_csv(out / "z0_collapse_summary.csv", index=False)
    logger.info("wrote z0_collapse_summary.csv (%d wells)", len(summary))

    signatures = []
    for label, path in csvs.items():
        # Same filter summarize_z0_collapse applies internally: without it the two
        # CSVs would disagree about a width-flagged well (plan 6.2).
        population = drop_unmeasurable(pd.read_csv(path))
        signature = summarize_z0_signature(population, label, collapse_summary=summary)
        signatures.append(signature)

        if not args.no_figures:
            wells = summary[summary["experiment"] == label]
            written = write_drug_dose_figures(
                population,
                wells,
                label,
                out / "figures" / label,
                dmso_well=DMSO_WELLS[label],
            )
            logger.info("%s: %d drug figures", label, len(written))

    all_signatures = pd.concat(signatures, ignore_index=True)
    all_signatures.to_csv(out / "z0_signature.csv", index=False)
    logger.info("wrote z0_signature.csv (%d wells)", len(all_signatures))
    logger.info(
        "signature counts: %s", all_signatures["signature"].value_counts().to_dict()
    )

    tracked_csv = Path(args.tracked_summary)
    if tracked_csv.is_file():
        comparison = _compare_to_tracked(summary, tracked_csv)
        comparison.to_csv(out / "z0_vs_tracked.csv", index=False)
        logger.info("wrote z0_vs_tracked.csv (%d wells)", len(comparison))
    else:
        logger.warning("tracked summary %s not found — comparison skipped", tracked_csv)


if __name__ == "__main__":
    main()
