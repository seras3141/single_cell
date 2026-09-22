#!/usr/bin/env python
"""CLI: Steps 3 and 4 of the z0 projection check.

Step 3 overlays each well's count-over-time from the z0 projection against the published
tracked tree, every curve normalised to its own peak (absolute counts are not comparable
between trees — plan §6.4).

Step 4 localises where detections are lost. For the vehicle wells it also measures the
raw optical slices z1..z20 and adds the *best* slice per timepoint as a third curve.
If the best raw slice holds up where the projection collapses, the failure is specific
to the projection; if it collapses too, raw per-slice segmentation is losing the cells
and tracking is not the culprit.

The z1..z20 masks live in the ORIGINAL processed tree; the staged z0 tree holds only the
projection, by design.

See docs/dataset_analysis/plan_z0_projection_count_check.md.
"""
import argparse
import logging
from pathlib import Path

import matplotlib
import pandas as pd

from src.dataset_analysis.z0_collapse import (
    SOURCE_BEST_SLICE,
    SOURCE_TRACKED,
    SOURCE_Z0,
    build_shape_comparison,
    plot_shape_grid,
    reduce_over_z,
)
from src.dataset_analysis.z0_population import (
    compute_multiz_population,
    drop_unmeasurable,
)
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

#: Experiment label -> (processed dir name, DMSO well).
EXPERIMENTS = {
    "HD1509": ("HD1509 MF5V1 0-72h 23-02-26", "N11"),
    "HD1883": ("HD1883 MF5V1 0-72h 20-03-26", "N11"),
    "SA110": ("SA110 MF5V1 0-72h 13-02-26", "N11"),
    "Ew2-1": ("Ew2-1 MF5V1 0-72h 06-03-26", "M11"),
    "Ew2-2": ("Ew2-2 MF5V1 072h 17-04-26", "M11"),
}

SOURCE_TREE = "data/MF5V1_processed Timelapse samples 19.03.2024"
OPTICAL_Z = tuple(range(1, 21))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", default="results/dataset_analysis")
    ap.add_argument("--source-tree", default=SOURCE_TREE)
    ap.add_argument(
        "--output-dir", default="results/dataset_analysis/z0_collapse/shape"
    )
    ap.add_argument("--model", default="cellpose_sam")
    ap.add_argument(
        "--skip-multiz",
        action="store_true",
        help="skip Step 4 (the z1..z20 sweep of the vehicle wells)",
    )
    ap.add_argument("--experiment", action="append", dest="experiments", default=None)
    args = ap.parse_args()

    # The plotting helpers import pyplot lazily, so the entry point can still
    # own the backend here — and this keeps it out of the import block.
    matplotlib.use("Agg")
    setup_logging()
    root = Path(args.results_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    labels = args.experiments or list(EXPERIMENTS)
    unknown = [label for label in labels if label not in EXPERIMENTS]
    if unknown:
        raise ValueError(
            f"unknown experiment(s) {unknown}; valid: {sorted(EXPERIMENTS)}"
        )
    all_comparisons = []
    all_multiz = []

    for label in labels:
        dir_name, dmso_well = EXPERIMENTS[label]
        z0_csv = root / label / "z0_population" / "z0_population.csv"
        tracked_csv = root / label / "cell_population" / "cell_population.csv"
        if not z0_csv.is_file():
            logger.warning("%s: no z0_population.csv — skipped", label)
            continue

        # Width-flagged counts must never be plotted or normalised against
        # (plan 6.2) — the same filter summarize_z0_collapse applies internally.
        z0 = drop_unmeasurable(pd.read_csv(z0_csv))
        sources = {SOURCE_Z0: (z0, "n_objects")}

        if tracked_csv.is_file():
            sources[SOURCE_TRACKED] = (pd.read_csv(tracked_csv), "n_cells")
        else:
            logger.warning("%s: no cell_population.csv — tracked curve omitted", label)

        # Step 4: the raw optical slices, vehicle well only. A failure here must not
        # discard the Step 3 output already accumulated for earlier experiments.
        if not args.skip_multiz:
            masks = (
                Path(args.source_tree) / dir_name / "inference" / args.model / "masks"
            )
            logger.info("%s: sweeping z%s of %s", label, list(OPTICAL_Z), dmso_well)
            try:
                multiz = compute_multiz_population(
                    masks, OPTICAL_Z, wells=[dmso_well], log_every=0
                )
            except FileNotFoundError as exc:
                logger.warning("%s: multi-z sweep skipped (%s)", label, exc)
            else:
                multiz["experiment"] = label
                all_multiz.append(multiz)
                # 2426 of the multi-z masks are uint8, so the wrap risk is
                # concentrated here rather than in the uint16 z0 tree.
                best = reduce_over_z(drop_unmeasurable(multiz))
                best.to_csv(out / f"{label}_best_slice.csv", index=False)
                sources[SOURCE_BEST_SLICE] = (best, "best_slice_n_objects")

        comparison = build_shape_comparison(sources, label)
        all_comparisons.append(comparison)
        plot_shape_grid(
            comparison,
            label,
            out / f"{label}_shape_comparison.png",
            dmso_well=dmso_well,
        )

    if all_comparisons:
        combined = pd.concat(all_comparisons, ignore_index=True)
        combined.to_csv(out / "shape_comparison.csv", index=False)
        logger.info("wrote shape_comparison.csv (%d rows)", len(combined))
    if all_multiz:
        pd.concat(all_multiz, ignore_index=True).to_csv(
            out / "multiz_population.csv", index=False
        )
        logger.info("wrote multiz_population.csv")


if __name__ == "__main__":
    main()
