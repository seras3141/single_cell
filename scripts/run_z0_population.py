#!/usr/bin/env python
"""CLI: per-(well, timepoint) population metrics from the z0 projection masks.

Measures the raw inference masks — upstream of 3D combine, blur filtering and tracking —
and writes count, coverage and mean-object-size per (well, timepoint), plus the
integer-width audit. Curves are rendered with cell_population's plotter so the z0 and
per-slice figures are visually comparable.

Counts here are NOT comparable in absolute terms to cell_population's n_cells: those are
3D-linked tracked identities, these are projection objects from raw inference. Only the
normalised shape over time may be compared.

See docs/dataset_analysis/plan_z0_projection_count_check.md.
"""
import argparse
import logging
from pathlib import Path

from src.dataset_analysis.cell_population import plot_population
from src.dataset_analysis.layout import load_plate_layout
from src.dataset_analysis.z0_population import (
    compute_z0_population,
    drop_unmeasurable,
    saturation_report,
)
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

PLOT_METRICS = ("n_objects", "coverage_fraction", "mean_object_area_fraction")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--experiment-dir",
        required=True,
        help="z0-tree experiment dir (contains inference/<model>/masks)",
    )
    ap.add_argument("--model", default="cellpose_sam")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dmso-well", default=None, help="e.g. N11 (HD/SA) or M11 (Ew2)")
    ap.add_argument("--layout", default="config/MF5v1_plate_layout.json")
    ap.add_argument("--label", default=None, help="title prefix; default: dir name")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    setup_logging()
    exp = Path(args.experiment_dir)
    masks = exp / "inference" / args.model / "masks"
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    label = args.label or exp.name.split()[0]

    layout_path = Path(args.layout)
    if layout_path.exists():
        layout = load_plate_layout(layout_path)
    else:
        layout = None
        logger.warning(
            "layout %s not found (cwd=%s) — wells will not be drug-annotated",
            layout_path,
            Path.cwd(),
        )
    logger.info("z0 population: %s (dmso=%s)", masks, args.dmso_well)

    df = compute_z0_population(masks, layout=layout, dmso_well=args.dmso_well)
    csv_path = out / "z0_population.csv"
    df.to_csv(csv_path, index=False)
    logger.info(
        "wrote %s (%d rows, %d wells)", csv_path, len(df), df["sample_id"].nunique()
    )

    flagged = saturation_report(df)
    flagged.to_csv(out / "z0_saturation_audit.csv", index=False)
    if not flagged.empty:
        logger.warning(
            "%d frames sit on an integer-width boundary — see z0_saturation_audit.csv",
            len(flagged),
        )

    if not args.no_plots:
        # The CSV keeps every row for audit, but a width-flagged count must not be
        # plotted as a measurement (plan 6.2).
        plottable = drop_unmeasurable(df)
        for metric in PLOT_METRICS:
            plot_population(
                plottable,
                metric=metric,
                out_png=out / f"z0_{metric}.png",
                title=f"{label} — z0 projection: {metric}",
                dmso_well=args.dmso_well,
            )
        logger.info("wrote %d figures to %s", len(PLOT_METRICS), out)

    logger.info(
        "%s: %d rows, %d wells, %d width-flagged, %d unreadable",
        label,
        len(df),
        df["sample_id"].nunique(),
        len(flagged),
        int(df["error"].notna().sum()),
    )


if __name__ == "__main__":
    main()
