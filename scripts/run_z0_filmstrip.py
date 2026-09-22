#!/usr/bin/env python
"""CLI: BF + mask filmstrips across timepoints, from the staged z0 tree.

Renders one strip per well: brightfield on the top row, brightfield + segmentation mask
below, at each requested timepoint. This is the direct test the mask-derived numbers
cannot perform — whether cells are visibly present but unlabelled (detection loss) or
genuinely fused into unresolvable masses (aggregation).

The acquisition grid steps by 10 (1, 11, 21, ... 351), so round requests like t=50 are
snapped to the nearest acquired frame and the substitution is logged.

See docs/dataset_analysis/plan_z0_projection_count_check.md.
"""
import argparse
import logging
from pathlib import Path

import pandas as pd

from src.utils.logging_utils import setup_logging
from src.dataset_analysis.z0_population import drop_unmeasurable
from src.visualize.filmstrip import (
    DEFAULT_DPI,
    DEFAULT_PANEL_INCHES,
    DEFAULT_TIMEPOINTS,
    EARLY_TIMEPOINTS,
    MASK_STYLE_FILL,
    MASK_STYLE_OUTLINE,
    available_wells,
    build_filmstrip,
    condition_label,
    resolve_frames,
)

logger = logging.getLogger(__name__)

PRESETS = {
    "default": DEFAULT_TIMEPOINTS,
    "early": EARLY_TIMEPOINTS,
}

#: Sensible base colormap per channel. mCherry is fluorescence, so an intensity ramp
#: reads better than gray; brightfield is not an intensity measurement.
CHANNEL_CMAP = {"BF": "gray", "mCherry": "inferno", "FlipGFP": "viridis"}


def _parse_timepoints(value: str):
    if value in PRESETS:
        return PRESETS[value]
    return [int(part) for part in value.split(",") if part.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--experiment-dir",
        required=True,
        help="z0-tree experiment dir (split_data/ + inference masks)",
    )
    ap.add_argument(
        "--well",
        action="append",
        dest="wells",
        default=None,
        help="well id; repeatable. Default: every well in the tree.",
    )
    ap.add_argument("--model", default="cellpose_sam")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument(
        "--timepoints",
        default="default",
        help="'default' (1,51,101,151,201), 'early' (1,11,21,31,41), "
        "or a comma-separated list",
    )
    ap.add_argument(
        "--crop",
        type=float,
        default=1.0,
        help="centred crop fraction. NOTE: the count/coverage annotations "
        "are measured on the WHOLE field, so a crop below 1.0 shows "
        "only crop^2 of the objects the title counts (0.35 -> 12%%). "
        "Whole-field strips need --dpi ~340 to stay legible.",
    )
    ap.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help="output resolution; panel_inches * dpi = pixels per panel",
    )
    ap.add_argument(
        "--panel-inches",
        type=float,
        default=DEFAULT_PANEL_INCHES,
        help="width allocated to each column",
    )
    ap.add_argument(
        "--format",
        default="png",
        choices=["png", "jpg"],
        help="jpg keeps whole-field strips to a few MB (lossy)",
    )
    ap.add_argument(
        "--quality",
        type=int,
        default=None,
        help="JPEG quality 1-95 (default 85); only with --format jpg",
    )
    ap.add_argument(
        "--population-csv",
        default=None,
        help="z0_population.csv, to annotate panels with count/coverage",
    )
    ap.add_argument(
        "--annotation-csv",
        default="results/dataset_analysis/z0_collapse/" "z0_collapse_summary.csv",
        help="per-well drug/dose table, for the figure title",
    )
    ap.add_argument(
        "--no-raw-row", action="store_true", help="omit the signal-only row"
    )
    ap.add_argument(
        "--channel",
        default="BF",
        help="image channel to show under the masks (BF, mCherry, FlipGFP)",
    )
    ap.add_argument(
        "--mask-style",
        choices=(MASK_STYLE_FILL, MASK_STYLE_OUTLINE),
        default=None,
        help="'fill' tints each label; 'outline' draws boundaries only, "
        "keeping the signal visible. Default: fill for BF, outline "
        "for every other channel.",
    )
    ap.add_argument(
        "--base-cmap",
        default=None,
        help="colormap for the image; default depends on --channel",
    )
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    setup_logging()
    exp = Path(args.experiment_dir)
    split = exp / "split_data"
    masks = exp / "inference" / args.model / "masks"
    out = Path(args.output_dir)
    label = args.label or exp.name.split()[0]
    timepoints = _parse_timepoints(args.timepoints)
    # A filled overlay hides the intensity you are trying to judge, so anything but
    # brightfield defaults to outlines.
    mask_style = args.mask_style or (
        MASK_STYLE_FILL if args.channel == "BF" else MASK_STYLE_OUTLINE
    )
    base_cmap = args.base_cmap or CHANNEL_CMAP.get(args.channel, "gray")

    population = None
    if args.population_csv:
        path = Path(args.population_csv)
        if path.is_file():
            # Same rule as the plots: never print a width-flagged count onto a
            # panel that is meant to adjudicate merging vs loss.
            population = drop_unmeasurable(pd.read_csv(path))
        else:
            logger.warning("population csv %s not found — panels unannotated", path)

    annotations_by_well = {}
    annotation_path = Path(args.annotation_csv)
    if annotation_path.is_file():
        table = pd.read_csv(annotation_path)
        subset = table[table["experiment"].astype(str) == label]
        annotations_by_well = {
            str(row["well"]).upper(): row.to_dict() for _, row in subset.iterrows()
        }
        if not annotations_by_well:
            logger.warning(
                "%s: no rows in %s — titles fall back to well ids",
                label,
                annotation_path,
            )
    else:
        logger.warning(
            "annotation csv %s not found — titles use well ids only", annotation_path
        )

    wells = args.wells or available_wells(masks)
    if not wells:
        raise FileNotFoundError(f"no wells found under {masks}")
    logger.info("%s: %d well(s) at timepoints %s", label, len(wells), list(timepoints))

    for well in wells:
        frames = resolve_frames(split, masks, well, timepoints, channel=args.channel)

        annotations = None
        if population is not None:
            subset = population[
                population["sample_id"].astype(str).str.upper() == str(well).upper()
            ]
            annotations = {
                int(row["ti"]): {
                    "n_objects": row["n_objects"],
                    "coverage_fraction": row["coverage_fraction"],
                }
                for _, row in subset.iterrows()
                if pd.notna(row["n_objects"])
            }

        condition = condition_label(well, annotations_by_well.get(str(well).upper()))
        stem = f"{label}_{well}_z0"
        if args.channel != "BF":
            stem += f"_{args.channel}"
        build_filmstrip(
            frames,
            out / f"{stem}_filmstrip.{args.format}",
            title=(
                f"{label} {well} — {condition}  |  "
                f"{args.channel} vs segmentation (z0)"
            ),
            crop=args.crop,
            show_raw_row=not args.no_raw_row,
            annotations=annotations,
            mask_style=mask_style,
            base_cmap=base_cmap,
            dpi=args.dpi,
            panel_inches=args.panel_inches,
            quality=args.quality,
        )

    logger.info("%s: %d filmstrip(s) -> %s", label, len(wells), out)


if __name__ == "__main__":
    main()
