"""Command-line entry point for mCherry metrics extraction."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import ExtractionConfig
from .core.batch import run_extraction


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser for the extraction CLI.
    """
    parser = argparse.ArgumentParser(
        description="Extract per-instance mCherry intensity metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mcherry-dir", required=True, type=Path)
    parser.add_argument("--mask-dir", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--image-pattern", default="*_mCherry.tif")
    parser.add_argument("--label-suffix", default="_Cells")
    parser.add_argument(
        "--percentiles",
        nargs="+",
        type=int,
        default=[75, 90, 95],
        metavar="P",
    )
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--min-area-px", type=int, default=10)
    parser.add_argument(
        "--normalize-before-extraction",
        action="store_true",
        help="Normalize intensities before computing metrics.",
    )
    parser.add_argument(
        "--normalize-mode",
        choices=["minmax", "percentile"],
        default="minmax",
    )
    parser.add_argument("--gaussian-sigma", type=float, default=0.0)
    parser.add_argument("--median-footprint", type=int, default=0)
    parser.add_argument("--background-subtract-radius", type=int, default=0)
    parser.add_argument(
        "--include-z0",
        action="store_true",
        help="Keep z0 slices instead of excluding them.",
    )
    parser.add_argument(
        "--skip-analytics",
        action="store_true",
        help="Do not write analytics outputs.",
    )
    parser.add_argument(
        "--no-save-individual-files",
        action="store_true",
        help="Only write the combined instance_metrics.csv; skip per-image "
        "CSVs in the split_data/ subdirectory.",
    )
    parser.add_argument("--plate-number", default=None)
    parser.add_argument("--wavelength-w1", default=None)
    parser.add_argument("--wavelength-w2", default=None)
    parser.add_argument("--wavelength-w3", default=None)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def main() -> None:
    """Run the extraction CLI."""
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
    )

    if not args.mcherry_dir.is_dir():
        parser.error(f"--mcherry-dir does not exist: {args.mcherry_dir}")

    if args.mask_dir is not None and not args.mask_dir.is_dir():
        parser.error(f"--mask-dir does not exist: {args.mask_dir}")

    config = ExtractionConfig(
        percentiles=args.percentiles,
        normalize_before_extraction=args.normalize_before_extraction,
        normalize_mode=args.normalize_mode,
        gaussian_sigma=args.gaussian_sigma,
        median_footprint=args.median_footprint,
        background_subtract_radius=args.background_subtract_radius,
        n_jobs=args.n_jobs,
        exclude_z0=not args.include_z0,
        min_area_px=args.min_area_px,
        write_analytics=not args.skip_analytics,
        save_individual_files=not args.no_save_individual_files,
    )

    wavelength_mappings = {
        index: channel
        for index, channel in {
            1: args.wavelength_w1,
            2: args.wavelength_w2,
            3: args.wavelength_w3,
        }.items()
        if channel
    }

    try:
        metrics_df = run_extraction(
            mcherry_dir=args.mcherry_dir,
            output_dir=args.output_dir,
            config=config,
            mask_dir=args.mask_dir,
            image_pattern=args.image_pattern,
            label_suffix=args.label_suffix,
            wavelength_mappings=wavelength_mappings or None,
            plate_number=args.plate_number,
        )
    except ValueError as exc:
        logging.error("%s", exc)
        sys.exit(1)

    logging.info("Wrote metrics for %d instances", len(metrics_df))


if __name__ == "__main__":
    main()
