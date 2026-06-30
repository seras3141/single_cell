"""CLI wrapper for mCherry metrics extraction with optional manifest integration."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.mcherry_metrics.__main__ import build_parser
from src.mcherry_metrics.config import ExtractionConfig
from src.mcherry_metrics.core.batch import run_extraction
from src.dataset_analysis.run_manifest import create_or_load_manifest


def _get_mcherry_snapshot(args) -> Dict[str, Any]:
    return {k: v for k, v in {
        "percentiles": args.percentiles,
        "normalize_before_extraction": args.normalize_before_extraction,
        "normalize_mode": args.normalize_mode,
        "gaussian_sigma": args.gaussian_sigma,
        "min_area_px": args.min_area_px,
    }.items() if v is not None}


def main() -> None:
    parser = build_parser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Experiment root directory where manifest.json lives. If omitted, manifest is not updated.",
    )
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

    manifest = None
    if args.run_dir is not None:
        run_dir = str(args.run_dir)
        os.makedirs(run_dir, exist_ok=True)
        manifest = create_or_load_manifest(run_dir, str(args.mcherry_dir), {})
        snapshot = _get_mcherry_snapshot(args)
        manifest.start_stage("mcherry", config=snapshot)

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
        if manifest is not None:
            manifest.complete_stage("mcherry", output_dir=str(args.output_dir))
            logging.info(manifest.summary())
    except Exception as exc:
        logging.error("%s", exc)
        if manifest is not None:
            manifest.fail_stage("mcherry", error=str(exc))
            logging.info(manifest.summary())
        sys.exit(1)

    logging.info("Wrote metrics for %d instances", len(metrics_df))


if __name__ == "__main__":
    main()
