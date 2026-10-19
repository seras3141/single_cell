#!/usr/bin/env python3
"""Audit per-well radiomics tables against the masks they were extracted from.

Exits 1 when any well fails and 2 when the audit could not run. Writes
``mask_parity_audit.json`` into the features folder unless ``--report`` says otherwise.

Example:
    uv run scripts/run_feature_audit.py \
        --features-dir "<exp>/inference_tracked/cellpose_sam/features_radiomics" \
        --mask-dir "<exp>/inference_tracked/cellpose_sam/final_2d" --n-jobs 8
"""

import argparse
import logging
import sys
from pathlib import Path

from src.feature_extraction.mask_parity_audit import (
    AUDIT_FILENAME,
    audit_tree,
    write_audit,
)
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument(
        "--wells",
        nargs="+",
        help="Default: every well with a feature or coverage table",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=None,
        help="Pixel floor used by the extraction (default: the PyradiomicsConfig one)",
    )
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help=f"Default: <features-dir>/{AUDIT_FILENAME}",
    )
    return parser.parse_args()


def main() -> int:
    args = get_args()
    setup_logging()
    try:
        audit = audit_tree(
            args.features_dir,
            args.mask_dir,
            wells=args.wells,
            min_pixels=args.min_pixels,
            n_jobs=args.n_jobs,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Audit could not run: %s", exc)
        return 2

    for well in audit.wells:
        for failure in well.failures:
            logger.error("%s: %s", well.well, failure)
    try:
        report = write_audit(audit, args.report or args.features_dir / AUDIT_FILENAME)
    except OSError as exc:
        logger.error("Audit could not write its report: %s", exc)
        return 2
    logger.info(
        "%d wells, %d rows: %s. Report: %s",
        len(audit.wells),
        audit.n_feature_rows,
        "PASS" if audit.passed else "FAIL",
        report,
    )
    return 0 if audit.passed else 1


if __name__ == "__main__":
    sys.exit(main())
