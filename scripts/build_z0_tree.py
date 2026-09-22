#!/usr/bin/env python
"""CLI: stage the z0 projection plane into a parallel processed tree.

Mirrors ``split_data/`` and ``inference/<model>/masks/`` for every experiment, keeping
only the z0 files and preserving the folder structure. Symlinks by default, so the tree
costs nothing and stays rebuildable.

Filtering, tracking and 3D combining are intentionally not staged: a single plane has
no stack to combine and no slices to choose between, and per-timepoint population
statistics need no temporal identity.

See docs/dataset_analysis/plan_z0_projection_count_check.md.
"""
import argparse
import logging
from pathlib import Path

from src.dataset_analysis.z0_tree import build_z0_tree
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

DEFAULT_SOURCE = "data/MF5V1_processed Timelapse samples 19.03.2024"
DEFAULT_DEST = "data/MF5V1_processed_z0 Timelapse samples 19.03.2024"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--source-root", default=DEFAULT_SOURCE, help="processed tree to stage from"
    )
    ap.add_argument("--dest-root", default=DEFAULT_DEST, help="z0 tree to create")
    ap.add_argument(
        "--experiment",
        action="append",
        dest="experiments",
        default=None,
        help="experiment dir name; repeatable. Default: all found.",
    )
    ap.add_argument("--model", default="cellpose_sam")
    ap.add_argument("--mode", choices=("symlink", "copy"), default="symlink")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be staged without writing anything",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="re-materialize entries that already look current "
        "(copy-mode payloads are not content-verified otherwise)",
    )
    ap.add_argument(
        "--summary-csv",
        default=None,
        help="optional path to write the per-stage summary table",
    )
    args = ap.parse_args()

    setup_logging()
    logger.info(
        "staging z0 (%s): %s -> %s", args.mode, args.source_root, args.dest_root
    )

    summary = build_z0_tree(
        source_root=args.source_root,
        dest_root=args.dest_root,
        experiments=args.experiments,
        model=args.model,
        mode=args.mode,
        dry_run=args.dry_run,
        force=args.force,
    )

    print(summary.to_string(index=False))
    if summary.empty:
        print("no stage folders found — nothing staged")
    else:
        print(
            f"\ntotal z0 staged: {int(summary['n_z0'].sum())} "
            f"({int(summary['n_created'].sum())} new, "
            f"{int(summary['n_repaired'].sum())} rebuilt, "
            f"{int(summary['n_existing'].sum())} already present)"
        )
    if args.dry_run:
        print("DRY RUN — nothing was written.")

    if args.summary_csv:
        out = Path(args.summary_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out, index=False)
        logger.info("wrote summary: %s", out)


if __name__ == "__main__":
    main()
