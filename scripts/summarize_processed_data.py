#!/usr/bin/env python3
"""Scan processed output directories and produce a per-stage file inventory and gap report."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.dataset_analysis.processed_inventory import (
    STAGE_ORDER,
    annotate_with_raw_issues,
    build_processed_inventory,
    build_processed_summary,
    detect_phantom_samples,
    print_phantom_report,
    print_summary_table,
)

_PHANTOM_STAGES = [s for s in STAGE_ORDER if s != "mcherry"]


def _delete_phantoms(phantoms: dict, stages: list) -> None:
    for stage in stages:
        files = phantoms.get(stage, [])
        if not files:
            print(f"  {stage}: no phantoms to delete.")
            continue
        for path in files:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            print(f"  Deleted: {path}")
        print(f"  {stage}: {len(files)} phantom(s) deleted.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan processed outputs and report per-stage file completeness."
    )
    parser.add_argument(
        "--raw-inventory",
        required=True,
        type=Path,
        help="Path to dataset_inventory.csv (ground truth of expected files)",
    )
    parser.add_argument(
        "--processed-dir",
        required=True,
        type=Path,
        help="Root of processed output directory for one experiment",
    )
    parser.add_argument(
        "--experiment-name",
        required=True,
        help="Experiment name for channel mapping (e.g. HD1509, Ew2-1)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write outputs (default: <processed-dir>/processed_summary/)",
    )
    parser.add_argument(
        "--delete",
        nargs="+",
        choices=_PHANTOM_STAGES,
        metavar="STAGE",
        default=None,
        help=(
            "Delete phantom files for the given stage(s) after detection. "
            f"Valid stages: {', '.join(_PHANTOM_STAGES)}. "
            "Example: --delete prepare-3d segment-2d"
        ),
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir or args.processed_dir / "processed_summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_inventory = pd.read_csv(args.raw_inventory)
    print(f"Loaded raw inventory: {len(raw_inventory)} rows from {args.raw_inventory}")

    issues_csv = args.raw_inventory.parent / "dataset_issues.csv"
    issues_df = None
    if issues_csv.exists():
        issues_df = pd.read_csv(issues_csv)
        print(f"Loaded raw issues: {len(issues_df)} rows from {issues_csv}")

    print("Building processed inventory...")
    inventory = build_processed_inventory(
        raw_inventory=raw_inventory,
        processed_dir=args.processed_dir,
        experiment_name=args.experiment_name,
    )

    issues = inventory[~inventory["found"]]
    summary = build_processed_summary(inventory, issues_df=issues_df)

    inventory_path = output_dir / "processed_inventory.csv"
    issues_path = output_dir / "processed_issues.csv"
    summary_path = output_dir / "processed_summary.json"

    out_inventory = annotate_with_raw_issues(inventory, issues_df) if issues_df is not None else inventory
    out_inventory.to_csv(inventory_path, index=False)
    print(f"Written: {inventory_path}  ({len(inventory)} rows)")

    issues.to_csv(issues_path, index=False)
    print(f"Written: {issues_path}  ({len(issues)} rows)")

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Written: {summary_path}")

    phantoms = detect_phantom_samples(raw_inventory, args.processed_dir)
    phantom_path = output_dir / "phantom_samples.json"
    phantom_data = {stage: [str(p) for p in paths] for stage, paths in phantoms.items()}
    with open(phantom_path, "w") as f:
        json.dump(phantom_data, f, indent=2)
    print(f"Written: {phantom_path}")

    print()
    print_summary_table(summary)
    print()
    print_phantom_report(phantoms)

    if args.delete:
        print()
        print(f"Deleting phantoms for stage(s): {', '.join(args.delete)}")
        _delete_phantoms(phantoms, args.delete)
        print("Done. Re-run the script without --delete to confirm removal.")


if __name__ == "__main__":
    main()
