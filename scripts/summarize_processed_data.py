#!/usr/bin/env python3
"""Scan processed output directories and produce a per-stage file inventory and gap report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.dataset_analysis.processed_inventory import (
    build_processed_inventory,
    build_processed_summary,
    print_summary_table,
)


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
    args = parser.parse_args()

    output_dir: Path = args.output_dir or args.processed_dir / "processed_summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_inventory = pd.read_csv(args.raw_inventory)
    print(f"Loaded raw inventory: {len(raw_inventory)} rows from {args.raw_inventory}")

    print("Building processed inventory...")
    inventory = build_processed_inventory(
        raw_inventory=raw_inventory,
        processed_dir=args.processed_dir,
        experiment_name=args.experiment_name,
    )

    issues = inventory[~inventory["found"]]
    summary = build_processed_summary(inventory)

    inventory_path = output_dir / "processed_inventory.csv"
    issues_path = output_dir / "processed_issues.csv"
    summary_path = output_dir / "processed_summary.json"

    inventory.to_csv(inventory_path, index=False)
    print(f"Written: {inventory_path}  ({len(inventory)} rows)")

    issues.to_csv(issues_path, index=False)
    print(f"Written: {issues_path}  ({len(issues)} rows)")

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Written: {summary_path}")

    print()
    print_summary_table(summary)


if __name__ == "__main__":
    main()
