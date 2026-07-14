#!/usr/bin/env python3
"""Inspect pipeline run manifests without re-running the pipeline."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.dataset_analysis.run_manifest import RunManifest, find_manifests
from src.dataset_analysis.processed_inventory import (
    build_processed_inventory,
    build_processed_summary,
    print_summary_table,
)


def _list_runs(results_dir: str) -> None:
    manifests = find_manifests(results_dir)
    if not manifests:
        print("No manifest.json files found.")
        return
    header = f"{'Experiment':<24} {'Status':<22} Next steps"
    print(header)
    print("-" * 70)
    for m in manifests:
        status = m.overall_status()
        remaining = " → ".join(m.next_steps()) or "—"
        print(f"{m.experiment_id:<24} {status:<22} {remaining}")


def _show_run(
    output_dir: str,
    *,
    as_json: bool,
    next_steps_only: bool,
    completeness: bool,
    raw_inventory: str | None,
    experiment_name: str | None,
) -> None:
    try:
        manifest = RunManifest.load_from_output_dir(output_dir)
    except FileNotFoundError:
        print(f"No manifest.json found in {output_dir}", file=sys.stderr)
        sys.exit(1)

    if next_steps_only:
        for step in manifest.next_steps():
            print(step)
        return

    if as_json:
        print(json.dumps(manifest.to_dict(), indent=2))
        return

    print(manifest.summary())

    if completeness:
        if not raw_inventory:
            print("\nError: --raw-inventory is required when using --completeness.", file=sys.stderr)
            sys.exit(1)
        if not experiment_name:
            print("\nError: --experiment-name is required when using --completeness.", file=sys.stderr)
            sys.exit(1)
        raw_df = pd.read_csv(raw_inventory)
        inventory = build_processed_inventory(
            raw_inventory=raw_df,
            processed_dir=Path(output_dir),
            experiment_name=experiment_name,
        )
        summary = build_processed_summary(inventory)
        print()
        print_summary_table(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect pipeline run manifests")
    parser.add_argument("path", help="Output directory of a single run, or results root with --list")
    parser.add_argument("--list", action="store_true", help="List all runs under path (tabular)")
    parser.add_argument("--next-steps", action="store_true", help="Print remaining stages one per line")
    parser.add_argument("--json", action="store_true", help="Dump full manifest as JSON")
    parser.add_argument(
        "--completeness",
        action="store_true",
        help="Run processed inventory check and print per-stage completeness table (single-run mode only)",
    )
    parser.add_argument(
        "--raw-inventory",
        default=None,
        help="Path to dataset_inventory.csv — required with --completeness",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Experiment name for channel mapping (e.g. HD1509) — required with --completeness",
    )
    args = parser.parse_args()

    if args.list:
        if args.completeness:
            print("Error: --completeness is not supported with --list.", file=sys.stderr)
            sys.exit(1)
        _list_runs(args.path)
    else:
        _show_run(
            args.path,
            as_json=args.json,
            next_steps_only=args.next_steps,
            completeness=args.completeness,
            raw_inventory=args.raw_inventory,
            experiment_name=args.experiment_name,
        )


if __name__ == "__main__":
    main()
