#!/usr/bin/env python3
"""Inspect pipeline run manifests without re-running the pipeline."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.run_manifest import RunManifest, find_manifests


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


def _show_run(output_dir: str, *, as_json: bool, next_steps_only: bool) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect pipeline run manifests")
    parser.add_argument("path", help="Output directory of a single run, or results root with --list")
    parser.add_argument("--list", action="store_true", help="List all runs under path (tabular)")
    parser.add_argument("--next-steps", action="store_true", help="Print remaining stages one per line")
    parser.add_argument("--json", action="store_true", help="Dump full manifest as JSON")
    args = parser.parse_args()

    if args.list:
        _list_runs(args.path)
    else:
        _show_run(args.path, as_json=args.json, next_steps_only=args.next_steps)


if __name__ == "__main__":
    main()
