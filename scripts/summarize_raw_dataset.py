#!/usr/bin/env python3
"""Report expected-vs-found inventory for a downloaded raw share tree.

Compares each experiment directory (and nested ``*_Projection`` folder) against
its share manifest and a dense acquisition grid, writing ``raw_summary/`` beside
each experiment.

Example::

    uv run scripts/summarize_raw_dataset.py \
        --dataset-root "data/MF5V1 Timepoints 1-40" \
        --manifest-dir data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset_analysis.raw_share_summary import (  # noqa: E402
    MEASURED_BF_WAVELENGTH,
    build_raw_summary,
    format_summary_table,
    write_raw_summary,
)


def _experiment_dirs(root: Path) -> list[Path]:
    """Top-level experiment directories under *root* (projections excluded)."""
    return sorted(
        p for p in root.iterdir() if p.is_dir() and not p.name.endswith("_Projection")
    )


def _folders_for(experiment: Path) -> list[Path]:
    """The experiment directory followed by any nested projection folders."""
    nested = sorted(
        p for p in experiment.iterdir() if p.is_dir() and p.name.endswith("_Projection")
    )
    return [experiment] + nested


def _manifest_for(folder: Path, manifest_dir: Path | None) -> Path | None:
    if manifest_dir is None:
        return None
    candidate = manifest_dir / f"{folder.name}.filelist.txt"
    return candidate if candidate.is_file() else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize raw-share download and acquisition completeness."
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        metavar="DIR",
        help="Dataset root holding one directory per experiment.",
    )
    parser.add_argument(
        "--manifest-dir",
        metavar="DIR",
        help="Directory holding '<folder>.filelist.txt' share manifests. "
        "When given, download completeness is reported.",
    )
    parser.add_argument(
        "--expected-timepoints",
        type=int,
        default=40,
        metavar="N",
        help="Dense grid runs t1..tN (default: 40).",
    )
    parser.add_argument(
        "--out-dirname",
        default="raw_summary",
        metavar="NAME",
        help="Subdirectory written beside each experiment (default: raw_summary).",
    )
    args = parser.parse_args()

    root = Path(args.dataset_root)
    if not root.is_dir():
        sys.exit(f"Dataset root not found: {root}")
    manifest_dir = Path(args.manifest_dir) if args.manifest_dir else None
    timepoints = range(1, args.expected_timepoints + 1)

    experiments = _experiment_dirs(root)
    if not experiments:
        sys.exit(f"No experiment directories under {root}")

    for experiment in experiments:
        summaries = []
        for folder in _folders_for(experiment):
            summaries.append(
                build_raw_summary(
                    folder,
                    manifest_path=_manifest_for(folder, manifest_dir),
                    expected_timepoints=tuple(timepoints),
                )
            )
        written = write_raw_summary(summaries, experiment / args.out_dirname)

        prefix = experiment.name.split()[0]
        bf = MEASURED_BF_WAVELENGTH.get(prefix)
        bf_note = f"brightfield = w{bf} (measured)" if bf else "brightfield = unknown"
        print(f"\n=== {experiment.name}  [{bf_note}] ===")
        print(format_summary_table(summaries))
        print(f"  wrote {written['summary'].parent}")


if __name__ == "__main__":
    main()
