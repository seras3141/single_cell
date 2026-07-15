#!/usr/bin/env python3
"""CLI wrapper for the feature_to_mcherry data-quality extreme-value diagnostic.

For every numeric feature/target column, flags rows outside a [lo, hi] quantile
range and reports whether extremes concentrate by well (sample_id), z-slice, or
timepoint — the productionized version of the ad hoc investigation that surfaced
the timepoint=11 / well-level findings during the feature_to_mcherry real-data run.

Usage:
    uv run scripts/run_data_quality_extremes.py \
        --config config/data_quality_config.yaml

    uv run scripts/run_data_quality_extremes.py \
        --feature-csv path/to/features --target-csv path/to/instance_metrics.csv \
        --label MyExperiment --id-column cell_id \
        --output-dir results/data_quality/my_experiment
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import pandas as pd

from src.feature_to_mcherry.data_quality.config import (
    DataQualityConfig,
    SourceConfig,
    load_config,
)
from src.feature_to_mcherry.data_quality.extremes import compute_extreme_value_report
from src.feature_to_mcherry.data_quality.loading import load_source
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def get_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Flag extreme feature/target values and report whether they cluster by "
            "well, z-slice, or timepoint."
        )
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=None,
        help="Path to a data_quality YAML config file (supports multiple sources).",
    )
    parser.add_argument(
        "--feature-csv",
        type=str,
        default=None,
        help="Single-source convenience: path to the per-cell feature CSV/directory.",
    )
    parser.add_argument(
        "--target-csv",
        type=str,
        default=None,
        help="Single-source convenience: path to the target CSV/directory.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Single-source convenience: label for the --feature-csv/--target-csv"
        " source.",
    )
    parser.add_argument(
        "--id-column",
        type=str,
        default="instance_id",
        help="Single-source convenience: id_column for the --feature-csv source.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override: directory to write the extreme-value report to.",
    )
    parser.add_argument(
        "--override",
        "-O",
        action="append",
        default=[],
        help="Additional config overrides in dot notation, e.g."
        " extreme_quantile_hi=0.995",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def _write_report(report: pd.DataFrame, output_dir: Path, top_n: int = 15) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_dir / "extreme_value_report.csv", index=False)

    lines = ["# Extreme-value clustering report", ""]
    for source in sorted(report["source"].unique()):
        lines.append(f"## {source}")
        lines.append("")
        for group_type in ("sample_id", "z_index", "timepoint"):
            subset = report[
                (report["source"] == source) & (report["group_type"] == group_type)
            ].sort_values("enrichment", ascending=False)
            if subset.empty:
                continue
            lines.append(f"### Top enrichment by {group_type}")
            lines.append("")
            lines.append("| value_column | group_value | n | enrichment |")
            lines.append("| --- | --- | --- | --- |")
            for _, row in subset.head(top_n).iterrows():
                lines.append(
                    f"| {row['value_column']} | {row['group_value']} | {row['n']} | "
                    f"{row['enrichment']:.2f} |"
                )
            lines.append("")
    (output_dir / "report.md").write_text("\n".join(lines))
    logger.info("Wrote extreme-value report to %s", output_dir)


def main() -> None:
    """Run the data-quality extreme-value diagnostic CLI."""
    args = get_args()
    setup_logging(level=args.log_level)

    if args.feature_csv is not None or args.target_csv is not None:
        if not (args.feature_csv and args.target_csv and args.label):
            raise ValueError(
                "--feature-csv/--target-csv/--label must all be given together for "
                "a single-source run."
            )
        config = DataQualityConfig(
            sources=[
                SourceConfig(
                    label=args.label,
                    feature_csv=args.feature_csv,
                    target_csv=args.target_csv,
                    id_column=args.id_column,
                )
            ],
            output_dir=args.output_dir or "results/data_quality",
        )
    else:
        overrides: List[str] = list(args.override)
        if args.output_dir is not None:
            overrides.append(f"output_dir={args.output_dir}")
        config = load_config(yaml_path=args.config, overrides=overrides)

    reports = []
    for source in config.sources:
        logger.info("Loading source %r", source.label)
        features_df, targets_df = load_source(
            source, target_columns=config.target_columns
        )
        reports.append(
            compute_extreme_value_report(
                features_df,
                targets_df,
                source_label=source.label,
                quantile_lo=config.extreme_quantile_lo,
                quantile_hi=config.extreme_quantile_hi,
            )
        )

    combined = pd.concat(reports, ignore_index=True)
    _write_report(combined, Path(config.output_dir))

    logger.info(
        "Done. %d sources, %d report rows. Outputs written to %s",
        len(config.sources),
        len(combined),
        config.output_dir,
    )


if __name__ == "__main__":
    main()
