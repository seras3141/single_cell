#!/usr/bin/env python3
"""CLI wrapper for the feature_to_mcherry data-quality interactive well/timepoint
report.

Renders a self-contained interactive HTML report (Plotly, no CDN calls) of median +
interquartile-range per well across timepoint, for every feature and target column —
the productionized version of the ad hoc chart investigation that surfaced the
timepoint=11 / E07-well findings during the feature_to_mcherry real-data run.

Usage:
    uv run scripts/run_data_quality_well_timepoint_report.py \
        --config config/data_quality_config.yaml

    uv run scripts/run_data_quality_well_timepoint_report.py \
        --feature-csv path/to/features --target-csv path/to/instance_metrics.csv \
        --label MyExperiment --id-column cell_id \
        --output-dir results/data_quality/my_experiment
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

from src.feature_to_mcherry.data_quality.config import (
    DataQualityConfig,
    SourceConfig,
    load_config,
)
from src.feature_to_mcherry.data_quality.loading import load_source
from src.feature_to_mcherry.data_quality.plots import build_interactive_report_html
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def get_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Render an interactive per-well, per-timepoint feature/target "
            "distribution report."
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
        help="Override: directory to write the report to.",
    )
    parser.add_argument(
        "--override",
        "-O",
        action="append",
        default=[],
        help="Additional config overrides in dot notation.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    """Run the data-quality well/timepoint report CLI."""
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

    sources_data = []
    for source in config.sources:
        logger.info("Loading source %r", source.label)
        features_df, targets_df = load_source(
            source, target_columns=config.target_columns
        )
        sources_data.append((source.label, features_df, targets_df))

    html = build_interactive_report_html(
        sources_data,
        target_columns=config.target_columns,
        feature_column_groups=config.feature_column_groups,
        flag_timepoints=config.flag_timepoints,
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "well_timepoint_report.html"
    report_path.write_text(html)

    logger.info(
        "Done. %d sources. Report written to %s (%.1f MB)",
        len(config.sources),
        report_path,
        report_path.stat().st_size / 1e6,
    )


if __name__ == "__main__":
    main()
