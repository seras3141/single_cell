#!/usr/bin/env python3
"""CLI wrapper for the feature_to_mcherry regression baselines.

Fits Ridge and linear-quantile-regression baselines mapping per-cell brightfield
features to mCherry intensity percentiles, using grouped cross-validation, and writes
a baseline-ladder comparison report.

Usage:
    uv run scripts/run_feature_to_mcherry.py \
        --config config/feature_to_mcherry_config.yaml \
        --override feature_csv=path/to/features.csv \
        --override target_csv=path/to/instance_metrics.csv

    uv run scripts/run_feature_to_mcherry.py \
        --feature-csv path/to/features.csv \
        --target-csv path/to/instance_metrics.csv \
        --output-dir results/feature_to_mcherry

    uv run scripts/run_feature_to_mcherry.py \
        --config config/feature_to_mcherry_config.yaml \
        --override ridge_alpha=10 --override group_by=timepoint
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

from src.feature_to_mcherry.config import load_config
from src.feature_to_mcherry.pipeline import run
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def get_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Fit Ridge + linear quantile regression baselines mapping brightfield "
            "features to mCherry percentiles."
        )
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=None,
        help="Path to a feature_to_mcherry YAML config file.",
    )
    parser.add_argument(
        "--feature-csv",
        type=str,
        default=None,
        help="Override: path to the per-cell feature CSV.",
    )
    parser.add_argument(
        "--target-csv",
        type=str,
        default=None,
        help="Override: path to the mcherry_metrics instance-metrics CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override: directory to write results to.",
    )
    parser.add_argument(
        "--override",
        "-O",
        action="append",
        default=[],
        help="Additional config overrides in dot notation, e.g. ridge_alpha=10",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    """Run the feature_to_mcherry CLI."""
    args = get_args()
    setup_logging(level=args.log_level)

    overrides: List[str] = list(args.override)
    if args.feature_csv is not None:
        overrides.append(f"feature_csv={args.feature_csv}")
    if args.target_csv is not None:
        overrides.append(f"target_csv={args.target_csv}")
    if args.output_dir is not None:
        overrides.append(f"output_dir={args.output_dir}")

    config = load_config(yaml_path=args.config, overrides=overrides)
    results = run(config)

    logger.info(
        "Done. n_cells=%d n_features=%d | ridge pooled: %s | "
        "linear_quantile pooled: %s",
        results.n_cells,
        results.n_features,
        results.ridge.pooled_metrics,
        results.linear_quantile.pooled_metrics,
    )


if __name__ == "__main__":
    main()
