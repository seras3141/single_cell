#!/usr/bin/env python3
"""CLI wrapper for the morphology-informativeness feasibility gate.

Quantifies how much interpretable brightfield morphology (size/shape + texture)
predicts per-cell mCherry percentiles, producing a univariate-association report, a
grouped-CV performance floor (linear + nonlinear), a replicate-based noise ceiling,
and PNG/HTML figures.

Usage:
    uv run scripts/run_morphology_informativeness.py \
        --config config/morphology_informativeness_config.yaml \
        --override feature_csv=path/to/features.csv \
        --override target_csv=path/to/instance_metrics.csv

    uv run scripts/run_morphology_informativeness.py \
        --feature-csv path/to/features.csv \
        --target-csv path/to/instance_metrics.csv \
        --output-dir results/morphology_informativeness
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

from src.feature_to_mcherry.informativeness.config import load_config
from src.feature_to_mcherry.informativeness.pipeline import run
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def get_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Quantify how much brightfield morphology predicts mCherry percentiles, "
            "as a feasibility gate before investing in a deep-feature model."
        )
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=None,
        help="Path to a morphology_informativeness YAML config file.",
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
        help="Additional config overrides in dot notation, e.g. n_splits=3",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    """Run the morphology-informativeness CLI."""
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
    bundle = run(config)

    logger.info(
        "Done. n_cells=%d n_features_all=%d n_features_clean=%d. "
        "Outputs written to %s",
        bundle.n_cells,
        bundle.n_features_all,
        bundle.n_features_clean,
        config.output_dir,
    )


if __name__ == "__main__":
    main()
