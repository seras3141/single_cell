#!/usr/bin/env python3
"""CLI wrapper for generating the report figures.

Reads already-computed feature_to_mcherry / morphology_informativeness /
data_quality results and writes the figures specified in
docs_local/figures/report_figures_brief.md, plus a manifest of what was generated,
partially generated, or blocked.

Usage:
    uv run scripts/run_report_figures.py \
        --config config/report_figures_config.yaml

    uv run scripts/run_report_figures.py \
        --figures A1,A3,C1,C3 \
        --output-dir results/report_figures

    uv run scripts/run_report_figures.py \
        --override experiments=[Ew2-1,Ew2-2] --override scportrait_experiments=[]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

from src.feature_to_mcherry.report_figures.config import load_config
from src.feature_to_mcherry.report_figures.pipeline import FIGURE_FUNCTIONS, run
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def get_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate the report figures from already-computed feature_to_mcherry / "
            "morphology_informativeness / data_quality results."
        )
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=None,
        help="Path to a report_figures YAML config file.",
    )
    parser.add_argument(
        "--figures",
        type=str,
        default=None,
        help=(
            "Comma-separated figure IDs to generate (e.g. A1,C3,C4). Default: all "
            f"known figures ({', '.join(FIGURE_FUNCTIONS)})."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override: directory to write figures and the manifest to.",
    )
    parser.add_argument(
        "--override",
        "-O",
        action="append",
        default=[],
        help="Additional config overrides in dot notation, e.g. dpi=150",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    """Run the report-figures CLI."""
    args = get_args()
    setup_logging(level=args.log_level)

    overrides: List[str] = list(args.override)
    if args.output_dir is not None:
        overrides.append(f"output_dir={args.output_dir}")

    config = load_config(yaml_path=args.config, overrides=overrides)
    figure_ids = args.figures.split(",") if args.figures else None

    manifest = run(config, figure_ids=figure_ids)

    counts = {
        status: sum(1 for e in manifest.entries if e.status == status)
        for status in ("generated", "partial", "blocked")
    }
    logger.info(
        "Done. generated=%d partial=%d blocked=%d. Manifest written to %s",
        counts["generated"],
        counts["partial"],
        counts["blocked"],
        config.output_dir,
    )


if __name__ == "__main__":
    main()
