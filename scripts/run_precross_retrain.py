#!/usr/bin/env python3
"""Retrain the radiomics-to-mCherry models on the pre-confluence window only.

Usage:
    uv run scripts/run_precross_retrain.py --culture HD1509 --experiment-dir "<dir>"
Writes to results/feature_to_mcherry_precross_retrain/<culture>/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.feature_to_mcherry.precross_retrain import retrain_culture
from src.utils.logging_utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-window retraining of the radiomics-to-mCherry models."
    )
    parser.add_argument("--culture", required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--dmso-well", default="N11")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/feature_to_mcherry_radiomics_config.yaml"),
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path(
            "results/dataset_analysis/all_experiments_cell_population_summary.csv"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/feature_to_mcherry_precross_retrain"),
    )
    args = parser.parse_args()
    setup_logging()
    retrain_culture(
        args.culture,
        args.experiment_dir,
        args.dmso_well,
        args.config,
        args.summary_csv,
        args.output_root,
    )


if __name__ == "__main__":
    main()
