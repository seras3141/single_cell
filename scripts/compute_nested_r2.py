#!/usr/bin/env python3
"""Write fold_spread.csv and nested_r2.csv beside a baselines run's predictions.

Usage:
    uv run scripts/compute_nested_r2.py <baselines_dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.feature_to_mcherry.evaluation.run_decomposition import decompose_run
from src.utils.logging_utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-fold and nested R² for a finished baselines run."
    )
    parser.add_argument("baselines_dir", type=Path)
    setup_logging()
    decompose_run(parser.parse_args().baselines_dir)


if __name__ == "__main__":
    main()
