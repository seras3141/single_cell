#!/usr/bin/env python3
"""Compare fold assignments between the radiomics and incarta baselines runs.

Usage:
    uv run scripts/compare_fold_assignments.py [--root results/radiomics_phase6]
Exits non-zero if no run is found, a run lacks a backend, or any pair differs.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.feature_to_mcherry.evaluation.fold_comparison import (
    compare_backends_under_root,
)
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check that radiomics and incarta baselines used the same folds."
    )
    parser.add_argument("--root", type=Path, default=Path("results/radiomics_phase6"))
    args = parser.parse_args()
    setup_logging()

    results = compare_backends_under_root(args.root)
    if not results:
        logger.error("No baselines runs found under %s", args.root)
        return 1
    for label, problems in results.items():
        if problems:
            logger.error("%s: %s", label, "; ".join(problems))
        else:
            logger.info("%s: folds identical", label)
    return 1 if any(results.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
