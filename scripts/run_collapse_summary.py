#!/usr/bin/env python
"""CLI: reduce per-experiment cell-population trajectories to a collapse-metrics table.

Reads each experiment's ``cell_population.csv`` (produced by
``scripts/run_cell_population.py``) and writes a single table with one row per well:
peak/edge counts, both end-percentage denominators, both ``t_cross`` variants, the width
of the usable pre-collapse window, and a shape label.

Runs the plate-layout regression guard (8 drug + 1 DMSO well per experiment, zero wells
annotated ``empty``) and fails loudly if it trips -- that is the check which would have
caught the column-7 mislabelling fixed in ``b35396f``.

See docs/feature_to_mcherry/plan_fatima_deliverable_pipeline.md, Step 1.

Example::

    python scripts/run_collapse_summary.py \\
        --results-root results/dataset_analysis \\
        --output results/dataset_analysis/all_experiments_cell_population_summary.csv
"""
import argparse
import logging
from pathlib import Path
from typing import Dict

import pandas as pd

from src.dataset_analysis.collapse_summary import (
    assert_well_composition,
    dmso_reference_table,
    summarize_all_experiments,
)
from src.dataset_analysis.layout import load_plate_layout
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

#: DMSO reference well per experiment, confirmed from on-disk filenames and from
#: cell_population.csv's own ``is_dmso`` column.
DEFAULT_DMSO_WELLS: Dict[str, str] = {
    "Ew2-1": "M11",
    "Ew2-2": "M11",
    "HD1509": "N11",
    "HD1883": "N11",
    "SA110": "N11",
}


def _discover_cell_population_csvs(results_root: Path) -> Dict[str, Path]:
    """Map ``{experiment_label: cell_population.csv}`` under ``results_root``."""
    found: Dict[str, Path] = {}
    for csv_path in sorted(results_root.glob("*/cell_population/cell_population.csv")):
        found[csv_path.parents[1].name] = csv_path
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--results-root",
        default="results/dataset_analysis",
        help="dir holding <experiment>/cell_population/cell_population.csv",
    )
    ap.add_argument(
        "--output",
        default="results/dataset_analysis/all_experiments_cell_population_summary.csv",
    )
    ap.add_argument("--layout", default="config/MF5v1_plate_layout.json")
    ap.add_argument(
        "--expected-rows",
        type=int,
        default=45,
        help="fail if the summary row count differs; 0 disables the check",
    )
    ap.add_argument(
        "--skip-composition-check",
        action="store_true",
        help="skip the 8-drug + 1-DMSO plate-layout regression guard (not advised)",
    )
    args = ap.parse_args()

    setup_logging()
    results_root = Path(args.results_root)
    csvs = _discover_cell_population_csvs(results_root)
    if not csvs:
        raise FileNotFoundError(
            f"no <experiment>/cell_population/cell_population.csv under {results_root}"
        )
    logger.info("found %d experiments: %s", len(csvs), sorted(csvs))

    layout_path = Path(args.layout)
    if not layout_path.is_file():
        raise FileNotFoundError(
            f"plate layout not found at {layout_path}; drug/dose annotation needs it"
        )
    layout = load_plate_layout(layout_path)

    summary = summarize_all_experiments(
        csvs, layout=layout, dmso_wells=DEFAULT_DMSO_WELLS
    )

    if not args.skip_composition_check:
        assert_well_composition(summary)

    if args.expected_rows and len(summary) != args.expected_rows:
        raise AssertionError(
            f"expected {args.expected_rows} summary rows, got {len(summary)} "
            f"({len(csvs)} experiments x wells) -- check well enumeration"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    logger.info("wrote %s (%d rows)", output, len(summary))

    with pd.option_context("display.width", 200, "display.max_columns", 40):
        logger.info(
            "DMSO reference wells:\n%s",
            dmso_reference_table(summary).to_string(index=False),
        )


if __name__ == "__main__":
    main()
