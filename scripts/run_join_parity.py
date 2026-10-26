#!/usr/bin/env python3
"""Check that radiomics, incarta and the mCherry target hold the same cells.

Exits 0 on an exact join of all three tables, 1 on a gap, a repeated or null key or an
empty table, and 2 when the check could not run. Writes a JSON report, and the gap keys
beside it when there are any.

Example:
    uv run scripts/run_join_parity.py --label HD1509_sharpness \
        --radiomics-dir "<tree>/features_radiomics" \
        --incarta-dir "<tree>/features_incarta/split_data" \
        --target-csv "<tree>/mcherry_metrics/instance_metrics.csv"
"""

import argparse
import logging
import re
import sys
from pathlib import Path

from src.feature_to_mcherry.data_quality.join_parity import (
    MAX_GAP_ROWS,
    compare_tables,
    count_raw_target_rows,
    load_feature_keys,
    load_target_keys,
    write_report,
)
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def _label(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise argparse.ArgumentTypeError(f"{value!r} is not a plain file name")
    return value


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", type=_label, required=True)
    parser.add_argument("--radiomics-dir", type=Path, required=True)
    parser.add_argument("--incarta-dir", type=Path, required=True)
    parser.add_argument("--target-csv", type=Path, required=True)
    parser.add_argument("--radiomics-pattern", default="*.parquet")
    parser.add_argument("--incarta-pattern", default="*.csv")
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Default: results/join_parity/<label>.json",
    )
    return parser.parse_args()


def main() -> int:
    args = get_args()
    setup_logging()
    report_path = args.report or Path("results/join_parity") / f"{args.label}.json"
    report_path.unlink(missing_ok=True)
    report_path.with_suffix(".gaps.csv").unlink(missing_ok=True)
    try:
        tables = {
            "radiomics": load_feature_keys(args.radiomics_dir, args.radiomics_pattern),
            "incarta": load_feature_keys(args.incarta_dir, args.incarta_pattern),
            "target": load_target_keys(args.target_csv),
        }
        report, gaps = compare_tables(
            args.label,
            tables,
            raw_rows={"target": count_raw_target_rows(args.target_csv)},
        )
        write_report(report, gaps, report_path)
    except Exception:
        logger.exception("Join parity could not run")
        return 2

    for name, n_rows in report.rows.items():
        failing = report.duplicates[name] or report.null_keys[name] or not n_rows
        logger.log(
            logging.ERROR if failing else logging.INFO,
            "%s: %d rows, %d repeated keys, %d null keys",
            name,
            n_rows,
            report.duplicates[name],
            report.null_keys[name],
        )
    for name, n_raw in report.raw_rows.items():
        logger.info("%s: %d rows in the CSV, before any filter", name, n_raw)
    if report.gaps_truncated:
        logger.warning(
            "%d gap rows; only the first %d are in the gaps file",
            report.n_gap_rows,
            MAX_GAP_ROWS,
        )
    for pair in report.pairs:
        logger.log(
            logging.INFO if pair.exact else logging.ERROR,
            "%s vs %s: matched %d, %s-only %d, %s-only %d",
            pair.left,
            pair.right,
            pair.matched,
            pair.left,
            pair.left_only,
            pair.right,
            pair.right_only,
        )
    logger.info(
        "%s: %s. Report: %s",
        args.label,
        "EXACT" if report.exact else "GAP",
        report_path,
    )
    return 0 if report.exact else 1


if __name__ == "__main__":
    sys.exit(main())
