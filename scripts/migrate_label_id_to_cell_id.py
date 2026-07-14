"""One-off CLI to migrate CSV headers from ``label_id`` to ``cell_id``.

Walks one or more root directories, finds every ``*.csv`` whose header contains
the legacy ``label_id`` column, and rewrites the header token to ``cell_id``
(row values are unchanged). Defaults to ``--dry-run``; pass ``--apply`` to write.

CSVs are gitignored, so this only touches on-disk outputs — run it on the
storage holding the per-experiment ``mcherry_metrics`` / activity CSVs *before*
the new (``cell_id``-only) loaders are used on them.

Example:
    uv run scripts/migrate_label_id_to_cell_id.py /path/to/experiments --apply
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.mcherry_metrics.io.migrations import migrate_label_id_header

logger = logging.getLogger("migrate_label_id_to_cell_id")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roots",
        type=Path,
        nargs="+",
        help="One or more root directories to scan recursively for *.csv files.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Rewrite files in place. Without this flag the run is a dry-run.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
    )
    dry_run = not args.apply

    changed = 0
    scanned = 0
    errors = 0
    for root in args.roots:
        if not root.exists():
            logger.error("Root does not exist: %s", root)
            errors += 1
            continue
        for csv_path in sorted(root.rglob("*.csv")):
            scanned += 1
            try:
                if migrate_label_id_header(csv_path, dry_run=dry_run):
                    changed += 1
            except ValueError as exc:
                logger.error("%s", exc)
                errors += 1

    verb = "would be renamed" if dry_run else "renamed"
    logger.info(
        "Scanned %d CSV(s); %d %s; %d error(s).%s",
        scanned,
        changed,
        verb,
        errors,
        "" if not dry_run else " Re-run with --apply to write.",
    )
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
