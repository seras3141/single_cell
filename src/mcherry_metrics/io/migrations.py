"""One-off migration utilities for the ``label_id`` → ``cell_id`` rename.

The per-cell identity column was renamed from ``label_id`` to ``cell_id`` with
no backward-compatibility shim (see
``docs/mcherry_activity/plan_label_id_to_cell_id_rename.md``). CSVs written
before the rename carry a ``label_id`` header token; the new loaders reject
them. This module rewrites the **header token only** — the column values are
integer cell IDs and never change — so already-written outputs keep loading.
"""

from __future__ import annotations

import csv
import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

OLD_COLUMN = "label_id"
NEW_COLUMN = "cell_id"


def migrate_label_id_header(csv_path: Path, *, dry_run: bool = True) -> bool:
    """Rewrite a CSV's ``label_id`` header token to ``cell_id`` in place.

    Only the header row is inspected and rewritten; data rows are copied
    verbatim. The operation is idempotent — a file already using ``cell_id``
    (and no ``label_id``) is a no-op.

    Args:
        csv_path: Path to the CSV file to migrate.
        dry_run: When ``True`` (default), report whether a change *would* be
            made without touching the file. When ``False``, rewrite in place
            via an atomic temp-file replace.

    Returns:
        ``True`` if the file needs / received the rename, ``False`` if it was
        already migrated (no ``label_id`` header present).

    Raises:
        ValueError: If the header contains *both* ``label_id`` and ``cell_id``
            (ambiguous — refuse rather than silently drop a column).
        FileNotFoundError: If ``csv_path`` does not exist.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with csv_path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            logger.warning("Skipping empty file: %s", csv_path)
            return False

    if OLD_COLUMN not in header:
        return False

    if NEW_COLUMN in header:
        raise ValueError(
            f"{csv_path} has both '{OLD_COLUMN}' and '{NEW_COLUMN}' columns; "
            "refusing to migrate. Resolve the duplicate manually."
        )

    if dry_run:
        logger.info(
            "[dry-run] would rename '%s' -> '%s' in %s",
            OLD_COLUMN,
            NEW_COLUMN,
            csv_path,
        )
        return True

    new_header = [NEW_COLUMN if col == OLD_COLUMN else col for col in header]

    fd, tmp_name = tempfile.mkstemp(
        dir=str(csv_path.parent), prefix=csv_path.name, suffix=".tmp"
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", newline="") as tmp_handle, csv_path.open(
            "r", newline=""
        ) as src_handle:
            writer = csv.writer(tmp_handle)
            src_reader = csv.reader(src_handle)
            next(src_reader)  # drop the original header
            writer.writerow(new_header)
            for row in src_reader:
                writer.writerow(row)
        os.replace(tmp_path, csv_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    logger.info("renamed '%s' -> '%s' in %s", OLD_COLUMN, NEW_COLUMN, csv_path)
    return True


__all__ = ["migrate_label_id_header", "OLD_COLUMN", "NEW_COLUMN"]
