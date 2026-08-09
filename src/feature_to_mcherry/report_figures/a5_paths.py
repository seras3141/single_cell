"""Path resolution for A5's extracted area/percentile_90 CSVs.

A narrow, standalone lookup (unlike ``paths.py``'s multi-convention resolver) --
these files are produced by exactly one place, ``docs_local/sync_a5_area_p90.sh``,
at ``<config.a5_area_p90_dir>/<exp>_<mask_source>.csv``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .config import ReportFiguresConfig

MASK_SOURCES = ("cellpose_sam", "scportrait")


def resolve_area_p90_csv(
    exp: str, source: str, config: ReportFiguresConfig
) -> Optional[Path]:
    """Path to the extracted ``area,percentile_90`` CSV for ``exp``/``source``, or
    ``None`` if it hasn't been synced (see ``docs_local/sync_a5_area_p90.sh``)."""
    candidate = Path(config.a5_area_p90_dir) / f"{exp}_{source}.csv"
    return candidate if candidate.exists() else None
