"""Orchestration: dispatch each figure ID to its group function, collect statuses
into a manifest, and write it out."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

from . import group_a, group_b, group_c, group_d
from .config import ReportFiguresConfig
from .manifest import FigureStatus, ManifestWriter

logger = logging.getLogger(__name__)

FIGURE_FUNCTIONS: Dict[str, Callable[[ReportFiguresConfig], FigureStatus]] = {
    "A1": group_a.figure_a1,
    "A2": group_a.figure_a2,
    "A3": group_a.figure_a3,
    "A4": group_a.figure_a4,
    "A5": group_a.figure_a5,
    "A6": group_a.figure_a6,
    "B1": group_b.figure_b1,
    "B2": group_b.figure_b2,
    "B3": group_b.figure_b3,
    "B4": group_b.figure_b4,
    "C1": group_c.figure_c1,
    "C2": group_c.figure_c2,
    "C3": group_c.figure_c3,
    "C4": group_c.figure_c4,
    "D1": group_d.figure_d1,
    "D2": group_d.figure_d2,
    "D3": group_d.figure_d3,
    "D4": group_d.figure_d4,
}


def run(
    config: ReportFiguresConfig, figure_ids: Optional[List[str]] = None
) -> ManifestWriter:
    """Generate the requested figures (default: all), writing outputs under
    ``config.output_dir`` and returning the collected manifest.

    A figure that raises is recorded as ``"blocked"`` (with the exception message
    as the note) rather than aborting the whole run -- one bad/missing input must
    not prevent every other figure from being generated.
    """
    ids = figure_ids if figure_ids is not None else list(FIGURE_FUNCTIONS)
    unknown = [figure_id for figure_id in ids if figure_id not in FIGURE_FUNCTIONS]
    if unknown:
        raise ValueError(
            f"Unknown figure id(s): {unknown}; known ids: {list(FIGURE_FUNCTIONS)}"
        )

    manifest = ManifestWriter()
    for figure_id in ids:
        logger.info("Generating figure %s", figure_id)
        try:
            status = FIGURE_FUNCTIONS[figure_id](config)
        except Exception as exc:  # noqa: BLE001 - one figure's failure must not
            # abort the run; every other figure still needs a chance to generate.
            logger.exception("Figure %s failed", figure_id)
            status = FigureStatus(
                figure_id=figure_id,
                title=figure_id,
                status="blocked",
                note=f"Raised {type(exc).__name__}: {exc}",
            )
        manifest.add(status)

    manifest.write(Path(config.output_dir))
    logger.info(
        "Report-figures run complete: %d generated, %d partial, %d blocked",
        sum(1 for e in manifest.entries if e.status == "generated"),
        sum(1 for e in manifest.entries if e.status == "partial"),
        sum(1 for e in manifest.entries if e.status == "blocked"),
    )
    return manifest
