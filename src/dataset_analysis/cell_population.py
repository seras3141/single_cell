"""Segmentation-only cell-population-over-time analysis.

Computes, per (well, timepoint), the distinct physical-cell count and a coverage
(confluence) proxy from the segmentation-derived ``instance_metrics.csv`` table
(``cell_id``/``area``/``z_index`` come from the ``final_2d`` tracked masks; the
mCherry intensity columns are deliberately ignored). This is the productionised
successor to the ``docs/_phase6_scratch`` "distinct-cell collapse" scripts.

See ``docs/dataset_analysis/plan_cell_population_over_time.md``.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import numpy as np
import pandas as pd

from src.dataset_analysis.layout import get_well_annotation

logger = logging.getLogger(__name__)

_WELL_RE = re.compile(r"^([A-Za-z]+)(\d+)$")

#: mCherry-free columns required from ``instance_metrics.csv``.
REQUIRED_COLUMNS = ("sample_id", "timepoint", "z_index", "cell_id", "area")


def _parse_well(well: str) -> tuple[Optional[str], Optional[int]]:
    """Split a well id like ``"N11"`` into ``("N", 11)``; ``(None, None)`` if unparseable."""
    match = _WELL_RE.match(str(well))
    if not match:
        return None, None
    return match.group(1).upper(), int(match.group(2))


def compute_cell_population(
    instance_metrics_csv: Union[str, Path],
    fov_pixels: int,
    layout: Optional[Mapping[str, Any]] = None,
    dmso_well: Optional[str] = None,
) -> pd.DataFrame:
    """Per-(well, timepoint) population metrics from a segmentation metrics table.

    Args:
        instance_metrics_csv: Path to ``mcherry_metrics/<model>/instance_metrics.csv``.
            Only the segmentation columns in :data:`REQUIRED_COLUMNS` are read.
        fov_pixels: Field-of-view size in pixels (H*W) for the coverage denominator.
        layout: Optional loaded plate layout (see ``dataset_analysis.layout``) used to
            annotate each well with its drug/control condition.
        dmso_well: Optional well id (e.g. ``"N11"``) flagged as the DMSO reference.

    Returns:
        DataFrame with one row per (``sample_id``, ``timepoint``): ``n_cells``
        (distinct ``cell_id``), ``coverage_fraction`` (mean over z of
        Σarea/``fov_pixels``), integer ``ti``, and — when available — ``drug`` and
        ``is_dmso`` columns. Sorted by well then timepoint.
    """
    if fov_pixels <= 0:
        raise ValueError(f"fov_pixels must be positive, got {fov_pixels}")

    df = pd.read_csv(
        instance_metrics_csv,
        usecols=lambda c: c in set(REQUIRED_COLUMNS),
    )
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"{instance_metrics_csv} missing required columns: {sorted(missing)}"
        )

    grp = df.groupby(["sample_id", "timepoint"], sort=False)
    n_cells = grp["cell_id"].nunique().rename("n_cells")

    # coverage: foreground pixels per z-slice / FOV, averaged over the z-slices present
    per_z = (
        df.groupby(["sample_id", "timepoint", "z_index"], sort=False)["area"].sum()
        / float(fov_pixels)
    )
    coverage = (
        per_z.groupby(["sample_id", "timepoint"], sort=False)
        .mean()
        .rename("coverage_fraction")
    )

    out = pd.concat([n_cells, coverage], axis=1).reset_index()
    out["ti"] = out["timepoint"].map(
        lambda s: int(s) if str(s).isdigit() else -1
    )

    if layout is not None:
        def _drug(well: str) -> Optional[str]:
            row, col = _parse_well(well)
            if row is None:
                return None
            try:
                ann = get_well_annotation(row, col, layout)
            except Exception:  # unknown row/col — leave unannotated
                return None
            return ann.get("drug") or ann.get("content")

        out["drug"] = out["sample_id"].map(_drug)

    if dmso_well is not None:
        out["is_dmso"] = out["sample_id"].str.upper() == str(dmso_well).upper()

    return out.sort_values(["sample_id", "ti"]).reset_index(drop=True)


def plot_population(
    df: pd.DataFrame,
    metric: str,
    out_png: Union[str, Path],
    title: Optional[str] = None,
    dmso_well: Optional[str] = None,
) -> None:
    """Per-well curves of ``metric`` vs timepoint, with the DMSO well drawn bold/black."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if metric not in df.columns:
        raise ValueError(f"metric {metric!r} not in dataframe columns {list(df.columns)}")

    fig, ax = plt.subplots(figsize=(10, 6))
    for well, g in df.groupby("sample_id"):
        g = g.sort_values("ti")
        is_dmso = dmso_well is not None and str(well).upper() == str(dmso_well).upper()
        ax.plot(
            g["ti"],
            g[metric],
            marker="o",
            markersize=3,
            linewidth=2.6 if is_dmso else 1.2,
            color="black" if is_dmso else None,
            zorder=5 if is_dmso else 2,
            label=f"{well} (DMSO)" if is_dmso else str(well),
        )
    ax.set_xlabel("timepoint index (×10 min)")
    ax.set_ylabel(metric)
    ax.set_title(title or metric)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
