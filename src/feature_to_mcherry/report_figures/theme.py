"""Shared matplotlib styling and figure-writing helpers for the report figures.

Forces the ``Agg`` backend before importing ``pyplot`` (headless-safe, no display, no
``plt.show()``) — same precaution as ``informativeness/plots.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

# Okabe-Ito colorblind-safe palette, one fixed color per experiment.
EXPERIMENT_COLORS: Dict[str, str] = {
    "HD1509": "#0072B2",
    "SA110": "#E69F00",
    "HD1883": "#009E73",
    "Ew2-1": "#D55E00",
    "Ew2-2": "#CC79A7",
}

# mCherry percentile targets actually modeled (see data/contract.py::TARGET_COLUMNS —
# there is no percentile_50 target in this pipeline, despite the brief's context
# section mentioning p50 descriptively).
PERCENTILE_ORDER: List[str] = ["percentile_75", "percentile_90", "percentile_95"]


def setup_style() -> None:
    """Apply consistent, publication-appropriate matplotlib rcParams."""
    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "savefig.bbox": "tight",
        }
    )


setup_style()


def save_fig(
    fig: "plt.Figure",
    output_dir: Path,
    stem: str,
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = 300,
) -> List[Path]:
    """Write ``fig`` to ``output_dir/<stem>.<format>`` for each format; close it."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for fmt in formats:
        path = output_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=dpi)
        written.append(path)
    plt.close(fig)
    return written
