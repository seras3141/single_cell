"""Shared matplotlib styling and figure-writing helpers for the report figures.

Forces the ``Agg`` backend before importing ``pyplot`` (headless-safe, no display, no
``plt.show()``) — same precaution as ``informativeness/plots.py``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

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

# mCherry percentile target columns, raw (``percentile_<N>``) or DMSO-normalized
# (``z_percentile_<N>``, added by data/normalize.py). Deliberately NOT anchored to a
# fixed list of raw names: see ordered_percentiles.
_PERCENTILE_TARGET_RE = re.compile(r"^(?:z_)?percentile_(\d+)$")


def ordered_percentiles(available: Iterable[object]) -> List[str]:
    """Percentile target columns present in ``available``, ordered by percentile.

    Derived from the run's ACTUAL target names rather than a hard-coded list, because a
    DMSO-normalized run models ``z_percentile_<N>`` while an un-normalized one models
    ``percentile_<N>`` (see ``data/normalize.py::apply_dmso_normalization``, which swaps
    in the ``z_``-prefixed names as the effective targets). Filtering against a fixed
    list of raw names silently selected NOTHING on a normalized run, and the figure
    builders then crashed downstream rather than degrading -- dividing by a zero target
    count, or asking for a zero-column subplot grid.

    Note there is no ``percentile_50`` target in this pipeline (see
    ``data/contract.py::TARGET_COLUMNS``), despite the brief's context section
    mentioning p50 descriptively; ordering is by the number itself, so a p50 target
    would simply sort first if one were ever added.

    Parameters
    ----------
    available : iterable
        Candidate names -- a DataFrame's ``.columns``, a ``Series.unique()``, or any
        iterable. Non-percentile entries are ignored.

    Returns
    -------
    list[str]
        Matching names ascending by percentile, e.g. ``["z_percentile_75",
        "z_percentile_90"]``. Empty if ``available`` holds no percentile targets, which
        callers must treat as "cannot draw this figure", not as "draw an empty one".
    """
    matched = []
    for name in available:
        match = _PERCENTILE_TARGET_RE.match(str(name))
        if match:
            # Sort on (number, name): the name breaks ties deterministically if a frame
            # somehow carried both the raw and the z_ variant of one percentile.
            matched.append((int(match.group(1)), str(name)))
    return [name for _, name in sorted(matched)]


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
