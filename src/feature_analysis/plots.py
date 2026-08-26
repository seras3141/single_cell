"""Low-level matplotlib rendering for feature-variation-over-time outputs.

Kept separate from computation (feature_trajectories.py) and deliberately NOT routed through
``feature_visualization/visualizer.py`` (which performs dimensionality reduction); these are
plain per-feature time-series renders. The Agg backend is selected once at import (before
pyplot is imported) so headless SLURM runs are safe.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence, Union

import matplotlib

matplotlib.use("Agg")  # once, before pyplot import — headless-safe
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

logger = logging.getLogger(__name__)


def plot_trajectories_grid(
    traj: pd.DataFrame,
    out_png: Union[str, Path],
    dmso_well: Optional[str] = None,
    features: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
) -> None:
    """Grid of per-feature median-vs-timepoint curves, one line per well, DMSO bold/black."""
    feats = list(features) if features is not None else sorted(traj["feature"].unique())
    n = len(feats)
    ncol = min(4, n) or 1
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow), squeeze=False)
    for i, feat in enumerate(feats):
        ax = axes[i // ncol][i % ncol]
        sub = traj[traj["feature"] == feat]
        for well, g in sub.groupby("sample_id"):
            g = g.sort_values("ti")
            is_dmso = dmso_well is not None and str(well).upper() == str(dmso_well).upper()
            ax.plot(
                g["ti"], g["median"],
                marker="o", markersize=2.5,
                linewidth=2.4 if is_dmso else 1.0,
                color="black" if is_dmso else None,
                zorder=5 if is_dmso else 2,
                label=f"{well} (DMSO)" if is_dmso else str(well),
            )
        ax.set_title(feat, fontsize=10)
        ax.set_xlabel("ti", fontsize=8)
        ax.tick_params(labelsize=7)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    # Collect legend entries across ALL subplots (a well absent from feature 0 but present
    # elsewhere must still appear), de-duplicated by label.
    seen: dict = {}
    for row in axes:
        for ax in row:
            for handle, label in zip(*ax.get_legend_handles_labels()):
                seen.setdefault(label, handle)
    if seen:
        fig.legend(list(seen.values()), list(seen.keys()), fontsize=7,
                   ncol=min(5, len(seen)), loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title or "feature trajectories (median vs timepoint)", fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_overall_divergence(
    div: pd.DataFrame,
    out_png: Union[str, Path],
    title: Optional[str] = None,
) -> None:
    """Overall divergence-from-DMSO (mean |std-shift|) vs timepoint, one line per drug well."""
    overall = div[div["feature"] == "__overall__"].copy()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for well, g in overall.groupby("sample_id"):
        g = g.sort_values("ti")
        ax.plot(g["ti"], g["std_shift"], marker="o", markersize=3, linewidth=1.3, label=str(well))
    ax.set_xlabel("timepoint index (×10 min)")
    ax.set_ylabel("overall divergence (mean |std-shift| vs DMSO)")
    ax.set_title(title or "divergence from DMSO over time")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
