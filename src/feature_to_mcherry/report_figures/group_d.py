"""Group D — context and design figures (D1-D4): study-design schematic, target
dynamic range, scPortrait PCA scree (blocked), and per-experiment cohort size."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd

from .config import ReportFiguresConfig
from .manifest import FigureStatus
from .paths import find_sibling_dir_case_insensitive, resolve_experiment_paths
from .theme import EXPERIMENT_COLORS, save_fig

import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def figure_d1(config: ReportFiguresConfig) -> FigureStatus:
    """Study-design schematic: no data dependency, pure diagram."""
    stages = [
        "Brightfield\nmicroscopy",
        "Cellpose\nsegmentation",
        "Per-cell\nmorphology features\n+ mCherry percentiles*",
        "Feature -> percentile\nmapping (Ridge /\nLightGBM)",
        "Dose-response\n(drug/dose per well)",
    ]
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.set_xlim(0, len(stages))
    ax.set_ylim(0, 1)
    ax.axis("off")

    for i, stage in enumerate(stages):
        box = mpatches.FancyBboxPatch(
            (i + 0.05, 0.3),
            0.9,
            0.4,
            boxstyle="round,pad=0.02",
            linewidth=1.5,
            edgecolor="#0072B2",
            facecolor="#E8F1F8",
        )
        ax.add_patch(box)
        ax.text(i + 0.5, 0.5, stage, ha="center", va="center", fontsize=9)
        if i < len(stages) - 1:
            ax.annotate(
                "",
                xy=(i + 1.05, 0.5),
                xytext=(i + 0.95, 0.5),
                arrowprops=dict(arrowstyle="->", color="black", linewidth=1.5),
            )

    ax.text(
        len(stages) / 2,
        0.05,
        "*mCherry percentiles are privileged information -- available for training "
        "the mapping, not for the eventual brightfield-only inference target.",
        ha="center",
        va="center",
        fontsize=8,
        style="italic",
    )
    ax.set_title("Study design: brightfield -> activity -> dose-response")

    written = save_fig(
        fig, config.output_dir, "D1_study_design_schematic", config.formats, config.dpi
    )

    return FigureStatus(
        figure_id="D1",
        title="Study design schematic",
        status="generated",
        output_paths=[str(p) for p in written],
    )


def figure_d2(config: ReportFiguresConfig) -> FigureStatus:
    """Target distribution montage: reuses each experiment's already-generated
    ``target_distributions.png`` (from
    ``informativeness.plots.plot_target_distributions``) rather than recomputing
    histograms from raw targets."""
    images = []
    missing: List[str] = []
    for exp in config.experiments:
        paths = resolve_experiment_paths(exp, config)
        if paths.informativeness_dir is None:
            missing.append(exp)
            continue
        image_path = paths.informativeness_dir / "figures" / "target_distributions.png"
        if not image_path.exists():
            missing.append(exp)
            continue
        images.append((exp, image_path))

    if not images:
        return FigureStatus(
            figure_id="D2",
            title="Target distribution and dynamic range",
            status="blocked",
            missing=missing,
            note="No target_distributions.png found for any experiment "
            "(run the morphology-informativeness gate first).",
        )

    fig, axes = plt.subplots(
        1, len(images), figsize=(5 * len(images), 4.5), squeeze=False
    )
    for idx, (exp, image_path) in enumerate(images):
        ax = axes[0][idx]
        ax.imshow(mpimg.imread(image_path))
        ax.set_title(exp)
        ax.axis("off")
    fig.suptitle(
        "mCherry target distributions (per-experiment, reused from the "
        "informativeness gate)"
    )
    fig.tight_layout()

    written = save_fig(
        fig,
        config.output_dir,
        "D2_target_distribution_montage",
        config.formats,
        config.dpi,
    )

    status = "generated" if not missing else "partial"
    return FigureStatus(
        figure_id="D2",
        title="Target distribution and dynamic range",
        status=status,
        output_paths=[str(p) for p in written],
        missing=missing,
        note="Montage of each experiment's existing target_distributions.png "
        "(reused, not recomputed).",
    )


def figure_d3(config: ReportFiguresConfig) -> FigureStatus:
    """scPortrait embedding PCA scree: blocked unless a saved explained-variance
    array is found under a synced scPortrait results dir."""
    ftm_dir = Path(config.feature_to_mcherry_dir)
    checked = []
    for exp in config.scportrait_experiments:
        scportrait_dir = find_sibling_dir_case_insensitive(
            ftm_dir.parent, f"{ftm_dir.name}_{exp}_scportrait"
        )
        checked.append(f"{ftm_dir.name}_{exp}_scportrait")
        if scportrait_dir is None:
            continue
        candidates = list(scportrait_dir.glob("*pca*")) + list(
            scportrait_dir.glob("*explained_variance*")
        )
        if candidates:
            return FigureStatus(
                figure_id="D3",
                title="scPortrait embedding PCA scree",
                status="blocked",
                note=f"Found candidate artifact(s) {candidates} but no plotting logic "
                "for them yet -- implement once a real artifact is confirmed.",
            )

    return FigureStatus(
        figure_id="D3",
        title="scPortrait embedding PCA scree",
        status="blocked",
        missing=config.scportrait_experiments,
        note="No saved PCA explained-variance artifact found under any scPortrait "
        f"results dir (checked: {checked}). Would require a full re-fit on raw "
        "embeddings. Not implemented this pass, per the brief's instruction to mark "
        "rather than invent.",
    )


def figure_d4(config: ReportFiguresConfig) -> FigureStatus:
    """Per-experiment cohort size: live count from instance_metrics.csv where
    locally available, else the brief's documented count."""
    rows = []
    fallback_used: List[str] = []
    missing: List[str] = []
    for exp in config.experiments:
        paths = resolve_experiment_paths(exp, config)
        count: Optional[int] = None
        if paths.instance_metrics_csv is not None:
            count = len(pd.read_csv(paths.instance_metrics_csv, usecols=[0]))
        if count is None:
            count = config.cohort_sizes.get(exp)
            if count is not None:
                fallback_used.append(exp)
        if count is not None:
            rows.append({"experiment": exp, "n_cells": count})
        else:
            missing.append(exp)

    if not rows:
        return FigureStatus(
            figure_id="D4",
            title="Per-experiment cohort size",
            status="blocked",
            missing=missing,
            note="No instance_metrics.csv found locally and no cohort_sizes set.",
        )

    table = pd.DataFrame(rows).sort_values("n_cells", ascending=True)
    fig, ax = plt.subplots(figsize=(7, max(3, 0.6 * len(table))))
    colors = [EXPERIMENT_COLORS.get(exp) for exp in table["experiment"]]
    ax.barh(table["experiment"], table["n_cells"], color=colors)
    for y, (exp, n) in enumerate(zip(table["experiment"], table["n_cells"])):
        label = f"{n:,}" + (" (fallback)" if exp in fallback_used else "")
        ax.text(n, y, f" {label}", va="center", fontsize=8)
    ax.set_xlabel("per-cell count (mCherry-metric rows)")
    ax.set_title("Per-experiment cohort size")
    fig.tight_layout()

    written = save_fig(
        fig, config.output_dir, "D4_cohort_size", config.formats, config.dpi
    )

    note = (
        f"Used the configured fallback count (not a live CSV) for: {fallback_used}"
        if fallback_used
        else ""
    )
    status = "generated" if not missing else "partial"
    return FigureStatus(
        figure_id="D4",
        title="Per-experiment cohort size",
        status=status,
        output_paths=[str(p) for p in written],
        missing=missing,
        note=note,
    )
