"""Group C — data-quality figures (C1-C4): the acquisition/QC findings that condition
the modeling (timepoint-11 artefact, well clustering, Gabor uninformativeness, the
z=3 sparse-slice tail)."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import ReportFiguresConfig
from .data_loading import mean_area_by_z
from .manifest import FigureStatus
from .paths import resolve_experiment_paths
from .plate import load_plate_layout, plate_grid_shape, well_to_drug_dose
from .theme import EXPERIMENT_COLORS, save_fig

import matplotlib.pyplot as plt

GABOR_FEATURES = ("gabor_mean", "gabor_std")
SIZE_FEATURES = ("area", "perimeter", "feret_diameter", "major_axis", "minor_axis")


def _load_extreme_reports(config: ReportFiguresConfig) -> Dict[str, pd.DataFrame]:
    reports: Dict[str, pd.DataFrame] = {}
    for exp in config.experiments:
        paths = resolve_experiment_paths(exp, config)
        if paths.data_quality_csv is None:
            continue
        df = pd.read_csv(paths.data_quality_csv)
        reports[exp] = df[df["source"] == exp]
    return reports


def figure_c1(config: ReportFiguresConfig) -> FigureStatus:
    """Timepoint-11 extreme-value effect: enrichment vs timepoint, per experiment."""
    reports = _load_extreme_reports(config)
    missing = [exp for exp in config.experiments if exp not in reports]

    if not reports:
        return FigureStatus(
            figure_id="C1",
            title="Timepoint-11 extreme-value effect",
            status="blocked",
            missing=missing,
            note="No extreme_value_report.csv found for any experiment.",
        )

    # The brief's "~3.1-3.5x at timepoint 11" finding is about p95-extreme clustering
    # in the mCherry *target* itself (an acquisition artefact), not a morphology
    # feature like area -- confirmed against extreme_value_report.csv (percentile_95
    # rows show ~3.09-3.55x at timepoint 11; area rows only ~1.0-1.3x).
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for exp, report in reports.items():
        subset = report[
            (report["group_type"] == "timepoint")
            & (report["value_column"] == "percentile_95")
        ].copy()
        if subset.empty:
            continue
        subset["group_value"] = subset["group_value"].astype(int)
        subset = subset.sort_values("group_value")
        ax.plot(
            subset["group_value"],
            subset["enrichment"],
            marker="o",
            label=exp,
            color=EXPERIMENT_COLORS.get(exp),
        )
    ax.axhline(1.0, color="grey", linewidth=1, linestyle="--")
    for tp in (11, 21):
        ax.axvline(tp, color="firebrick", alpha=0.2)
    ax.annotate(
        "timepoint 11", xy=(11, ax.get_ylim()[1]), ha="center", va="top", fontsize=8
    )
    ax.set_xlabel("timepoint")
    ax.set_ylabel("extreme-value enrichment\n(percentile_95, vs overall rate)")
    ax.set_title("Timepoint-11 extreme-value effect (mCherry p95)")
    ax.legend()
    fig.tight_layout()

    written = save_fig(
        fig,
        config.output_dir,
        "C1_timepoint11_extreme_effect",
        config.formats,
        config.dpi,
    )

    status = "generated" if not missing else "partial"
    return FigureStatus(
        figure_id="C1",
        title="Timepoint-11 extreme-value effect",
        status=status,
        output_paths=[str(p) for p in written],
        missing=missing,
    )


def figure_c2(config: ReportFiguresConfig) -> FigureStatus:
    """Well-level extreme enrichment, as a plate heatmap per experiment."""
    reports = _load_extreme_reports(config)
    missing = [exp for exp in config.experiments if exp not in reports]

    if not reports:
        return FigureStatus(
            figure_id="C2",
            title="Well-level extreme enrichment",
            status="blocked",
            missing=missing,
            note="No extreme_value_report.csv found for any experiment.",
        )

    layout = load_plate_layout(config.plate_layout_json)
    row_labels, col_numbers = plate_grid_shape(layout)

    n_present = len(reports)
    fig, axes = plt.subplots(1, n_present, figsize=(6 * n_present, 5), squeeze=False)
    for idx, (exp, report) in enumerate(reports.items()):
        ax = axes[0][idx]
        subset = report[
            (report["group_type"] == "sample_id") & (report["value_column"] == "area")
        ]
        grid = np.full((len(row_labels), len(col_numbers)), np.nan)
        for _, row in subset.iterrows():
            well = str(row["group_value"])
            try:
                drug, _ = well_to_drug_dose(well, layout)
            except ValueError:
                continue
            row_letter = well[0].upper()
            col_number = int(well[1:])
            if row_letter in row_labels and col_number in col_numbers:
                grid[row_labels.index(row_letter), col_numbers.index(col_number)] = row[
                    "enrichment"
                ]
        im = ax.imshow(
            grid,
            cmap="RdBu_r",
            vmin=0,
            vmax=np.nanmax(grid) if np.any(~np.isnan(grid)) else 1,
        )
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels)
        ax.set_xticks(range(len(col_numbers)))
        ax.set_xticklabels(col_numbers, fontsize=6, rotation=90)
        ax.set_title(exp)
        fig.colorbar(im, ax=ax, label="extreme-value enrichment", shrink=0.7)
    fig.suptitle("Well-level extreme-value enrichment")
    fig.tight_layout()

    written = save_fig(
        fig, config.output_dir, "C2_well_level_enrichment", config.formats, config.dpi
    )

    status = "generated" if not missing else "partial"
    return FigureStatus(
        figure_id="C2",
        title="Well-level extreme enrichment",
        status=status,
        output_paths=[str(p) for p in written],
        missing=missing,
    )


def figure_c3(config: ReportFiguresConfig) -> FigureStatus:
    """Gabor texture is uninformative: |rho| bar (i) + per-z extreme-count (ii)."""
    rho_rows = []
    missing_rho: List[str] = []
    for exp in config.experiments:
        paths = resolve_experiment_paths(exp, config)
        if paths.univariate_correlations_csv is None:
            missing_rho.append(exp)
            continue
        univariate = pd.read_csv(paths.univariate_correlations_csv)
        pooled = univariate[univariate["scope"] == "pooled"]
        for feature, kind in [(f, "gabor") for f in GABOR_FEATURES] + [
            (f, "size") for f in SIZE_FEATURES
        ]:
            subset = pooled[pooled["feature"] == feature]
            if subset.empty:
                continue
            rho_rows.append(
                {
                    "experiment": exp,
                    "feature": feature,
                    "kind": kind,
                    "abs_rho": subset["rho"].abs().mean(),
                }
            )

    if not rho_rows:
        return FigureStatus(
            figure_id="C3",
            title="Gabor texture is uninformative",
            status="blocked",
            missing=missing_rho,
            note="No univariate_correlations.csv found for any experiment.",
        )

    rho_table = pd.DataFrame(rho_rows)

    reports = _load_extreme_reports(config)
    missing_z: List[str] = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax_rho, ax_z = axes

    pivot = rho_table.pivot_table(
        index="feature", columns="experiment", values="abs_rho"
    )
    pivot = pivot.reindex(list(GABOR_FEATURES) + list(SIZE_FEATURES))
    pivot.plot(kind="barh", ax=ax_rho)
    ax_rho.set_xlabel("mean |Spearman rho| with mCherry percentiles (pooled)")
    ax_rho.set_title("Gabor vs size features")
    ax_rho.invert_yaxis()

    for exp in config.experiments:
        if exp not in reports:
            missing_z.append(exp)
            continue
        report = reports[exp]
        subset = report[
            (report["group_type"] == "z_index")
            & (report["value_column"].isin(GABOR_FEATURES))
        ].copy()
        if subset.empty:
            continue
        subset["group_value"] = subset["group_value"].astype(int)
        agg = subset.groupby("group_value")["n_extreme"].sum().sort_index()
        ax_z.plot(
            agg.index,
            agg.values,
            marker="o",
            label=exp,
            color=EXPERIMENT_COLORS.get(exp),
        )
    ax_z.set_xlabel("z-index")
    ax_z.set_ylabel("extreme-value count (Gabor features)")
    ax_z.set_title("Gabor extremes vs z-slice")
    ax_z.legend()
    fig.tight_layout()

    written = save_fig(
        fig, config.output_dir, "C3_gabor_uninformative", config.formats, config.dpi
    )

    missing = sorted(set(missing_rho) | set(missing_z))
    status = "generated" if not missing else "partial"
    return FigureStatus(
        figure_id="C3",
        title="Gabor texture is uninformative",
        status=status,
        output_paths=[str(p) for p in written],
        missing=missing,
    )


def figure_c4(config: ReportFiguresConfig) -> FigureStatus:
    """Cell count and mean cell-size vs z-slice (the sparse-slice-tail explanation)."""
    reports = _load_extreme_reports(config)
    missing: List[str] = []

    def _mark_missing(exp: str) -> None:
        if exp not in missing:
            missing.append(exp)

    fig, (ax_count, ax_area) = plt.subplots(1, 2, figsize=(11, 4.5))
    any_data = False
    for exp in config.experiments:
        color = EXPERIMENT_COLORS.get(exp)
        report = reports.get(exp)
        counts: Optional[pd.Series] = None
        if report is not None:
            subset = report[
                (report["group_type"] == "z_index") & (report["value_column"] == "area")
            ].copy()
            if not subset.empty:
                subset["group_value"] = subset["group_value"].astype(int)
                counts = subset.set_index("group_value")["n"].sort_index()
                ax_count.plot(
                    counts.index, counts.values, marker="o", label=exp, color=color
                )
                any_data = True
        if counts is None:
            _mark_missing(exp)

        paths = resolve_experiment_paths(exp, config)
        if paths.feature_dir is None:
            _mark_missing(exp)
            continue
        mean_area = mean_area_by_z(paths.feature_dir)
        if mean_area is not None and not mean_area.empty:
            ax_area.plot(
                mean_area.index, mean_area.values, marker="o", label=exp, color=color
            )
            any_data = True

    if not any_data:
        plt.close(fig)
        return FigureStatus(
            figure_id="C4",
            title="Cells and cell-size vs z-slice",
            status="blocked",
            missing=missing,
            note="No data_quality or raw feature data found for any experiment.",
        )

    ax_count.set_xlabel("z-index")
    ax_count.set_ylabel("cell count")
    ax_count.set_title("Cell count vs z-slice")
    ax_count.legend()
    ax_area.set_xlabel("z-index")
    ax_area.set_ylabel("mean area")
    ax_area.set_title("Mean cell area vs z-slice")
    ax_area.legend()
    fig.tight_layout()

    written = save_fig(
        fig, config.output_dir, "C4_cells_and_size_vs_z", config.formats, config.dpi
    )

    status = "generated" if not missing else "partial"
    return FigureStatus(
        figure_id="C4",
        title="Cells and cell-size vs z-slice",
        status=status,
        output_paths=[str(p) for p in written],
        missing=missing,
    )
