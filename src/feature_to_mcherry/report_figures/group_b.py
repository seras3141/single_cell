"""Group B — mechanism figures (B1-B4): which morphology axes carry signal, whether
it survives intensity-proxy removal, what the nonlinear floor relies on, and what an
R^2~0.2 Ridge fit actually looks like."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from ..informativeness.plots import plot_correlation_heatmap
from .config import ReportFiguresConfig
from .manifest import FigureStatus
from .paths import resolve_experiment_paths
from .theme import EXPERIMENT_COLORS, ordered_percentiles, save_fig

import matplotlib.pyplot as plt


def figure_b1(config: ReportFiguresConfig) -> FigureStatus:
    """Univariate feature<->percentile correlation heatmap, per experiment.

    Reuses ``informativeness.plots.plot_correlation_heatmap`` directly (it already
    implements the pooled-scope pivot + heatmap) rather than reimplementing the
    heatmap math; that function is self-contained (renders and saves its own
    figure), so this writes one heatmap per experiment into per-experiment
    subdirectories instead of forcing all experiments into a single montage.
    Note: the reused function's own PNG dpi (150) and lack of PDF output are an
    accepted tradeoff of reuse-over-reimplementation, not this module's convention.
    """
    written: List[str] = []
    missing: List[str] = []
    for exp in config.experiments:
        paths = resolve_experiment_paths(exp, config)
        if paths.univariate_correlations_csv is None:
            missing.append(exp)
            continue
        univariate_df = pd.read_csv(paths.univariate_correlations_csv)
        exp_dir = Path(config.output_dir) / "B1_correlation_heatmaps" / exp
        exp_dir.mkdir(parents=True, exist_ok=True)
        written.extend(str(p) for p in plot_correlation_heatmap(univariate_df, exp_dir))

    if not written:
        return FigureStatus(
            figure_id="B1",
            title="Univariate feature-percentile correlation heatmap",
            status="blocked",
            missing=missing,
            note="No univariate_correlations.csv found for any experiment.",
        )

    status = "generated" if not missing else "partial"
    return FigureStatus(
        figure_id="B1",
        title="Univariate feature-percentile correlation heatmap",
        status=status,
        output_paths=written,
        missing=missing,
        note="One heatmap per experiment (150 dpi PNG + HTML, via the reused "
        "informativeness plotting function) rather than a single montage.",
    )


def figure_b2(config: ReportFiguresConfig) -> FigureStatus:
    """Signal survives intensity-proxy removal: nonlinear floor R^2, with vs
    without the suspect (intensity-proxy) features."""
    rows = []
    missing: List[str] = []
    for exp in config.experiments:
        paths = resolve_experiment_paths(exp, config)
        if paths.floor_metrics_csv is None:
            missing.append(exp)
            continue
        floor = pd.read_csv(paths.floor_metrics_csv)
        nonlinear = floor[floor["model"] != "ridge"]
        for _, row in nonlinear.iterrows():
            rows.append(
                {
                    "experiment": exp,
                    "target": row["target"],
                    "variant": row["variant"],
                    "r2": row["r2"],
                }
            )

    if not rows:
        return FigureStatus(
            figure_id="B2",
            title="Signal survives intensity-proxy removal",
            status="blocked",
            missing=missing,
            note="No floor_metrics.csv found for any experiment.",
        )

    table = pd.DataFrame(rows)
    percentiles = ordered_percentiles(table["target"].unique())
    if not percentiles:
        return FigureStatus(
            figure_id="B2",
            title="Signal survives intensity-proxy removal",
            status="blocked",
            missing=missing,
            note=(
                "No percentile targets in floor_metrics.csv; found "
                f"{sorted(map(str, table['target'].unique()))}."
            ),
        )

    fig, axes = plt.subplots(
        1, len(percentiles), figsize=(4 * len(percentiles), 4), squeeze=False
    )
    for idx, target in enumerate(percentiles):
        ax = axes[0][idx]
        pivot = table[table["target"] == target].pivot(
            index="experiment", columns="variant", values="r2"
        )
        pivot = pivot.reindex(columns=["with_suspect", "without_suspect"])
        pivot.plot(
            kind="bar",
            ax=ax,
            color={"with_suspect": "#0072B2", "without_suspect": "#009E73"},
        )
        ax.set_title(target)
        ax.set_ylabel("nonlinear floor R²" if idx == 0 else "")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("Signal survives intensity-proxy removal (nonlinear floor)")
    fig.tight_layout()

    written = save_fig(
        fig, config.output_dir, "B2_intensity_proxy_removal", config.formats, config.dpi
    )

    status = "generated" if not missing else "partial"
    return FigureStatus(
        figure_id="B2",
        title="Signal survives intensity-proxy removal",
        status=status,
        output_paths=[str(p) for p in written],
        missing=missing,
    )


def figure_b3(config: ReportFiguresConfig, top_k: int = 8) -> FigureStatus:
    """Top-N nonlinear-floor feature importances, one panel per experiment."""
    importances: Dict[str, pd.DataFrame] = {}
    missing: List[str] = []
    for exp in config.experiments:
        paths = resolve_experiment_paths(exp, config)
        if paths.feature_importances_csv is None:
            missing.append(exp)
            continue
        importances[exp] = pd.read_csv(paths.feature_importances_csv)

    if not importances:
        return FigureStatus(
            figure_id="B3",
            title="Feature importance for the nonlinear floor",
            status="blocked",
            missing=missing,
            note="No feature_importances.csv found for any experiment "
            "(requires rerunning the morphology-informativeness gate after the "
            "feature-importance pipeline addition).",
        )

    fig, axes = plt.subplots(
        1, len(importances), figsize=(5 * len(importances), 4.5), squeeze=False
    )
    for idx, (exp, table) in enumerate(importances.items()):
        ax = axes[0][idx]
        top = (
            table.groupby("feature")["importance_mean"]
            .mean()
            .sort_values(ascending=True)
            .tail(top_k)
        )
        ax.barh(top.index, top.values, color=EXPERIMENT_COLORS.get(exp))
        ax.set_title(exp)
        ax.set_xlabel("mean importance (nonlinear floor, avg. over percentiles)")
    fig.suptitle("Top feature importances, nonlinear floor")
    fig.tight_layout()

    written = save_fig(
        fig, config.output_dir, "B3_feature_importances", config.formats, config.dpi
    )

    status = "generated" if not missing else "partial"
    return FigureStatus(
        figure_id="B3",
        title="Feature importance for the nonlinear floor",
        status=status,
        output_paths=[str(p) for p in written],
        missing=missing,
    )


def _pick_representative_experiments(
    config: ReportFiguresConfig, available: List[str]
) -> List[str]:
    """Pick a high-floor and a low-floor experiment among ``available``, by mean
    Ridge R^2 from baseline_ladder.csv where resolvable; else just the first two."""
    scored = []
    for exp in available:
        paths = resolve_experiment_paths(exp, config)
        if paths.baseline_ladder_csv is None:
            continue
        ladder = pd.read_csv(paths.baseline_ladder_csv)
        ridge = ladder[ladder["model"] == "ridge"]
        if not ridge.empty:
            scored.append((exp, ridge["r2"].mean()))

    if len(scored) >= 2:
        scored.sort(key=lambda item: item[1])
        return (
            [scored[0][0], scored[-1][0]]
            if scored[0][0] != scored[-1][0]
            else [scored[0][0]]
        )
    return available[:2]


def figure_b4(config: ReportFiguresConfig) -> FigureStatus:
    """Predicted-vs-actual + residual-vs-fitted (Ridge), for one high-floor and one
    low-floor experiment, from the persisted out-of-fold predictions."""
    available = []
    for exp in config.experiments:
        paths = resolve_experiment_paths(exp, config)
        oof_path = None
        if paths.baseline_ladder_csv is not None:
            candidate = paths.baseline_ladder_csv.parent / "oof_predictions.csv"
            if candidate.exists():
                oof_path = candidate
        if oof_path is not None:
            available.append(exp)

    missing = [exp for exp in config.experiments if exp not in available]

    if not available:
        return FigureStatus(
            figure_id="B4",
            title="Predicted-vs-actual and residual calibration (Ridge)",
            status="blocked",
            missing=missing,
            note="No oof_predictions.csv found for any experiment (requires "
            "rerunning feature_to_mcherry after the OOF-persistence addition).",
        )

    chosen = _pick_representative_experiments(config, available)

    fig, axes = plt.subplots(
        2, len(chosen), figsize=(5 * len(chosen), 8), squeeze=False
    )
    for col, exp in enumerate(chosen):
        paths = resolve_experiment_paths(exp, config)
        oof_path = paths.baseline_ladder_csv.parent / "oof_predictions.csv"
        oof = pd.read_csv(oof_path)
        # One representative target: the highest-tau percentile present.
        target = sorted(oof["target_name"].unique())[-1]
        subset = oof[oof["target_name"] == target]

        # Robust (1st-99th percentile) axis limits: a handful of extreme mCherry
        # outliers otherwise dominate the range and squash the bulk of the fit into
        # an unreadable corner. bins="log" for the same reason on the color scale.
        lo, hi = np.percentile(subset[["y_true", "y_pred"]].to_numpy(), [1, 99])
        pad = 0.05 * (hi - lo)

        ax_scatter = axes[0][col]
        ax_scatter.hexbin(
            subset["y_true"],
            subset["y_pred"],
            gridsize=40,
            mincnt=1,
            cmap="viridis",
            bins="log",
            extent=(lo - pad, hi + pad, lo - pad, hi + pad),
        )
        ax_scatter.plot(
            [lo - pad, hi + pad],
            [lo - pad, hi + pad],
            color="firebrick",
            linewidth=1,
            linestyle="--",
        )
        ax_scatter.set_xlim(lo - pad, hi + pad)
        ax_scatter.set_ylim(lo - pad, hi + pad)
        ax_scatter.set_xlabel(f"true {target}")
        ax_scatter.set_ylabel("predicted (Ridge, OOF)")
        ax_scatter.set_title(exp)

        residuals = subset["y_pred"] - subset["y_true"]
        resid_lo, resid_hi = np.percentile(residuals, [1, 99])
        resid_pad = 0.05 * (resid_hi - resid_lo)
        ax_resid = axes[1][col]
        ax_resid.hexbin(
            subset["y_pred"],
            residuals,
            gridsize=40,
            mincnt=1,
            cmap="viridis",
            bins="log",
            extent=(
                lo - pad,
                hi + pad,
                resid_lo - resid_pad,
                resid_hi + resid_pad,
            ),
        )
        ax_resid.axhline(0, color="firebrick", linewidth=1, linestyle="--")
        ax_resid.set_xlim(lo - pad, hi + pad)
        ax_resid.set_ylim(resid_lo - resid_pad, resid_hi + resid_pad)
        ax_resid.set_xlabel("predicted (Ridge, OOF)")
        ax_resid.set_ylabel("residual (pred - true)")
    fig.suptitle("Predicted-vs-actual and residual calibration (Ridge, out-of-fold)")
    fig.tight_layout()

    written = save_fig(
        fig, config.output_dir, "B4_predicted_vs_actual", config.formats, config.dpi
    )

    status = "generated" if not missing else "partial"
    return FigureStatus(
        figure_id="B4",
        title="Predicted-vs-actual and residual calibration (Ridge)",
        status=status,
        output_paths=[str(p) for p in written],
        missing=missing,
    )
