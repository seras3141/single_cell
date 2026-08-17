"""Static (PNG) and interactive (HTML) figures for the informativeness report.

Standalone, modern matplotlib/seaborn + plotly plotting — not coupled to the legacy
``feature_visualization`` API/data-flow contract. The matplotlib ``Agg`` backend is
forced before importing pyplot so figure generation works headless (no display, no
``plt.show()``). The plotly import is guarded: PNGs are always produced; HTML is
skipped with a logged note if plotly is unavailable.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from .univariate import top_associations  # noqa: E402

logger = logging.getLogger(__name__)

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAVE_PLOTLY = True
except ImportError:  # pragma: no cover - exercised via a monkeypatch in tests
    HAVE_PLOTLY = False


def _rolling_trend(x: np.ndarray, y: np.ndarray, window_frac: float = 0.2) -> tuple:
    """Rank-smoothed trend: sort by x, rolling median with a size-adaptive window."""
    order = np.argsort(x)
    x_sorted, y_sorted = x[order], y[order]
    window = min(max(5, int(len(x_sorted) * window_frac)), len(x_sorted))
    trend = pd.Series(y_sorted).rolling(window, center=True, min_periods=1).median()
    return x_sorted, trend.to_numpy()


def plot_correlation_heatmap(
    univariate_df: pd.DataFrame, output_dir: Path
) -> List[Path]:
    """Feature x target Spearman-rho heatmap (pooled scope)."""
    pooled = univariate_df[univariate_df["scope"] == "pooled"]
    pivot = pooled.pivot(index="feature", columns="target", values="rho")

    written: List[Path] = []
    fig, ax = plt.subplots(
        figsize=(max(6, 1.2 * len(pivot.columns) + 3), max(4, 0.35 * len(pivot.index)))
    )
    sns.heatmap(pivot, cmap="vlag", center=0, annot=True, fmt=".2f", ax=ax)
    ax.set_title("Feature x target Spearman correlation (pooled)")
    png_path = output_dir / "correlation_heatmap.png"
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    written.append(png_path)

    if HAVE_PLOTLY:
        html_fig = px.imshow(
            pivot,
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            labels=dict(color="Spearman rho"),
            title="Feature x target Spearman correlation (pooled)",
        )
        html_path = output_dir / "correlation_heatmap.html"
        html_fig.write_html(str(html_path))
        written.append(html_path)
    else:
        logger.info("plotly not installed; skipping interactive correlation heatmap")

    return written


def plot_top_feature_scatter(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    target_names: Sequence[str],
    univariate_df: pd.DataFrame,
    output_dir: Path,
    top_k: int,
) -> List[Path]:
    """Scatter + rank-smoothed trend for the top-K |rho| features per target."""
    written: List[Path] = []
    feature_names = list(feature_names)
    pooled = univariate_df[univariate_df["scope"] == "pooled"].copy()
    pooled["abs_rho"] = pooled["rho"].abs()

    for j, target in enumerate(target_names):
        top_features = (
            pooled[pooled["target"] == target]
            .sort_values("abs_rho", ascending=False, na_position="last")
            .head(top_k)["feature"]
            .tolist()
        )
        if not top_features:
            continue

        n_cols = len(top_features)
        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4), squeeze=False)
        for i, feature in enumerate(top_features):
            feature_idx = feature_names.index(feature)
            x_vals, y_vals = X[:, feature_idx], y[:, j]
            ax = axes[0][i]
            ax.scatter(x_vals, y_vals, s=8, alpha=0.4)
            x_trend, y_trend = _rolling_trend(x_vals, y_vals)
            ax.plot(x_trend, y_trend, color="firebrick", linewidth=2)
            ax.set_xlabel(feature)
            ax.set_ylabel(target)
        fig.suptitle(f"Top-{len(top_features)} morphology features vs {target}")
        fig.tight_layout()
        png_path = output_dir / f"top_features_{target}.png"
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        written.append(png_path)

        if HAVE_PLOTLY:
            html_fig = make_subplots(rows=1, cols=n_cols, subplot_titles=top_features)
            for i, feature in enumerate(top_features):
                feature_idx = feature_names.index(feature)
                x_vals, y_vals = X[:, feature_idx], y[:, j]
                x_trend, y_trend = _rolling_trend(x_vals, y_vals)
                html_fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        mode="markers",
                        opacity=0.4,
                        showlegend=False,
                    ),
                    row=1,
                    col=i + 1,
                )
                html_fig.add_trace(
                    go.Scatter(
                        x=x_trend,
                        y=y_trend,
                        mode="lines",
                        line=dict(color="firebrick"),
                        showlegend=False,
                    ),
                    row=1,
                    col=i + 1,
                )
                html_fig.update_xaxes(title_text=feature, row=1, col=i + 1)
            html_fig.update_yaxes(title_text=target, row=1, col=1)
            html_fig.update_layout(title=f"Top morphology features vs {target}")
            html_path = output_dir / f"top_features_{target}.html"
            html_fig.write_html(str(html_path))
            written.append(html_path)

    return written


def plot_floor_bar_chart(
    floor_metrics_df: pd.DataFrame,
    noise_ceiling_df: pd.DataFrame,
    output_dir: Path,
) -> List[Path]:
    """Floor R2 per target (linear/nonlinear x with/without suspect features), with the
    noise ceiling drawn as a horizontal reference line."""
    written: List[Path] = []
    targets = floor_metrics_df["target"].unique().tolist()
    fig, axes = plt.subplots(
        1, len(targets), figsize=(4 * len(targets), 4), squeeze=False
    )

    for idx, target in enumerate(targets):
        ax = axes[0][idx]
        subset = floor_metrics_df[floor_metrics_df["target"] == target].copy()
        subset["label"] = subset["model"] + "\n(" + subset["variant"] + ")"
        ax.bar(subset["label"], subset["r2"])
        ax.set_title(target)
        ax.set_ylabel("R2 (grouped CV, out-of-fold)")
        ax.tick_params(axis="x", rotation=45)

        ceiling_row = noise_ceiling_df[noise_ceiling_df["target"] == target]
        if not ceiling_row.empty and np.isfinite(ceiling_row["ceiling"].iloc[0]):
            ax.axhline(
                ceiling_row["ceiling"].iloc[0],
                color="firebrick",
                linestyle="--",
                label="noise ceiling (ICC)",
            )
            ax.legend()

    fig.tight_layout()
    png_path = output_dir / "floor_r2_bar_chart.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    written.append(png_path)

    if HAVE_PLOTLY:
        html_fig = go.Figure()
        for target in targets:
            subset = floor_metrics_df[floor_metrics_df["target"] == target].copy()
            subset["label"] = (
                subset["model"] + " (" + subset["variant"] + ") - " + target
            )
            html_fig.add_trace(go.Bar(x=subset["label"], y=subset["r2"], name=target))
        html_fig.update_layout(
            title="Floor R2 by model/variant/target",
            yaxis_title="R2 (grouped CV, out-of-fold)",
        )
        html_path = output_dir / "floor_r2_bar_chart.html"
        html_fig.write_html(str(html_path))
        written.append(html_path)

    return written


def plot_target_distributions(
    y: np.ndarray, target_names: Sequence[str], output_dir: Path
) -> List[Path]:
    """Target-distribution histograms (dynamic-range check)."""
    written: List[Path] = []
    fig, axes = plt.subplots(
        1, len(target_names), figsize=(4 * len(target_names), 4), squeeze=False
    )
    for idx, target in enumerate(target_names):
        ax = axes[0][idx]
        ax.hist(y[:, idx], bins=30, color="steelblue")
        ax.set_title(target)
        ax.set_xlabel("mCherry intensity")
        ax.set_ylabel("cell count")
    fig.tight_layout()
    png_path = output_dir / "target_distributions.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    written.append(png_path)

    if HAVE_PLOTLY:
        html_fig = make_subplots(
            rows=1, cols=len(target_names), subplot_titles=list(target_names)
        )
        for idx in range(len(target_names)):
            html_fig.add_trace(
                go.Histogram(x=y[:, idx], nbinsx=30, showlegend=False),
                row=1,
                col=idx + 1,
            )
        html_fig.update_layout(title="Target distributions")
        html_path = output_dir / "target_distributions.html"
        html_fig.write_html(str(html_path))
        written.append(html_path)

    return written


def plot_pooled_vs_group_rho(
    univariate_df: pd.DataFrame, output_dir: Path
) -> List[Path]:
    """Pooled-vs-per-group rho comparison (batch-effect check)."""
    pooled = univariate_df[univariate_df["scope"] == "pooled"][
        ["feature", "target", "rho"]
    ].rename(columns={"rho": "pooled_rho"})
    per_group_mean = (
        univariate_df[univariate_df["scope"] == "per_group"]
        .groupby(["feature", "target"])["rho"]
        .mean()
        .reset_index()
        .rename(columns={"rho": "mean_group_rho"})
    )
    merged = pooled.merge(per_group_mean, on=["feature", "target"], how="inner")

    written: List[Path] = []
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(merged["pooled_rho"], merged["mean_group_rho"], s=12, alpha=0.6)
    ax.plot([-1, 1], [-1, 1], color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("pooled Spearman rho")
    ax.set_ylabel("mean per-group Spearman rho")
    ax.set_title("Pooled vs per-group association (batch-effect check)")
    fig.tight_layout()
    png_path = output_dir / "pooled_vs_group_rho.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    written.append(png_path)

    if HAVE_PLOTLY:
        html_fig = px.scatter(
            merged,
            x="pooled_rho",
            y="mean_group_rho",
            hover_data=["feature", "target"],
            title="Pooled vs per-group association (batch-effect check)",
        )
        html_fig.add_shape(
            type="line", x0=-1, y0=-1, x1=1, y1=1, line=dict(dash="dash", color="grey")
        )
        html_path = output_dir / "pooled_vs_group_rho.html"
        html_fig.write_html(str(html_path))
        written.append(html_path)

    return written


def _select_wells(groups: np.ndarray, max_wells: Optional[int]) -> np.ndarray:
    """Unique wells (sorted by name); if ``max_wells`` is set, keep the ``max_wells``
    with the most cells (logging which were dropped)."""
    wells, counts = np.unique(np.asarray(groups), return_counts=True)
    if max_wells is None or len(wells) <= max_wells:
        return np.asarray(wells)

    keep_idx = np.argsort(counts, kind="stable")[::-1][:max_wells]
    kept = set(wells[keep_idx].tolist())
    dropped = [w for w in wells.tolist() if w not in kept]
    logger.info(
        "well-timepoint scatter: keeping %d of %d wells by cell count; dropped %s",
        max_wells,
        len(wells),
        dropped,
    )
    return np.array(sorted(kept))


def _subsample_indices(
    indices: np.ndarray, cap: int, rng: np.random.Generator
) -> np.ndarray:
    """Return ``indices`` unchanged if ``len(indices) <= cap``, else a random
    ``cap``-sized subset (without replacement)."""
    indices = np.asarray(indices)
    if len(indices) <= cap:
        return indices
    return rng.choice(indices, size=cap, replace=False)


def plot_feature_scatter_by_well_timepoint(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    timepoints: np.ndarray,
    feature_names: Sequence[str],
    target_names: Sequence[str],
    univariate_df: pd.DataFrame,
    output_dir: Path,
    top_k: int,
    max_wells: Optional[int] = None,
    max_points_per_well: int = 2000,
    colormap: str = "viridis",
    seed: int = 0,
) -> List[Path]:
    """One figure per (target, top-feature) pair: a grid with one subplot per well,
    points colored by timepoint on a single shared (global) colorbar.

    Makes temporal drift, the ``timepoint=11`` artifact, and well-to-well shape
    differences visible directly on the feature-vs-target relationship. ``groups`` is
    the per-cell well/sample_id label; ``timepoints`` is the per-cell numeric
    timepoint (both aligned with ``X``/``y``). Feature selection reuses
    :func:`~feature_to_mcherry.informativeness.univariate.top_associations`.
    """
    written: List[Path] = []
    feature_names = list(feature_names)
    groups = np.asarray(groups)
    timepoints = np.asarray(timepoints)

    if timepoints.size == 0:
        return written

    # One global vmin/vmax over the whole run so colour is comparable across every
    # subplot in every figure (not per-well, not per-figure).
    global_min = int(np.min(timepoints))
    global_max = int(np.max(timepoints))

    wells = _select_wells(groups, max_wells)
    n_wells = len(wells)
    if n_wells == 0:
        return written

    n_rows = int(math.ceil(math.sqrt(n_wells)))
    n_cols = int(math.ceil(n_wells / n_rows))

    rng = np.random.default_rng(seed)

    for j, target in enumerate(target_names):
        top_features = top_associations(univariate_df, target, top_k, scope="pooled")[
            "feature"
        ].tolist()
        # Be defensive: a stale/external univariate_df could name a feature absent
        # from feature_names; skip those (with a warning) rather than aborting every
        # remaining figure on a feature_names.index ValueError.
        missing = [feature for feature in top_features if feature not in feature_names]
        if missing:
            logger.warning(
                "well-timepoint scatter: %d top feature(s) for target %r not in "
                "feature_names; skipping them: %s",
                len(missing),
                target,
                missing,
            )
        top_features = [feature for feature in top_features if feature in feature_names]

        for feature in top_features:
            feature_idx = feature_names.index(feature)

            # Subsample each renderable well ONCE per (target, feature) so the PNG and
            # the HTML show identical points (a single RNG draw, not one per renderer).
            well_to_idx: Dict[str, np.ndarray] = {}
            for well in wells:
                well_indices = np.where(groups == well)[0]
                if len(well_indices) >= 3:
                    well_to_idx[well] = _subsample_indices(
                        well_indices, max_points_per_well, rng
                    )

            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(4 * n_cols, 3.5 * n_rows),
                squeeze=False,
                constrained_layout=True,
            )
            try:
                for position in range(n_rows * n_cols):
                    row, col = divmod(position, n_cols)
                    ax = axes[row][col]
                    if position >= n_wells:
                        ax.axis("off")
                        continue

                    well = wells[position]
                    ax.set_title(str(well))
                    if well not in well_to_idx:
                        ax.text(
                            0.5,
                            0.5,
                            "insufficient n",
                            ha="center",
                            va="center",
                            transform=ax.transAxes,
                        )
                        ax.set_xticks([])
                        ax.set_yticks([])
                        continue

                    idx = well_to_idx[well]
                    ax.scatter(
                        X[idx, feature_idx],
                        y[idx, j],
                        c=timepoints[idx],
                        cmap=colormap,
                        vmin=global_min,
                        vmax=global_max,
                        s=6,
                        alpha=0.5,
                    )
                    ax.set_xlabel(feature)
                    ax.set_ylabel(target)

                mappable = plt.cm.ScalarMappable(
                    norm=plt.Normalize(vmin=global_min, vmax=global_max), cmap=colormap
                )
                mappable.set_array([])
                fig.colorbar(mappable, ax=axes, label="timepoint", fraction=0.046)
                fig.suptitle(f"{feature} vs {target} by well (colored by timepoint)")

                png_path = output_dir / f"well_timepoint_{target}_{feature}.png"
                fig.savefig(png_path, dpi=150)
            finally:
                plt.close(fig)
            written.append(png_path)

            if HAVE_PLOTLY:
                html_fig = make_subplots(
                    rows=n_rows,
                    cols=n_cols,
                    subplot_titles=[str(w) for w in wells],
                )
                shown_scale = False
                for position, well in enumerate(wells):
                    row, col = divmod(position, n_cols)
                    if well not in well_to_idx:
                        continue
                    idx = well_to_idx[well]
                    html_fig.add_trace(
                        go.Scatter(
                            x=X[idx, feature_idx],
                            y=y[idx, j],
                            mode="markers",
                            marker=dict(
                                color=timepoints[idx],
                                colorscale=colormap,
                                cmin=global_min,
                                cmax=global_max,
                                showscale=not shown_scale,
                                colorbar=dict(title="timepoint"),
                            ),
                            showlegend=False,
                            hovertext=[
                                f"well={well}, timepoint={t}" for t in timepoints[idx]
                            ],
                        ),
                        row=row + 1,
                        col=col + 1,
                    )
                    shown_scale = True
                html_fig.update_layout(
                    title=f"{feature} vs {target} by well (colored by timepoint)"
                )
                html_path = output_dir / f"well_timepoint_{target}_{feature}.html"
                html_fig.write_html(str(html_path))
                written.append(html_path)

    return written


def write_figures(
    output_dir: Path,
    univariate_df: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    target_names: List[str],
    floor_metrics_df: pd.DataFrame,
    noise_ceiling_df: pd.DataFrame,
    top_k: int,
    groups: Optional[np.ndarray] = None,
    timepoints: Optional[np.ndarray] = None,
    well_timepoint_top_k: int = 3,
    well_timepoint_max_wells: Optional[int] = None,
    well_timepoint_max_points_per_well: int = 2000,
    well_timepoint_colormap: str = "viridis",
    well_timepoint_seed: int = 0,
) -> Dict[str, List[Path]]:
    """Write all figures into ``output_dir/figures/`` and return the paths by kind.

    The per-well timepoint scatter is produced only when both ``groups`` (per-cell
    well/sample_id) and ``timepoints`` (per-cell numeric timepoint) are supplied;
    otherwise its key is present but empty (the caller owns the decision to skip,
    e.g. when timepoint coercion fails — see ``informativeness/pipeline.py``).
    """
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not HAVE_PLOTLY:
        logger.info(
            "plotly not installed; writing PNG figures only (no interactive HTML)"
        )

    figures: Dict[str, List[Path]] = {
        "correlation_heatmap": plot_correlation_heatmap(univariate_df, figures_dir),
        "top_feature_scatter": plot_top_feature_scatter(
            X, y, feature_names, target_names, univariate_df, figures_dir, top_k
        ),
        "floor_bar_chart": plot_floor_bar_chart(
            floor_metrics_df, noise_ceiling_df, figures_dir
        ),
        "target_distributions": plot_target_distributions(y, target_names, figures_dir),
        "pooled_vs_group_rho": plot_pooled_vs_group_rho(univariate_df, figures_dir),
    }

    if groups is not None and timepoints is not None:
        figures["well_timepoint_scatter"] = plot_feature_scatter_by_well_timepoint(
            X,
            y,
            np.asarray(groups),
            np.asarray(timepoints),
            feature_names,
            target_names,
            univariate_df,
            figures_dir,
            top_k=well_timepoint_top_k,
            max_wells=well_timepoint_max_wells,
            max_points_per_well=well_timepoint_max_points_per_well,
            colormap=well_timepoint_colormap,
            seed=well_timepoint_seed,
        )
    else:
        figures["well_timepoint_scatter"] = []

    return figures
