"""Static (PNG) and interactive (HTML) figures for the informativeness report.

Standalone, modern matplotlib/seaborn + plotly plotting — not coupled to the legacy
``feature_visualization`` API/data-flow contract. The matplotlib ``Agg`` backend is
forced before importing pyplot so figure generation works headless (no display, no
``plt.show()``). The plotly import is guarded: PNGs are always produced; HTML is
skipped with a logged note if plotly is unavailable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

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
) -> Dict[str, List[Path]]:
    """Write all figures into ``output_dir/figures/`` and return the paths by kind."""
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not HAVE_PLOTLY:
        logger.info(
            "plotly not installed; writing PNG figures only (no interactive HTML)"
        )

    return {
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
