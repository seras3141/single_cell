"""Correlation heatmap analytics for extracted mCherry metrics."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_metric_correlation_heatmap(metrics_df: pd.DataFrame, save_dir: Path) -> Path:
    """Write a Pearson correlation heatmap for numeric metric columns."""
    metric_columns = [
        column
        for column in metrics_df.columns
        if column in {"area", "mean_intensity", "max_intensity", "min_intensity", "sum_intensity"}
        or column.startswith("percentile_")
    ]

    correlation = metrics_df.loc[:, metric_columns].corr(method="pearson")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation, cmap="viridis", annot=False, ax=ax)
    ax.set_title("Metric correlation heatmap")

    output_path = save_dir / "metric_correlation_heatmap.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path