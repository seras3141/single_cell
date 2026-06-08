"""Distribution plots for extracted mCherry metrics."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _intensity_metric_columns(metrics_df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in metrics_df.columns
        if column in {"mean_intensity", "max_intensity", "min_intensity", "sum_intensity"}
        or column.startswith("percentile_")
    ]


def plot_metric_distributions(metrics_df: pd.DataFrame, save_dir: Path) -> list[Path]:
    """Write histogram plots for each intensity metric."""
    save_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    for metric in _intensity_metric_columns(metrics_df):
        fig, ax = plt.subplots(figsize=(8, 5))
        hue = None
        if "sample_id" in metrics_df.columns and metrics_df["sample_id"].nunique() > 1:
            hue = "sample_id"

        sns.histplot(
            data=metrics_df,
            x=metric,
            hue=hue,
            kde=metrics_df[metric].nunique() > 1,
            ax=ax,
            stat="count",
            common_norm=False,
        )
        ax.set_title(f"Distribution of {metric}")
        ax.set_xlabel(metric)
        ax.set_ylabel("Count")

        output_path = save_dir / f"distribution_{metric}.png"
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        outputs.append(output_path)

    return outputs


def plot_metric_violins(metrics_df: pd.DataFrame, save_dir: Path) -> list[Path]:
    """Write violin plots comparing metrics across samples."""
    if "sample_id" not in metrics_df.columns or metrics_df["sample_id"].nunique() < 2:
        return []

    outputs: list[Path] = []
    for metric in _intensity_metric_columns(metrics_df):
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.violinplot(data=metrics_df, x="sample_id", y=metric, ax=ax, cut=0)
        ax.set_title(f"{metric} by sample")
        ax.set_xlabel("Sample")
        ax.set_ylabel(metric)

        output_path = save_dir / f"violin_{metric}_by_sample.png"
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        outputs.append(output_path)

    return outputs


def plot_area_vs_intensity(metrics_df: pd.DataFrame, save_dir: Path) -> Path:
    """Write a scatter plot of instance area versus mean intensity."""
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=metrics_df, x="area", y="mean_intensity", ax=ax)
    ax.set_title("Area vs mean intensity")
    ax.set_xlabel("Area (px)")
    ax.set_ylabel("Mean intensity")

    output_path = save_dir / "area_vs_intensity.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path