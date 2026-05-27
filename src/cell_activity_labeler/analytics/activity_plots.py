"""Activity-ratio plots for labeled results."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_activity_ratio_bar(summary_df: pd.DataFrame, output_path: Path) -> Path:
    """Save a bar chart of activity ratio per image."""
    ordered = summary_df.sort_values("activity_ratio", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(ordered["image"], ordered["activity_ratio"], color="#35618f")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Activity ratio")
    ax.set_title("Activity ratio per image")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def save_threshold_sensitivity(
    labeled_df: pd.DataFrame, metric: str, output_path: Path
) -> Path:
    """Save activity ratio as a function of threshold over metric quantiles."""
    values = labeled_df[metric].dropna().astype(float)
    if values.empty:
        raise ValueError(f"Metric '{metric}' has no non-null values for sensitivity plot")

    quantiles = np.linspace(0.05, 0.95, 19)
    thresholds = np.quantile(values, quantiles)
    activity_ratios = [(values > threshold).mean() for threshold in thresholds]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, activity_ratios, marker="o", color="#a0462b")
    ax.set_xlabel(metric)
    ax.set_ylabel("Activity ratio")
    ax.set_title("Threshold sensitivity")
    ax.grid(alpha=0.2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


__all__ = ["save_activity_ratio_bar", "save_threshold_sensitivity"]