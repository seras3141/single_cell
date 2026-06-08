"""Summary-table analytics for extracted mCherry metrics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _metric_columns(metrics_df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in metrics_df.columns
        if column in {"area", "mean_intensity", "max_intensity", "min_intensity", "sum_intensity"}
        or column.startswith("percentile_")
    ]


def build_metrics_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Build a per-image summary table.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Instance-level metrics table.

    Returns
    -------
    pd.DataFrame
        One row per image with summary statistics for each metric.
    """
    if metrics_df.empty:
        return pd.DataFrame(columns=["image_path", "sample_id", "n_instances"])

    metric_columns = _metric_columns(metrics_df)
    grouped = metrics_df.groupby(["image_path", "sample_id"], dropna=False)
    summary = grouped.size().rename("n_instances").reset_index()

    for metric in metric_columns:
        stats = grouped[metric].agg(["mean", "median", "std", "min", "max"])
        stats = stats.rename(
            columns={
                "mean": f"{metric}_mean",
                "median": f"{metric}_median",
                "std": f"{metric}_std",
                "min": f"{metric}_min",
                "max": f"{metric}_max",
            }
        ).reset_index()
        summary = summary.merge(stats, on=["image_path", "sample_id"], how="left")

    return summary.sort_values(["sample_id", "image_path"]).reset_index(drop=True)


def write_metrics_summary(metrics_df: pd.DataFrame, output_csv: Path) -> Path:
    """Write the per-image metrics summary CSV."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    build_metrics_summary(metrics_df).to_csv(output_csv, index=False)
    return output_csv