"""Summary helpers for threshold-based activity classification."""
from __future__ import annotations

import pandas as pd


def generate_activity_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-instance classification results into per-image summaries."""
    summary = metrics_df.groupby("image").agg(
        n_instances=("label", "count"),
        n_active=("is_active", "sum"),
        n_dead=("is_active", lambda values: (~values).sum()),
        activity_ratio=("is_active", "mean"),
        metric_median=("metric_value", "median"),
        metric_mean=("metric_value", "mean"),
        metric_std=("metric_value", "std"),
        threshold_used=("threshold", "first"),
    ).reset_index()
    summary["percent_active"] = (summary["n_active"] / summary["n_instances"]) * 100
    return summary


def print_activity_summary(summary: pd.DataFrame, out_widget=None) -> None:
    """Print a compact text summary, optionally routing the table to a widget."""
    print(f"Total images processed: {len(summary)}")
    print(
        "Average active cell percentage: "
        f"{summary['activity_ratio'].mean() * 100:.1f}% "
        f"+/- {summary['activity_ratio'].std() * 100:.1f}%"
    )

    target = out_widget
    if target is None:
        print("\nPer-image summary:")
        print(summary.round(2))
    else:
        with target:
            print("\nPer-image summary:")
            print(summary.round(2))
