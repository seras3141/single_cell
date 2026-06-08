"""Export helpers for mCherry metrics tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


INSTANCE_METRICS_COLUMNS = [
    "image_path",
    "label_path",
    "label_id",
    "area",
    "mean_intensity",
    "max_intensity",
    "min_intensity",
    "sum_intensity",
    "percentile_75",
    "percentile_90",
    "percentile_95",
    "sample_id",
    "z_index",
    "timepoint",
    "image",
    "sample",
    "time",
    "ID",
]


def finalize_metrics_dataframe(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Return a CSV-ready metrics table with stable column ordering."""
    ordered_columns = list(INSTANCE_METRICS_COLUMNS)
    extra_columns = [
        column for column in metrics_df.columns if column not in ordered_columns
    ]
    return metrics_df.loc[:, ordered_columns + sorted(extra_columns)]


def validate_metrics_dataframe(metrics_df: pd.DataFrame) -> None:
    """Validate that the table satisfies the milestone-1 CSV contract."""
    missing_columns = [
        column for column in INSTANCE_METRICS_COLUMNS[:14] if column not in metrics_df.columns
    ]
    if missing_columns:
        raise ValueError(
            "metrics dataframe is missing required columns: "
            + ", ".join(missing_columns)
        )


def write_instance_metrics(metrics_df: pd.DataFrame, output_csv: Path) -> Path:
    """Write the instance-level metrics CSV."""
    validate_metrics_dataframe(metrics_df)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    finalized = finalize_metrics_dataframe(metrics_df)
    finalized.to_csv(output_csv, index=False)
    return output_csv