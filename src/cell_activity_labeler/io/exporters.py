"""CSV export helpers for labeled activity results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


LABELED_INSTANCE_COLUMNS = [
    "image_path",
    "label_path",
    "cell_id",
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
    "metric_value",
    "threshold",
    "threshold_value",
    "metric_used",
    "is_active",
    "cell_status",
    "labeling_method",
    "labeling_scope",
]


def finalize_labeled_dataframe(labeled_df: pd.DataFrame) -> pd.DataFrame:
    """Return a labeled dataframe with stable output column ordering."""
    ordered = [column for column in LABELED_INSTANCE_COLUMNS if column in labeled_df.columns]
    extra_columns = [column for column in labeled_df.columns if column not in ordered]
    return labeled_df.loc[:, ordered + sorted(extra_columns)]


def write_labeled_instances(labeled_df: pd.DataFrame, output_csv: Path) -> Path:
    """Write the labeled instance CSV using milestone-2 column ordering."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    finalize_labeled_dataframe(labeled_df).to_csv(output_csv, index=False)
    return output_csv


def write_label_summary(summary_df: pd.DataFrame, output_csv: Path) -> Path:
    """Write the per-image label summary CSV."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_csv, index=False)
    return output_csv


__all__ = [
    "LABELED_INSTANCE_COLUMNS",
    "finalize_labeled_dataframe",
    "write_labeled_instances",
    "write_label_summary",
]