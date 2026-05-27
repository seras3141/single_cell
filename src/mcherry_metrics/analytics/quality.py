"""Quality-control analytics for extracted mCherry metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def build_quality_report(
    metrics_df: pd.DataFrame,
    processed_image_paths: list[Path] | None = None,
) -> str:
    """Build a text QC report for a completed extraction run."""
    processed_paths = {
        str(path.resolve()) for path in processed_image_paths or []
    }
    observed_paths = set(metrics_df.get("image_path", pd.Series(dtype=str)).dropna())
    zero_instance_images = sorted(processed_paths - observed_paths)

    metric_columns = [
        column
        for column in metrics_df.columns
        if column in {"mean_intensity", "max_intensity", "min_intensity", "sum_intensity"}
        or column.startswith("percentile_")
    ]

    outlier_counts: list[str] = []
    for column in metric_columns:
        values = metrics_df[column].astype(float)
        std = float(values.std(ddof=0)) if not values.empty else 0.0
        if std == 0.0:
            count = 0
        else:
            z_scores = np.abs((values - float(values.mean())) / std)
            count = int((z_scores > 3.0).sum())
        outlier_counts.append(f"- {column}: {count}")

    return "\n".join(
        [
            "mCherry metrics QC report",
            f"total_instances: {len(metrics_df)}",
            f"unique_images: {metrics_df.get('image_path', pd.Series(dtype=str)).nunique()}",
            f"images_with_zero_instances: {len(zero_instance_images)}",
            f"missing_sample_id_rows: {int((metrics_df.get('sample_id', pd.Series(dtype=str)) == '').sum())}",
            "outlier_counts:",
            *outlier_counts,
        ]
    )


def write_quality_report(
    metrics_df: pd.DataFrame,
    output_path: Path,
    processed_image_paths: list[Path] | None = None,
) -> Path:
    """Write the quality-control report to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_quality_report(metrics_df, processed_image_paths=processed_image_paths)
        + "\n",
        encoding="utf-8",
    )
    return output_path