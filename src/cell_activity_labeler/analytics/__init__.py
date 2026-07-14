"""Analytics helpers for milestone-2 activity labeling."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .activity_plots import save_activity_ratio_bar, save_threshold_sensitivity
from .comparison import save_activity_ratio_by_sample, save_plate_activity_heatmap
from .distributions import save_intensity_overlay
from .label_summary import build_label_summary, write_label_summary_csv


def run_labeling_analytics(
    labeled_df: pd.DataFrame,
    output_dir: Path,
    metric: str | None = None,
) -> dict[str, Path]:
    """Generate the standard analytics bundle for labeled results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_metric = metric or str(labeled_df["metric_used"].iloc[0])
    summary_df = build_label_summary(labeled_df)

    outputs: dict[str, Path] = {
        "label_summary_csv": write_label_summary_csv(
            summary_df, output_dir / "label_summary.csv"
        ),
        "activity_ratio_bar": save_activity_ratio_bar(
            summary_df, output_dir / "activity_ratio_bar.png"
        ),
        "intensity_overlay": save_intensity_overlay(
            labeled_df,
            selected_metric,
            output_dir / f"intensity_overlay_{selected_metric}.png",
        ),
        "threshold_sensitivity": save_threshold_sensitivity(
            labeled_df,
            selected_metric,
            output_dir / "threshold_sensitivity.png",
        ),
    }

    activity_ratio_by_sample = save_activity_ratio_by_sample(
        labeled_df, output_dir / "activity_ratio_by_sample.png"
    )
    if activity_ratio_by_sample is not None:
        outputs["activity_ratio_by_sample"] = activity_ratio_by_sample

    plate_heatmap = save_plate_activity_heatmap(
        labeled_df, output_dir / "plate_activity_heatmap.png"
    )
    if plate_heatmap is not None:
        outputs["plate_activity_heatmap"] = plate_heatmap

    return outputs


__all__ = [
    "build_label_summary",
    "write_label_summary_csv",
    "save_intensity_overlay",
    "save_activity_ratio_bar",
    "save_threshold_sensitivity",
    "save_activity_ratio_by_sample",
    "save_plate_activity_heatmap",
    "run_labeling_analytics",
]