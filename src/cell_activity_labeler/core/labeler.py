"""Instance-level labeling flow for precomputed metrics tables."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..config import LabelingConfig, ThresholdConfig, ThresholdParams
from ..io.loaders import normalize_metrics_dataframe
from .strategies import get_labeling_strategy


class ThresholdInstanceLabeler:
    """Apply threshold-based activity labels to an instance-metrics table."""

    def __init__(self, config: Optional[LabelingConfig] = None) -> None:
        self.config = config or LabelingConfig()

    def label_instances(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """Return a labeled copy of the metrics table."""
        labeled_df = normalize_metrics_dataframe(metrics_df)

        if self.config.metric not in labeled_df.columns:
            raise ValueError(
                f"Requested metric '{self.config.metric}' not found in metrics DataFrame. "
                f"Available columns: {labeled_df.columns.tolist()}"
            )

        labeled_df["metric_value"] = labeled_df[self.config.metric].astype(float)
        strategy = get_labeling_strategy(self.config.method, self.config.params)

        if self.config.per_image:
            thresholds: dict[object, float] = {}
            for image_name, group in labeled_df.groupby("image", dropna=False):
                thresholds[image_name] = strategy.compute_threshold(
                    group["metric_value"].to_numpy()
                )
            labeled_df["threshold_value"] = labeled_df["image"].map(thresholds)
            labeling_scope = "per_image"
        else:
            threshold_value = strategy.compute_threshold(
                labeled_df["metric_value"].to_numpy()
            )
            labeled_df["threshold_value"] = float(threshold_value)
            labeling_scope = "global"

        labeled_df["threshold"] = labeled_df["threshold_value"].astype(float)
        labeled_df["is_active"] = (
            labeled_df["metric_value"] > labeled_df["threshold_value"]
        ).astype(bool)
        labeled_df["cell_status"] = np.where(
            labeled_df["is_active"], "active", "dead"
        )
        labeled_df["metric_used"] = self.config.metric
        labeled_df["labeling_method"] = self.config.method
        labeled_df["labeling_scope"] = labeling_scope

        return labeled_df

    def run(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """Alias for label_instances to support frontend orchestration."""
        return self.label_instances(metrics_df)


def apply_threshold_classification(
    metrics_df: pd.DataFrame,
    threshold_metric: str,
    threshold_method: str,
    manual_value: float,
    per_image_threshold: bool,
) -> pd.DataFrame:
    """Legacy wrapper preserved for notebook compatibility."""
    config = ThresholdConfig.from_dict(
        {
            "metric": threshold_metric,
            "method": threshold_method,
            "per_image": per_image_threshold,
            "params": {
                "manual_value": manual_value,
                "percentile": manual_value,
            },
        }
    )
    labeler = ThresholdInstanceLabeler(LabelingConfig.from_dict(config.to_dict()))
    return labeler.label_instances(metrics_df)


__all__ = ["ThresholdInstanceLabeler", "apply_threshold_classification"]