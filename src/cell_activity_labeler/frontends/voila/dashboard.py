"""Voila-oriented frontend wrapper for milestone-2 labeling."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ...analytics import run_labeling_analytics
from ...config import LabelingConfig
from ...core import ThresholdInstanceLabeler, save_activity_images
from ...io import load_metrics_csv, write_labeled_instances, write_label_summary
from ...analytics import build_label_summary
from ..base import AbstractFrontend


class VoilaFrontend(AbstractFrontend):
    """Notebook-friendly orchestration wrapper for the core labeler."""

    def __init__(
        self,
        config: LabelingConfig | None = None,
        analytics_dir: Path | None = None,
    ) -> None:
        self._config = config
        self.analytics_dir = analytics_dir

    def load_metrics(self, source: Path) -> pd.DataFrame:
        return load_metrics_csv(source)

    def configure_labeling(self) -> LabelingConfig:
        if self._config is None:
            raise ValueError("VoilaFrontend requires a LabelingConfig before labeling")
        return self._config

    def run_labeling(
        self, metrics_df: pd.DataFrame, config: LabelingConfig
    ) -> pd.DataFrame:
        return ThresholdInstanceLabeler(config).run(metrics_df)

    def show_analytics(self, labeled_df: pd.DataFrame) -> None:
        if self.analytics_dir is None:
            return
        run_labeling_analytics(labeled_df, self.analytics_dir)

    def export_results(self, labeled_df: pd.DataFrame, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_labeled_instances(labeled_df, output_dir / "labeled_instances.csv")
        write_label_summary(
            build_label_summary(labeled_df), output_dir / "label_summary.csv"
        )
        save_activity_images(labeled_df, output_dir / "activity_images")


__all__ = ["VoilaFrontend"]