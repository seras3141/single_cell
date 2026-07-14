"""Frontend contract for milestone-2 activity labeling."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from ..config import LabelingConfig


class AbstractFrontend(ABC):
    """Plugin contract for activity-labeling frontends."""

    @abstractmethod
    def load_metrics(self, source: Path) -> pd.DataFrame:
        """Load instance_metrics.csv and return a validated dataframe."""

    @abstractmethod
    def configure_labeling(self) -> LabelingConfig:
        """Obtain the labeling configuration."""

    @abstractmethod
    def run_labeling(
        self, metrics_df: pd.DataFrame, config: LabelingConfig
    ) -> pd.DataFrame:
        """Apply labeling and return the labeled dataframe."""

    @abstractmethod
    def show_analytics(self, labeled_df: pd.DataFrame) -> None:
        """Generate or display analytics for the labeled results."""

    @abstractmethod
    def export_results(self, labeled_df: pd.DataFrame, output_dir: Path) -> None:
        """Write labeled CSV outputs and activity masks."""


__all__ = ["AbstractFrontend"]