"""Data-quality diagnostics for feature_to_mcherry feature/target tables.

Distributional QC on already-extracted features + targets (extreme-value
clustering by well/z-slice/timepoint, per-well/timepoint distribution reports) —
distinct from ``src.dataset_analysis``, which covers raw/processed-file inventory
and completeness. Standalone config, mirroring ``feature_to_mcherry.config`` and
``feature_to_mcherry.informativeness.config``.
"""

from .config import DataQualityConfig, SourceConfig, load_config
from .extremes import compute_extreme_value_report
from .loading import load_source
from .plots import build_interactive_report_html

__all__ = [
    "DataQualityConfig",
    "SourceConfig",
    "load_config",
    "compute_extreme_value_report",
    "load_source",
    "build_interactive_report_html",
]
