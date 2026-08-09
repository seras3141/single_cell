"""Generate the figures specified in ``docs_local/figures/report_figures_brief.md``
from already-computed ``feature_to_mcherry``/``informativeness``/``data_quality``
results. Reads results only -- does not fit or refit models (except the two small,
additive out-of-fold/feature-importance persistence changes in the source pipelines
themselves, tracked separately)."""

from .config import ReportFiguresConfig, load_config
from .manifest import FigureStatus, ManifestWriter
from .pipeline import FIGURE_FUNCTIONS, run

__all__ = [
    "ReportFiguresConfig",
    "load_config",
    "FigureStatus",
    "ManifestWriter",
    "FIGURE_FUNCTIONS",
    "run",
]
