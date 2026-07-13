"""Morphology-informativeness feasibility gate for feature_to_mcherry.

Quantifies how much interpretable brightfield morphology (size/shape + texture)
predicts per-cell mCherry percentiles, bounding what any downstream deep-feature
model can achieve. This is an analysis, not a predictive product.
"""

from .config import InformativenessConfig, load_config
from .pipeline import ResultsBundle, run

__all__ = [
    "InformativenessConfig",
    "load_config",
    "ResultsBundle",
    "run",
]
