"""mCherry activity metrics extraction.

This package provides a standalone extraction path for per-instance mCherry
intensity measurements without importing the activity labeling stack.
"""

from .config import ExtractionConfig
from .core.batch import MetricsExtractor, run_extraction

__all__ = ["ExtractionConfig", "MetricsExtractor", "run_extraction"]