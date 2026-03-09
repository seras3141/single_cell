"""
Evaluation module for single-cell instance segmentation.

This module provides comprehensive evaluation metrics for instance segmentation tasks,
including both instance-level and pixel-level metrics, as well as visualization tools.
"""

from .instance_metrics import InstanceSegmentationMetrics
from .pixel_metrics import PixelWiseMetrics
from .evaluation_pipeline import EvaluationPipeline
from .plotting import EvaluationPlotter, plot_evaluation_results

__all__ = [
    'InstanceSegmentationMetrics',
    'PixelWiseMetrics', 
    'EvaluationPipeline',
    'EvaluationPlotter',
    'plot_evaluation_results'
]
