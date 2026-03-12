"""Core thresholding functionality.

This module provides the main ThresholdClassifier class and supporting functions
for image preprocessing, threshold computation, and mask generation.
"""
from __future__ import annotations

# Import preprocessing
from .preprocessing import ImagePreprocessor

# Import threshold computation
from .thresholding import ThresholdComputer

# Import main classifier
from .classifier import ThresholdClassifier, create_classifier

# Import metrics extraction
from .metrics import InstanceMetricsExtractor

__all__ = [
    'ImagePreprocessor',
    'ThresholdComputer',
    'ThresholdClassifier',
    'create_classifier',
    'InstanceMetricsExtractor',
]