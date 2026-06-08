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

# Import instance labeling
from .labeler import ThresholdInstanceLabeler, apply_threshold_classification

# Import activity image writer
from .image_writer import create_activity_labeled_image, save_activity_images

# Import strategies
from .strategies import LabelingStrategy, get_labeling_strategy

__all__ = [
    'ImagePreprocessor',
    'ThresholdComputer',
    'ThresholdClassifier',
    'ThresholdInstanceLabeler',
    'create_classifier',
    'apply_threshold_classification',
    'create_activity_labeled_image',
    'save_activity_images',
    'LabelingStrategy',
    'get_labeling_strategy',
]