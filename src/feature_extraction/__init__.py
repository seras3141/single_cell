"""
Feature extraction module for single-cell analysis.

This module provides tools for extracting morphological, intensity, spatial,
and texture features from 2D and 3D cell instance segmentations.
"""

from .feature_extractor_incarta import (
    compute_morphology_features,
    compute_intensity_features,
    compute_spatial_features,
    compute_texture_features,
    extract_instance_features,
    extract_all_instance_features
)

# scPortrait is an optional dependency; guard the import so the package still
# imports cleanly when scportrait is not installed.
try:
    from .feature_extractor_scportrait import get_scportrait_features
except ImportError:
    get_scportrait_features = None


__all__ = [
    'compute_morphology_features',
    'compute_intensity_features',
    'compute_spatial_features',
    'compute_texture_features',
    'extract_instance_features',
    'extract_all_instance_features',
    'get_scportrait_features'
]
