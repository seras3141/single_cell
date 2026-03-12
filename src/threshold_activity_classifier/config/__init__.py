"""Configuration management for threshold activity classifier.

This module provides data classes and utilities for managing configuration
settings, with validation, serialization, and defaults management.
"""
from __future__ import annotations

# Import data models
from .models import (
    PreprocessingConfig,
    ThresholdParams,
    ThresholdConfig,
    Metric, Method, NormalizeMode
)

# Import validation utilities
from .validation import (
    get_default_config,
    validate_method_params
)

__all__ = [
    # Types
    'Metric',
    'Method',
    'NormalizeMode',
    # Models
    'PreprocessingConfig',
    'ThresholdParams',
    'ThresholdConfig',
    # Utilities
    'get_default_config',
    'validate_method_params',
]