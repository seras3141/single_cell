"""Validation and utility functions for configuration."""
from __future__ import annotations

from .models import Method
from .models import ThresholdConfig, ThresholdParams


def get_default_config() -> ThresholdConfig:
    """Get the default threshold configuration."""
    return ThresholdConfig()


def validate_method_params(method: Method, params: ThresholdParams) -> None:
    """Validate that parameters are appropriate for the chosen method.
    
    Args:
        method: Thresholding method
        params: Parameters to validate
        
    Raises:
        ValueError: If parameters are invalid for the method
    """
    if method == 'percentile' and params.percentile is None:
        raise ValueError("percentile method requires percentile parameter")
    
    if method == 'manual' and params.manual_value is None:
        raise ValueError("manual method requires manual_value parameter")
    
    if method in ('local', 'adaptive') and params.block_size <= 0:
        raise ValueError("local/adaptive methods require positive block_size")


__all__ = ['get_default_config', 'validate_method_params']
