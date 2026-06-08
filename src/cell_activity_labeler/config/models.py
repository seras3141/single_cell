"""Data classes for configuration management."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Any, Dict, Union

from typing import Literal


# Type aliases for better code readability
Metric = Literal['max_intensity', 'mean_intensity', 'sum_intensity', 'percentile_95', 'percentile_90', 'percentile_75']
Method = Literal['otsu', 'yen', 'li', 'triangle', 'percentile', 'manual', 'local', 'adaptive']
NormalizeMode = Literal['minmax', 'percentile']

@dataclass(frozen=True)
class PreprocessingConfig:
    """Configuration for image preprocessing steps.
    
    Attributes:
        normalize: Whether to normalize image intensities
        normalize_mode: Method for normalization ('minmax' or 'percentile')
        gaussian_sigma: Standard deviation for Gaussian blur (0 = no blur)
        median_footprint: Radius for median filter (0 = no median filtering)
        background_subtract_radius: Radius for morphological opening background subtraction (0 = none)
    """
    normalize: bool = True
    normalize_mode: NormalizeMode = 'minmax'
    gaussian_sigma: float = 0.0
    median_footprint: int = 0
    background_subtract_radius: int = 0

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.gaussian_sigma < 0:
            raise ValueError("gaussian_sigma must be non-negative")
        if self.median_footprint < 0:
            raise ValueError("median_footprint must be non-negative")
        if self.background_subtract_radius < 0:
            raise ValueError("background_subtract_radius must be non-negative")


@dataclass(frozen=True)
class ThresholdParams:
    """Parameters for specific thresholding methods.
    
    Attributes:
        percentile: Percentile value for percentile-based thresholding (0-100)
        manual_value: Fixed threshold value for manual thresholding
        block_size: Block size for local/adaptive thresholding (must be odd)
        offset: Offset value for local thresholding
    """
    percentile: Optional[float] = None
    manual_value: Optional[float] = None
    block_size: int = 51
    offset: float = 0.0

    def __post_init__(self):
        """Validate threshold parameters."""
        if self.percentile is not None:
            if not (0 <= self.percentile <= 100):
                raise ValueError("percentile must be between 0 and 100")
        
        if self.block_size <= 0 or self.block_size % 2 == 0:
            raise ValueError("block_size must be a positive odd integer")


@dataclass(frozen=True)
class ThresholdConfig:
    """Complete configuration for threshold-based image analysis.
    
    Attributes:
        metric: Image metric to compute for analysis
        method: Thresholding method to use
        per_image: Whether to compute thresholds per image or globally
        params: Parameters specific to the thresholding method
        preprocessing: Preprocessing configuration
    """
    metric: Metric = 'mean_intensity'
    method: Method = 'otsu'
    per_image: bool = False
    params: ThresholdParams = field(default_factory=ThresholdParams)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ThresholdConfig':
        """Create configuration from dictionary."""
        # Handle nested dataclasses
        if 'params' in data and isinstance(data['params'], dict):
            data['params'] = ThresholdParams(**data['params'])
        if 'preprocessing' in data and isinstance(data['preprocessing'], dict):
            data['preprocessing'] = PreprocessingConfig(**data['preprocessing'])
        
        return cls(**data)

    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access: config['metric']"""
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        """Get attribute by name with optional default, like dict.get()"""
        try:
            return getattr(self, key)
        except AttributeError:
            return default

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'ThresholdConfig':
        """Load configuration from JSON file."""
        path = Path(path)
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


class LabelingConfig(ThresholdConfig):
    """Configuration alias for instance-level activity labeling."""


__all__ = [
    'PreprocessingConfig',
    'ThresholdParams',
    'ThresholdConfig',
    'LabelingConfig',
    'Metric',
    'Method',
    'NormalizeMode',
]
