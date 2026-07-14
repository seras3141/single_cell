"""Threshold computation methods."""
from __future__ import annotations

from typing import Union

import numpy as np
from skimage import filters

from cell_activity_labeler.config import ThresholdParams, Method, validate_method_params


class ThresholdComputer:
    """Computes threshold values using various methods."""
    
    AVAILABLE_METHODS = ['otsu', 'yen', 'li', 'triangle', 'percentile', 'manual', 'local', 'adaptive']
    
    def __init__(self, method: Method, params: ThresholdParams):
        """Initialize threshold computer.
        
        Args:
            method: Thresholding method to use
            params: Parameters for the method
        """
        self.method = method
        self.params = params
        validate_method_params(method, params)
    
    def compute(self, image: np.ndarray) -> Union[float, np.ndarray]:
        """Compute threshold value or threshold map.
        
        Args:
            image: Preprocessed image array
            
        Returns:
            Scalar threshold value or per-pixel threshold array
            
        Raises:
            ValueError: If method is not supported
        """
        if self.method == 'otsu':
            return self._threshold_otsu(image)
        elif self.method == 'yen':
            return self._threshold_yen(image)
        elif self.method == 'li':
            return self._threshold_li(image)
        elif self.method == 'triangle':
            return self._threshold_triangle(image)
        elif self.method == 'percentile':
            return self._threshold_percentile(image)
        elif self.method == 'manual':
            return self._threshold_manual()
        elif self.method in ('local', 'adaptive'):
            return self._threshold_local(image)
        else:
            raise ValueError(f"Unsupported thresholding method: {self.method}")
    
    def _threshold_otsu(self, image: np.ndarray) -> float:
        """Compute Otsu threshold."""
        return float(filters.threshold_otsu(image))
    
    def _threshold_yen(self, image: np.ndarray) -> float:
        """Compute Yen threshold."""
        return float(filters.threshold_yen(image))
    
    def _threshold_li(self, image: np.ndarray) -> float:
        """Compute Li threshold."""
        return float(filters.threshold_li(image))
    
    def _threshold_triangle(self, image: np.ndarray) -> float:
        """Compute Triangle threshold."""
        return float(filters.threshold_triangle(image))
    
    def _threshold_percentile(self, image: np.ndarray) -> float:
        """Compute percentile-based threshold."""
        valid_pixels = image[~np.isnan(image)].ravel()
        percentile = self.params.percentile or 90.0
        return float(np.percentile(valid_pixels, percentile))
    
    def _threshold_manual(self) -> float:
        """Return manual threshold value."""
        manual_val = self.params.manual_value or 0.5
        return float(manual_val)
    
    def _threshold_local(self, image: np.ndarray) -> np.ndarray:
        """Compute local adaptive threshold."""
        return filters.threshold_local(
            image,
            block_size=self.params.block_size,
            offset=int(self.params.offset)
        )


__all__ = ['ThresholdComputer']
