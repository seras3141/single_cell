"""Image preprocessing functionality."""
from __future__ import annotations

import numpy as np
from skimage import filters, morphology
from scipy import ndimage as ndi

from threshold_activity_classifier.config import PreprocessingConfig


class ImagePreprocessor:
    """Handles image preprocessing operations."""
    
    def __init__(self, config: PreprocessingConfig):
        """Initialize preprocessor with configuration.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply preprocessing to an image.
        
        Args:
            image: Input image array
            
        Returns:
            Preprocessed image
        """
        return self.preprocess(image)
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Apply preprocessing steps to image.
        
        Args:
            image: Input image array
            
        Returns:
            Preprocessed image array
        """
        result = image.astype(np.float64)
        
        # Normalization
        if self.config.normalize:
            result = self._normalize(result)
        
        # Gaussian blur
        if self.config.gaussian_sigma > 0:
            result = self._gaussian_blur(result)
        
        # Median filtering
        if self.config.median_footprint > 0:
            result = self._median_filter(result)
        
        # Background subtraction
        if self.config.background_subtract_radius > 0:
            result = self._subtract_background(result)
        
        return result
    
    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize image intensities."""
        if self.config.normalize_mode == 'minmax':
            min_val, max_val = np.nanmin(image), np.nanmax(image)
            if max_val > min_val:
                return (image - min_val) / (max_val - min_val)
        else:  # percentile
            p1, p99 = np.nanpercentile(image, (1, 99))
            if p99 > p1:
                return (image - p1) / (p99 - p1)
        
        return image
    
    def _gaussian_blur(self, image: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur."""
        return ndi.gaussian_filter(image, sigma=self.config.gaussian_sigma)
    
    def _median_filter(self, image: np.ndarray) -> np.ndarray:
        """Apply median filtering."""
        footprint = morphology.disk(self.config.median_footprint)
        return filters.median(image, footprint=footprint)
    
    def _subtract_background(self, image: np.ndarray) -> np.ndarray:
        """Subtract background using morphological opening."""
        structuring_element = morphology.disk(self.config.background_subtract_radius)
        background = morphology.opening(image, structuring_element)
        return image - background


__all__ = ['ImagePreprocessor']
