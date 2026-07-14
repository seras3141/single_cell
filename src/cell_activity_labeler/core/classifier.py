"""Main threshold classifier class."""
from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from cell_activity_labeler.config import ThresholdConfig, Method
from .image_writer import create_activity_labeled_image as _create_activity_labeled_image
from .labeler import apply_threshold_classification as _apply_threshold_classification
from .preprocessing import ImagePreprocessor
from .thresholding import ThresholdComputer


def apply_threshold_classification(
    metrics_df: pd.DataFrame,
    threshold_metric: str,
    threshold_method: str,
    manual_value: float,
    per_image_threshold: bool,
):
    return _apply_threshold_classification(
        metrics_df=metrics_df,
        threshold_metric=threshold_metric,
        threshold_method=threshold_method,
        manual_value=manual_value,
        per_image_threshold=per_image_threshold,
    )

class ThresholdClassifier:
    """Main class for threshold-based image classification.
    
    This class combines preprocessing, threshold computation, and mask generation
    into a single, easy-to-use interface.
    """
    
    def __init__(self, config: Optional[ThresholdConfig] = None):
        """Initialize threshold classifier.
        
        Args:
            config: Configuration object. If None, uses default configuration.
        """
        self.config = config or ThresholdConfig()
        self.preprocessor = ImagePreprocessor(self.config.preprocessing)
        self.threshold_computer = ThresholdComputer(self.config.method, self.config.params)
    
    def process(self, image: np.ndarray) -> Tuple[np.ndarray, Union[float, np.ndarray]]:
        """Process image and return binary mask and threshold value.
        
        Args:
            image: Input image array
            
        Returns:
            Tuple of (binary_mask, threshold_value)
        """
        # Preprocess image
        preprocessed = self.preprocessor.preprocess(image)
        
        # Compute threshold
        threshold = self.threshold_computer.compute(preprocessed)
        
        # Generate binary mask
        mask = self.apply_threshold(preprocessed, threshold)
        
        return mask, threshold
    
    def preprocess_only(self, image: np.ndarray) -> np.ndarray:
        """Apply only preprocessing steps to image.
        
        Args:
            image: Input image array
            
        Returns:
            Preprocessed image array
        """
        return self.preprocessor.preprocess(image)
    
    def compute_threshold_only(self, image: np.ndarray) -> Union[float, np.ndarray]:
        """Compute threshold on preprocessed image.
        
        Args:
            image: Preprocessed image array
            
        Returns:
            Threshold value or array
        """
        return self.threshold_computer.compute(image)
    
    @staticmethod
    def apply_threshold(image: np.ndarray, threshold: Union[float, np.ndarray]) -> np.ndarray:
        """Apply threshold to create binary mask.
        
        Args:
            image: Image array
            threshold: Threshold value (scalar) or array (per-pixel)
            
        Returns:
            Binary mask array
        """
        if np.isscalar(threshold):
            return image > np.asarray(threshold, dtype=float)
        else:
            return image > threshold
    
    def update_config(self, **kwargs) -> 'ThresholdClassifier':
        """Create new classifier with updated configuration.
        
        Args:
            **kwargs: Configuration parameters to update
            
        Returns:
            New ThresholdClassifier instance with updated config
        """
        config_dict = self.config.to_dict()
        config_dict.update(kwargs)
        new_config = ThresholdConfig.from_dict(config_dict)
        return ThresholdClassifier(new_config)
    
    def get_available_methods(self) -> List[str]:
        """Get list of available thresholding methods.
        
        Returns:
            List of method names
        """
        return ThresholdComputer.AVAILABLE_METHODS.copy()


def create_classifier(
    method: Method = 'otsu',
    **kwargs
) -> ThresholdClassifier:
    """Convenience function to create a ThresholdClassifier.
    
    Args:
        method: Thresholding method to use
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured ThresholdClassifier instance
    """
    config_dict = {'method': method}
    config_dict.update(kwargs)
    
    config = ThresholdConfig.from_dict(config_dict)
    return ThresholdClassifier(config)


def create_activity_labeled_image(label_data, classification_data, label_dict=None):
    return _create_activity_labeled_image(
        label_data,
        classification_data,
        label_dict=label_dict,
    )


__all__ = [
    'ThresholdClassifier',
    'apply_threshold_classification',
    'create_activity_labeled_image',
    'create_classifier',
]
