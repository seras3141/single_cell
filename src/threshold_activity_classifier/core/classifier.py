"""Main threshold classifier class."""
from __future__ import annotations

from typing import Tuple, Union, Optional, List
import logging
import numpy as np
import pandas as pd
from skimage import filters


from config import ThresholdConfig, Method
from .preprocessing import ImagePreprocessor
from .thresholding import ThresholdComputer

class ThresholdInstanceClassifier:

    def __init__(self, config: Optional[ThresholdConfig] = None):
        self.config = config or ThresholdConfig()
        self.preprocessor = None # Skipping preprocessor for instance classification
        self.threshold_computer = ThresholdComputer(self.config.method, self.config.params)

    def classify_instances(self, metrics_df):

        if self.config.metric not in metrics_df.columns:
            raise ValueError(f"Requested metric '{self.config.metric}' not found in metrics DataFrame. Available columns: {metrics_df.columns.tolist()}")

        # Validate metric selection
        if self.config.metric not in metrics_df.columns:
            raise ValueError(f"Requested metric '{self.config.metric}' not found. Available: {metrics_df.columns.tolist()}")

        # Set metric values for classification
        metrics_df['metric_value'] = metrics_df[self.config.metric].astype(float)

        if self.config.method == "manual":
            manual_threshold = self.config.params.manual_value
            if manual_threshold is None:
                raise ValueError("Manual threshold method requires 'manual_value' parameter to be set.")
            fixed_thresh = float(manual_threshold)
            metrics_df['threshold'] = fixed_thresh
            metrics_df['is_active'] = metrics_df['metric_value'] > fixed_thresh
            logging.info(f"Using manual threshold: {fixed_thresh}")
        elif self.config.method in ['otsu', 'percentile']:
            if self.config.per_image:
                # Compute per-image percentile thresholds
                thresholds = {}
                for name, g in metrics_df.groupby('image'):
                    vals = g['metric_value'].values
                    if vals.size > 0:
                        thresholds[name] = self.threshold_computer.compute(vals)
                    else:
                        thresholds[name] = 0
                metrics_df['threshold'] = metrics_df['image'].map(thresholds)
                metrics_df['is_active'] = metrics_df['metric_value'] > metrics_df['threshold']
            else:
                # Global percentile threshold
                thresh = self.threshold_computer.compute(metrics_df['metric_value'].values)
                metrics_df['threshold'] = thresh
                metrics_df['is_active'] = metrics_df['metric_value'] > thresh
        else:
            raise ValueError(f"Unsupported thresholding method: {self.config.method}")

        return metrics_df

# Apply threshold-based classification - legacy function, may need refactor for cleaner separation of concerns and better handling of per-image thresholds
# TODO - refactor to separate threshold computation from classification logic, and to handle per-image thresholds more cleanly
def apply_threshold_classification(
    metrics_df : pd.DataFrame,
    threshold_metric : str, threshold_method: str, 
    manual_value: float, per_image_threshold: bool):
    """
    Apply threshold-based classification to determine active vs dead cells
    
    Args:
    - metrics_df: DataFrame containing instance metrics, must include 'image' and threshold_metric columns
    - threshold_metric: Name of the metric column to use for thresholding
    - threshold_method: Method to compute threshold ('otsu', 'percentile', 'manual', or numeric value)
    - manual_value: Value to use if threshold_method is 'manual'
    - per_image_threshold: Whether to compute thresholds separately for each image (only applicable for 'otsu' and 'percentile' methods)

    Returns:
    - DataFrame with additional columns for threshold value, classification result, and labels
    """

    active_label = 'active'
    dead_label = 'dead'

    # Validate metric selection
    if threshold_metric not in metrics_df.columns:
        raise ValueError(f"Requested metric '{threshold_metric}' not found. Available: {metrics_df.columns.tolist()}")
    
    # Set metric values for classification
    metrics_df['metric_value'] = metrics_df[threshold_metric].astype(float)
    
    # Apply threshold based on method
    if threshold_method == "manual":
        manual_threshold = manual_value
        # Use manually specified threshold
        fixed_thresh = float(manual_threshold)
        metrics_df['threshold'] = fixed_thresh
        metrics_df['is_active'] = metrics_df['metric_value'] > fixed_thresh
        logging.info(f"Using manual threshold: {fixed_thresh}")

    elif threshold_method == "percentile":
        # Compute global percentile threshold
        threshold_percentile = manual_value
        if per_image_threshold:
            # Compute per-image percentile thresholds
            thresholds = {}
            for name, g in metrics_df.groupby('image'):
                vals = g['metric_value'].values
                if vals.size > 0: # type: ignore
                    thresholds[name] = np.percentile(vals, threshold_percentile) # type: ignore
                else:
                    thresholds[name] = 0
            metrics_df['threshold'] = metrics_df['image'].map(thresholds)
            metrics_df['is_active'] = metrics_df['metric_value'] > metrics_df['threshold']
            logging.info(f"Using per-image {threshold_percentile}th percentile thresholds")
        else:
            # Global percentile threshold
            thresh = np.percentile(metrics_df['metric_value'].values, threshold_percentile) # type: ignore
            metrics_df['threshold'] = thresh
            metrics_df['is_active'] = metrics_df['metric_value'] > thresh
            logging.info(f"Using global {threshold_percentile}th percentile threshold: {thresh}")

    elif isinstance(threshold_method, (int, float)):
        # Use numeric threshold directly
        fixed_thresh = float(threshold_method)
        metrics_df['threshold'] = fixed_thresh
        metrics_df['is_active'] = metrics_df['metric_value'] > fixed_thresh
        logging.info(f"Using fixed threshold: {fixed_thresh}")

    elif threshold_method == "otsu":
        if per_image_threshold:
            # Compute per-image Otsu threshold on metric_value
            thresholds = {}
            for name, g in metrics_df.groupby('image'):
                vals = g['metric_value'].values
                if vals.size < 2: # type: ignore
                    thresholds[name] = vals.mean() if len(vals) > 0 else 0 # type: ignore
                else:
                    try:
                        thresholds[name] = float(filters.threshold_otsu(vals))
                    except Exception:
                        thresholds[name] = float(np.median(vals)) # type: ignore
            metrics_df['threshold'] = metrics_df['image'].map(thresholds)
            metrics_df['is_active'] = metrics_df['metric_value'] > metrics_df['threshold']
            logging.info("Using per-image Otsu thresholds")
        else:
            # Global Otsu threshold
            vals = metrics_df['metric_value'].values
            thresh = float(filters.threshold_otsu(vals))
            metrics_df['threshold'] = thresh
            metrics_df['is_active'] = metrics_df['metric_value'] > thresh
            logging.info(f"Using global Otsu threshold: {thresh}")
    else:
        raise ValueError(f"Unknown threshold_method '{threshold_method}'. Use 'otsu', 'manual', or numeric value.")
    
    # Add classification labels
    metrics_df['cell_status'] = metrics_df['is_active'].map({True: active_label, False: dead_label})
    
    return metrics_df

# Apply the classification
# metrics_df = apply_threshold_classification(metrics_df, threshold_metric, threshold_method, 
#                                         manual_threshold, per_image_threshold)
    


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
    """
    Create a labeled image where each instance is colored by its activity status
    
    Parameters:
    -----------
    label_data : numpy.ndarray
        Instance segmentation labels
    classification_data : pandas.DataFrame
        DataFrame containing label IDs and is_active classification (of a given image)
    label_dict : Dict
        Labels for active and dead classes
        
    Returns:
    --------
    activity_labels : numpy.ndarray
        Label image where active cells = original label ID, dead cells = -(label ID)
    """
    # Ensure classification_data has required columns
    required_cols = {'label', 'is_active', 'image'}
    if not required_cols.issubset(classification_data.columns):
        raise ValueError(f"Classification data must contain columns: {required_cols}")
    
    # Make sure image_name is unique in classification_data
    if classification_data['image'].nunique() > 1:
        raise ValueError("Classification data contains multiple images. Please filter to a single image before calling this function.")

    activity_labels = np.zeros_like(label_data, dtype=np.int32)
    
    # Create lookup dictionary for activity status
    activity_dict = dict(zip(classification_data['label'], classification_data['is_active']))
    
    # Assign labels based on activity: positive for active, negative for dead
    for label_id in np.unique(label_data):
        if label_id == 0:  # Skip background
            continue
            
        mask = label_data == label_id
        is_active = activity_dict.get(label_id, False)  # Default to dead if not found

        if label_dict:
            activity_labels[mask] = label_dict['active'] if is_active else label_dict['dead']
        else:
            # Positive label for active cells # Negative label for dead cells
            activity_labels[mask] = label_id if is_active else -label_id

    return activity_labels


__all__ = ['ThresholdClassifier', 'create_classifier']
