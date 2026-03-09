"""
Base predictor class for cell segmentation models.

This module defines the abstract base class that all model predictors
should inherit from, ensuring a consistent interface across different
segmentation models.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
from pathlib import Path


class BasePredictor(ABC):
    """
    Abstract base class for all cell segmentation predictors.
    
    This class defines the interface that all model predictors must implement
    to ensure consistency across different segmentation models (Cellpose, Omnipose, etc.).
    """
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the predictor.
        
        Args:
            model_name: Name of the model for identification
            **kwargs: Additional model-specific parameters
        """
        self.model_name = model_name
        self.model = None
        
    @abstractmethod
    def load_model(self, model_path: Optional[str] = None, **kwargs) -> None:
        """
        Load the segmentation model.
        
        Args:
            model_path: Path to the model weights/checkpoints
            **kwargs: Additional loading parameters
        """
        pass
    
    @abstractmethod
    def predict(self, image: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Run inference on a single image.
        
        Args:
            image: Input image as numpy array
            **kwargs: Additional prediction parameters
            
        Returns:
            Tuple of (masks, metadata) where:
            - masks: Segmentation masks as numpy array
            - metadata: Dictionary containing flows, probabilities, etc.
        """
        pass

    def predict_3d(
        self,
        image: np.ndarray,
        do_2d: bool = False,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Run prediction on a 3D image.
        
        Args:
            image: 3D numpy array (z, y, x)
            **kwargs: Additional prediction parameters
            
        Returns:
            Tuple of (masks, metadata) for the 3D image
        """
        if image.ndim != 3:
            raise ValueError("Input must be a 3D array (z, y, x)")
        
        if do_2d:
            # Process each Z-slice independently
            all_masks = []
            all_metadata = []
            
            for z_idx in range(image.shape[0]):
                slice_img = image[z_idx]
                masks, metadata = self.predict(slice_img, **kwargs)

                all_masks.append(masks)
                all_metadata.append(metadata)
            
            # Stack masks
            stacked_masks = np.stack(all_masks, axis=0)
            
            # Combine metadata
            combined_metadata = {
                'per_slice_metadata': all_metadata,
                'total_cells': sum([m['num_cells'] for m in all_metadata]),
                'stack_shape': image.shape,
                'processing_mode': '2d_per_slice'
            }
            
            return stacked_masks, combined_metadata
        
        # Default to 3D prediction
        else:
            return self.predict(image, **kwargs)

    def predict_batch(self, images: list, **kwargs) -> Tuple[list, list]:
        """
        Run inference on a batch of images.
        
        Args:
            images: List of input images as numpy arrays
            **kwargs: Additional prediction parameters
            
        Returns:
            Tuple of (masks_list, metadata_list)
        """
        masks_list = []
        metadata_list = []
        
        for image in images:
            masks, metadata = self.predict(image, **kwargs)
            masks_list.append(masks)
            metadata_list.append(metadata)
            
        return masks_list, metadata_list
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        pass
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None

    def validate_input(self, image: np.ndarray) -> None:
        """
        Validate input image format.
        
        Args:
            image: Input image to validate
            
        Raises:
            ValueError: If image format is invalid
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Input must be a numpy array")
        
        if image.ndim not in [2, 3]:
            raise ValueError(f"Image must be 2D or 3D, got {image.ndim}D")
        
        if image.size == 0:
            raise ValueError("Image cannot be empty")
