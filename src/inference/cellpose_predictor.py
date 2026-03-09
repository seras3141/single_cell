"""
Cellpose predictor implementation for cell segmentation.

This module provides a concrete implementation of the BasePredictor
for Cellpose models, handling both 2D and 3D segmentation tasks.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import logging

try:
    from cellpose import models, transforms
    CELLPOSE_AVAILABLE = True
except ImportError:
    CELLPOSE_AVAILABLE = False
    logging.warning("Cellpose not available. Install cellpose to use CellposePredictor.")

from .base_predictor import BasePredictor


class CellposePredictor(BasePredictor):
    """
    Cellpose model predictor for cell segmentation.
    
    This class provides a wrapper around Cellpose models with a standardized
    interface for inference on both 2D and 3D microscopy images.
    """
    
    def __init__(
        self,
        model_type: str = "cyto3",
        gpu: bool = True,
        device: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Cellpose predictor.
        
        Args:
            model_type: Type of Cellpose model ('cyto', 'cyto2', 'cyto3', 'nuclei', etc.)
            gpu: Whether to use GPU for inference
            **kwargs: Additional parameters
        """
        super().__init__(f"cellpose_{model_type}")
        
        if not CELLPOSE_AVAILABLE:
            raise ImportError("Cellpose is not installed. Please install cellpose to use this predictor.")
        
        self.model_type = model_type
        self.gpu = gpu
        self.device = device
        self.channels = kwargs.get('channels', [0, 0])  # Default grayscale
        self.diameter = kwargs.get('diameter', None)  # Auto-diameter
        self.flow_threshold = kwargs.get('flow_threshold', 0.4)
        self.cellprob_threshold = kwargs.get('cellprob_threshold', 0.0)
        self.min_size = kwargs.get('min_size', 30)
        self.normalize = kwargs.get('normalize', True)
        self.invert = kwargs.get('invert', False)

        self.model = None

        model_path = kwargs.get('model_path', None)
        if model_path is None and self.model_type is None:
            raise ValueError("Either model_type or model_path must be specified.")
        
        # Load model immediately
        self.load_model(model_path=model_path)
    
    def load_model(self, model_path: Optional[str] = None, **kwargs) -> bool:
        """
        Load the Cellpose model.
        
        Args:
            model_path: Path to custom model weights. If None, uses pretrained model.
            **kwargs: Additional loading parameters
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            if model_path and Path(model_path).exists():
                # Load custom trained model
                self.model = models.CellposeModel(
                    model_type=None,
                    pretrained_model=model_path, # type: ignore
                    gpu=self.gpu,
                    device=self.device
                )
                self.model_name = f"cellpose_custom_{Path(model_path).stem}"

                logging.info(f"Loaded custom model from {model_path}")

            else:
                # Load pretrained model
                self.model = models.CellposeModel(
                    model_type=self.model_type,
                    gpu=self.gpu,
                    device=self.device
                )
                self.model_name = f"cellpose_{self.model_type}"
    
                logging.info(f"Loaded Cellpose model: {self.model_type}")
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to load Cellpose model: {e}")
            self.model = None
            return False
    
    def predict(
        self,
        image: np.ndarray,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Run Cellpose inference on a single image.
        
        Args:
            image: Input image as numpy array (2D or 3D)
            **kwargs: Override prediction parameters
            
        Returns:
            Tuple of (masks, metadata) where:
            - masks: Segmentation masks as numpy array
            - metadata: Dictionary containing flows, styles, and parameters
        """
        if self.model is None or not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        self.validate_input(image)
        
        # Override default parameters with any provided kwargs
        channels = kwargs.get('channels', self.channels)
        diameter = kwargs.get('diameter', self.diameter)
        flow_threshold = kwargs.get('flow_threshold', self.flow_threshold)
        cellprob_threshold = kwargs.get('cellprob_threshold', self.cellprob_threshold)
        min_size = kwargs.get('min_size', self.min_size)
        normalize = kwargs.get('normalize', self.normalize)
        invert = kwargs.get('invert', self.invert)
        
        try:
            # Run prediction
            masks, flows, styles = self.model.eval(
                image,
                channels=channels,
                diameter=diameter,
                normalize=normalize,
                invert=invert,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold,
                min_size=min_size
            )
            
            # Prepare metadata
            metadata = {
                'flows': flows,
                'styles': styles,
                'parameters': {
                    'channels': channels,
                    'diameter': diameter,
                    'flow_threshold': flow_threshold,
                    'cellprob_threshold': cellprob_threshold,
                    'min_size': min_size,
                    'normalize': normalize,
                    'invert': invert
                },
                'num_cells': len(np.unique(masks)) - 1,  # Exclude background
                'image_shape': image.shape
            }
            
            return masks, metadata
            
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            raise
    
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
            max_existing = 0
            
            for z_idx in range(image.shape[0]):
                slice_img = image[z_idx]
                masks, metadata = self.predict(slice_img, **kwargs)
                
                # Adjust mask values to avoid conflicts between slices
                if len(all_masks) > 0:
                    max_existing = max(max_existing, np.max(all_masks[-1]))
                    masks[masks > 0] += max_existing
                
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

        return self.predict(image, **kwargs)

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded Cellpose model.
        
        Returns:
            Dictionary containing model information
        """
        if not self.is_loaded():
            return {'status': 'not_loaded'}
        
        return {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'gpu_enabled': self.gpu,
            'default_parameters': {
                'channels': self.channels,
                'diameter': self.diameter,
                'flow_threshold': self.flow_threshold,
                'cellprob_threshold': self.cellprob_threshold,
                'min_size': self.min_size,
                'normalize': self.normalize,
                'invert': self.invert
            },
            'status': 'loaded'
        }
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for Cellpose inference.
        
        Args:
            image: Raw input image
            
        Returns:
            Preprocessed image
        """
        raise NotImplementedError("Preprocessing is handled internally by Cellpose from v4.")

        # Convert to appropriate format
        processed_img = transforms.convert_image(
            image, 
            channel_axis=None, 
            z_axis=None
        )
        
        # Normalize if needed
        if self.normalize:
            processed_img = transforms.normalize99(processed_img, copy=False)
        
        return processed_img
