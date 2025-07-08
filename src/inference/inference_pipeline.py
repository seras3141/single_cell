"""
Main inference pipeline for running cell segmentation predictions.

This module provides a high-level interface for running inference
on datasets using various segmentation models with organized output.
"""

import os
import numpy as np
from typing import Dict, Any, Optional, Union, List, Callable
from pathlib import Path
import logging
import tifffile as tiff
from tqdm import tqdm
import yaml
from ..utils.conversion import combine_2d_to_3d

from .base_predictor import BasePredictor
from .cellpose_predictor import CellposePredictor
from .output_manager import OutputManager
from ..utils.config import load_config


class InferencePipeline:
    """
    High-level pipeline for running inference on cell segmentation datasets.
    
    This class orchestrates the entire inference process from loading data
    to saving results with proper organization and logging.
    """
    
    def __init__(
        self,
        predictor: BasePredictor,
        output_manager: OutputManager,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize inference pipeline.
        
        Args:
            predictor: Model predictor instance
            output_manager: Output management instance
            config: Configuration dictionary
        """
        self.predictor = predictor
        self.output_manager = output_manager
        self.config = config or {}
        
        # Set up logging
        self._setup_logging()
        
        logging.info(f"Inference pipeline initialized with {predictor.model_name}")
    
    @classmethod
    def from_config(
        cls,
        config_path: Union[str, Path],
        model_name: str,
        output_dir: Union[str, Path],
        dataset_name: str = "test",
        **kwargs
    ) -> "InferencePipeline":
        """
        Create pipeline from configuration file.
        
        Args:
            config_path: Path to configuration YAML file
            model_name: Name of the model to use
            output_dir: Base output directory
            dataset_name: Name of the dataset
            **kwargs: Additional arguments to override config
            
        Returns:
            Initialized InferencePipeline instance
        """
        config = load_config(config_path)
        
        # Override config with kwargs
        config.update(kwargs)
        
        # Initialize predictor based on model type
        segmentation_config = config.get('segmentation', {})
        if 'cellpose' in segmentation_config:
            cellpose_config = segmentation_config['cellpose']
            predictor = CellposePredictor(**cellpose_config)
        else:
            raise ValueError("No supported model configuration found")
        
        # Initialize output manager
        output_manager = OutputManager(
            base_output_dir=output_dir,
            model_name=model_name,
            dataset_name=dataset_name
        )
        
        return cls(predictor, output_manager, config)
    
    def run_inference(
        self,
        input_dir: Union[str, Path],
        file_pattern: str = "*_BF.tif",
        process_z_stacks: bool = False,
        save_overlays: bool = True,
        save_metadata: bool = True,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Run inference on all files in the input directory.
        
        Args:
            input_dir: Directory containing input images
            file_pattern: Glob pattern for input files
            process_z_stacks: Whether to process as Z-stacks
            save_overlays: Whether to save overlay visualizations
            save_metadata: Whether to save prediction metadata
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary containing run statistics and results
        """
        input_dir = Path(input_dir)
        
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        # Find input files
        input_files = sorted(input_dir.glob(file_pattern))
        
        if not input_files:
            raise ValueError(f"No files found matching pattern '{file_pattern}' in {input_dir}")
        
        logging.info(f"Found {len(input_files)} files to process")
        
        # Process files
        results = {
            'processed_files': [],
            'failed_files': [],
            'total_cells': 0,
            'total_files': len(input_files),
            '2d_files': 0,
        }
        
        for idx, file_path in enumerate(tqdm(input_files, desc="Processing files")):
            try:
                # Process single file
                file_result = self._process_single_file(
                    file_path,
                    process_z_stacks=process_z_stacks,
                    save_overlays=save_overlays,
                    save_metadata=save_metadata
                )
                
                results['processed_files'].append(file_result)
                results['total_cells'] += file_result.get('num_cells', 0)
                if file_result.get('is_2d', False):
                    results['2d_files'] += 1
                
                # Call progress callback if provided
                if progress_callback:
                    progress_callback(idx + 1, len(input_files), file_path)
                
            except Exception as e:
                logging.error(f"Failed to process {file_path}: {e}")
                results['failed_files'].append({
                    'file': str(file_path),
                    'error': str(e)
                })
        
        # Combine 2D masks to 3D if needed
        if process_z_stacks and input_files and results['2d_files']:
            mask_dir = self.output_manager.masks_dir
            output_dir = self.output_manager.masks_dir.parent / (self.output_manager.masks_dir.name + "_3d")

            print(mask_dir, output_dir)  # Debugging line to check directories
            pattern = r"(.+?)_z(\d+)(?:_(masks))?\.(tif|tiff)"
            combine_2d_to_3d(str(mask_dir), str(output_dir), pattern=pattern)
        
        # Finalize run
        summary_path = self.output_manager.finalize_run()
        results['summary_path'] = summary_path
        
        logging.info(f"Inference completed. Processed {len(results['processed_files'])} files successfully")
        logging.info(f"Total cells detected: {results['total_cells']}")
        
        if results['failed_files']:
            logging.warning(f"Failed to process {len(results['failed_files'])} files")
        
        return results
    
    def run_inference_single(
        self,
        input_path: Union[str, Path],
        process_z_stacks: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run inference on a single file.
        
        Args:
            input_path: Path to input file
            process_z_stacks: Whether to process as Z-stack
            **kwargs: Additional arguments for processing
            
        Returns:
            Dictionary containing processing results
        """
        return self._process_single_file(
            input_path,
            process_z_stacks=process_z_stacks,
            **kwargs
        )
    
    def _process_single_file(
        self,
        file_path: Path,
        process_z_stacks: bool = False,
        save_overlays: bool = True,
        save_metadata: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a single input file.
        
        Args:
            file_path: Path to input file
            process_z_stacks: Whether to process as Z-stack
            save_overlays: Whether to save overlays
            save_metadata: Whether to save metadata
            **kwargs: Additional processing arguments
            
        Returns:
            Dictionary containing processing results
        """
        try:
            # Load image
            image = self._load_image(file_path)
            
            if process_z_stacks and image.ndim == 3:
                # Process as Z-stack
                masks, metadata = self.predictor.predict_z_stack(image, **kwargs)
                
                # Save results
                saved_files = self.output_manager.save_z_stack_prediction(
                    masks,
                    metadata,
                    file_path,
                    original_stack=image,
                    save_overlay=save_overlays,
                    save_metadata=save_metadata
                )
                
                num_cells = metadata.get('total_cells', 0)
                
            else:
                # Process as single image (2D or handle 3D as single volume)
                masks, metadata = self.predictor.predict(image, **kwargs)
                
                # Save results
                saved_files = self.output_manager.save_prediction(
                    masks,
                    metadata,
                    file_path,
                    original_image=image,
                    save_overlay=save_overlays,
                    save_metadata=save_metadata
                )
                
                num_cells = metadata.get('num_cells', 0)

            return {
                'file_path': str(file_path),
                'num_cells': num_cells,
                'saved_files': saved_files,
                'image_shape': image.shape,
                'processing_mode': 'z_stack' if process_z_stacks and image.ndim == 3 else 'single',
                'status': 'success',
                'is_2d': image.ndim == 2
            }
            
        except Exception as e:
            logging.error(f"Failed to process {file_path}: {e}")
            return {
                'file_path': str(file_path),
                'status': 'failed',
                'error': str(e)
            }
    
    def _load_image(self, file_path: Path) -> np.ndarray:
        """
        Load image from file.
        
        Args:
            file_path: Path to image file
            
        Returns:
            Loaded image as numpy array
        """
        try:
            if file_path.suffix.lower() in ['.tif', '.tiff']:
                image = tiff.imread(str(file_path))
            else:
                # Try with PIL for other formats
                from PIL import Image
                image = np.array(Image.open(file_path))
            
            return image
            
        except Exception as e:
            logging.error(f"Failed to load image {file_path}: {e}")
            raise
    
    def _setup_logging(self) -> None:
        """Set up logging configuration."""
        log_level = self.config.get('logging', {}).get('level', 'INFO')
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(
                    self.output_manager.output_dir / 'inference.log',
                    mode='w'
                )
            ]
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return self.predictor.get_model_info()
    
    def validate_setup(self) -> Dict[str, bool]:
        """
        Validate that the pipeline is properly set up.
        
        Returns:
            Dictionary indicating validation status
        """
        validation = {
            'predictor_loaded': self.predictor.is_loaded(),
            'output_dir_exists': self.output_manager.output_dir.exists(),
            'output_dir_writable': os.access(self.output_manager.output_dir, os.W_OK)
        }
        
        all_valid = all(validation.values())
        validation['overall'] = all_valid
        
        if not all_valid:
            issues = [k for k, v in validation.items() if not v and k != 'overall']
            logging.warning(f"Pipeline validation failed. Issues: {issues}")
        
        return validation
