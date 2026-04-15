"""
Main inference pipeline for running cell segmentation predictions.

This module provides a high-level interface for running inference
on datasets using various segmentation models with organized output.
"""
from __future__ import annotations

import os
import numpy as np
from typing import Dict, Any, Optional, Union, List, Callable
from pathlib import Path
import logging
import tifffile as tiff
import torch
from tqdm import tqdm

from .base_predictor import BasePredictor
from .cellpose_predictor import CellposePredictor
from .output_manager import OutputManager

from src.utils.config import ConfigManager
from src.utils.conversion import combine_2d_to_3d
from src.utils.image_utils import load_image

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
        log_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize inference pipeline.
        
        Args:
            predictor: Model predictor instance
            output_manager: Output management instance
            log_config: Configuration dictionary
        """
        self.predictor = predictor
        self.output_manager = output_manager
        self.log_config = log_config or {}
        
        # Set up logging
        self._setup_logging()
        
        logging.info(f"Inference pipeline initialized with {predictor.model_name}")

    @classmethod
    def _resolve_config(
        cls,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[Union[str, Path]]=None,
        overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Resolve configuration from dict or YAML file with optional overrides.
        
        Args:
            config: A full configuration dict
            config_path: Path to a YAML config file
            overrides: Additional overrides (dot notation or flat dict)
        
        Returns:
            Resolved configuration dictionary
        """
        if (config_path is None and config is None) or (config_path and config):
            raise ValueError("Provide either `config_path` or `config`, but not both.")
        
        if config_path:
            base_config = ConfigManager(config_path)
            if overrides:
                base_config = base_config.merge_with_overrides(overrides)
            resolved_config = base_config.to_dict()
            
        else:
            from omegaconf import OmegaConf
            base_config = OmegaConf.create(config)
            if overrides:
                base_config = OmegaConf.merge(base_config, OmegaConf.create(overrides))
            resolved_config = OmegaConf.to_container(base_config, resolve=True)

        assert isinstance(resolved_config, dict), "Resolved config must be a dictionary"
        return resolved_config # type: ignore
    
    @classmethod
    def from_config(
        cls,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[Union[str, Path]]=None,
        output_dir: Optional[Union[str, Path]]=None,
        dataset_name: Optional[str] = "test",
        model_name: Optional[str]=None,
        device: Optional[Union[str, torch.device]]=None,
        **kwargs
    ) -> "InferencePipeline":
        """
        Create pipeline from a config dictionary or YAML file path.

        Precedence:
            1. Explicit keyword overrides
            2. YAML file (if provided)
            3. Direct config dict (fallback)
        
        Args:
            config: A full configuration dict
            config_path: Path to a YAML config file
            output_dir: Optional output dir override
            dataset_name: Optional dataset override
            model_name: Optional model name override
            device: Optional device override
            overrides: Additional overrides (dot notation or flat dict)
        
        Returns:
            Initialized InferencePipeline instance
        """
        if (config_path is None and config is None) or (config_path and config):
            raise ValueError("Provide either `config_path` or `config`, but not both.")

        # Step 1+2: Resolve configuration
        resolved_config = InferencePipeline._resolve_config(
            config=config,
            config_path=config_path,
            overrides=kwargs
        )
        
        # Step 3: Instantiate predictor
        segmentation_config = resolved_config.get("segmentation", {})
        if not segmentation_config:
            raise ValueError("Missing `segmentation` config section.")
        
        # Ensure cellpose config exists
        cellpose_cfg = segmentation_config.get("cellpose", {})
        if not cellpose_cfg:
            raise ValueError("Missing `segmentation.cellpose` config.")

        if model_name is not None:
            cellpose_cfg["model_type"] = model_name
        if device is not None:
            cellpose_cfg["device"] = device

        predictor = CellposePredictor(**cellpose_cfg)

        # Step 4: Prepare output manager
        inference_config = segmentation_config.get("inference", {})
        paths_config = resolved_config.get("paths", {})
        results_folder = inference_config.get("results_folder", "results")

        model_name = cellpose_cfg.get("model_type", "cyto3")
        output_dir = output_dir or Path(paths_config.get("output_dir", "results")) / results_folder        
        dataset_name = dataset_name or inference_config.get("dataset_name", "test")
        
        # Initialize output manager
        output_manager = OutputManager(
            base_output_dir=output_dir,     # type: ignore
            model_name=model_name,          # type: ignore
            dataset_name=dataset_name       # type: ignore
        )

        log_config = resolved_config.get("logging", {})

        return cls(predictor, output_manager, log_config)
    
    @classmethod
    def get_file_list(
        cls,
        input_dir: Union[str, Path],
        file_pattern: str = "*_BF.tif"
    ) -> List[Path]:
        """
        Get list of input files matching the pattern in the input directory.
        
        Args:
            input_dir: Directory containing input images
            file_pattern: Glob pattern for input files
        Returns:
            List of matching file paths
        """
        input_dir = Path(input_dir)
        
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        input_files = sorted(input_dir.glob(file_pattern))
        
        if not input_files:
            raise ValueError(f"No files found matching pattern '{file_pattern}' in {input_dir}")

        return input_files

    def run_inference(
        self,
        input_dir: Optional[Union[str, Path]] = None,
        input_files: Optional[List[Path]] = None,
        file_pattern: str = "*_BF.tif",
        process_z_stacks: bool = False,
        save_overlays: bool = True,
        save_metadata: bool = True,
        combine_z_stacks: bool = True,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Run inference on all files in the input directory and generates segmentation masks.

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

        # Either read files from directory or use provided list
        if input_dir:
            if input_files:
                raise ValueError("Provide either `input_dir` or `input_files`, not both.")
            input_files = self.get_file_list(input_dir, file_pattern)
        elif input_files is None:
            raise ValueError("Either `input_dir` or `input_files` must be provided.")
        
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
        if input_files and results['2d_files'] and combine_z_stacks:
            self._combine_2d_to_3d()
        
        # Finalize run
        summary_path = self.output_manager.finalize_run()
        results['summary_path'] = summary_path
        
        logging.info(f"Inference completed. Processed {len(results['processed_files'])} files successfully")
        logging.info(f"Total cells detected: {results['total_cells']}")
        
        if results['failed_files']:
            logging.warning(f"Failed to process {len(results['failed_files'])} files")
        
        return results
    
    def _combine_2d_to_3d(self) -> None:
        """
        Combine 2D mask slices into 3D volumes.
        """
        from src.utils.conversion import combine_2d_to_3d

        logging.info("Combining 2D masks into 3D volumes")
        mask_dir = self.output_manager.masks_dir
        output_dir = self.output_manager.masks_dir.parent / (self.output_manager.masks_dir.name + "_3d")

        pattern = r"(.+?)_z(\d+)(?:_(masks))?\.(tif|tiff)"
        combine_2d_to_3d(str(mask_dir), str(output_dir), pattern=pattern)
    
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
        file_path: Union[str, Path],
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
            try:
                image = load_image(file_path)        
            except Exception as e:
                logging.error(f"Failed to load image {file_path}: {e}")
                raise
            
            if image.ndim == 3:
                # raise NotImplementedError("Z-stack processing for 3D data not implemented yet")
                # Process as 3D Z-stack
                masks, metadata = self.predictor.predict_3d(image, do_2d=process_z_stacks, **kwargs)

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

    
    def _setup_logging(self) -> None:
        """Set up logging configuration."""

        from src.utils.logging_utils import setup_logging

        log_level = self.log_config.get('level', 'INFO')
        log_file = self.log_config.get('filename', 'inference.log')
        log_file = self.output_manager.output_dir / log_file

        setup_logging(log_level, log_file) # type: ignore
            
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
