"""
Output manager for handling prediction results and file organization.

This module provides utilities for organizing and saving prediction outputs
with proper directory structure and file naming conventions.
"""

import numpy as np
from typing import Dict, Any, Optional, Union
from pathlib import Path
import logging
import json
from datetime import datetime

from src.utils.image_utils import LABEL_FORMATS, save_labels, load_labels

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    from cellpose import plot
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logging.warning("Plotting libraries not available. Visualization outputs will be skipped.")


class OutputManager:
    """
    Manages output directory structure and file saving for predictions.
    
    Creates organized output directories and handles saving of segmentation
    masks, visualizations, and metadata with consistent naming conventions.
    """
    
    def __init__(
        self,
        base_output_dir: Union[str, Path],
        model_name: str = "",
        dataset_name: str = "test",
        create_subdirs: bool = True,
        label_format: str = "zarr",
        pred_mask_suffix: str = "_pred_mask",
        overwrite: bool = False,
    ):
        """
        Initialize output manager.

        Args:
            base_output_dir: Base directory for all outputs
            model_name: Name of the model used for predictions
            dataset_name: Name of the dataset being processed
            create_subdirs: Whether to create subdirectories for different output types
            label_format: Format for saving segmentation labels. One of
                ``"tif"``, ``"zarr"`` (default), or ``"hdf5"``.
            pred_mask_suffix: Suffix for prediction mask files
            overwrite: If False (default), skip files whose mask already exists.
        """
        if label_format not in LABEL_FORMATS:
            raise ValueError(f"label_format must be one of {list(LABEL_FORMATS)}; got {label_format!r}")
        self.label_format = label_format
        self._label_ext = LABEL_FORMATS[label_format]

        self.base_output_dir = Path(base_output_dir)
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.create_subdirs = create_subdirs
        self.pred_mask_suffix = pred_mask_suffix
        self.overwrite = overwrite
        
        # Create main output directory: {out_dir}/{model_name}/{dataset}
        self.output_dir = self.base_output_dir / model_name / dataset_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories if requested
        if create_subdirs:
            self.masks_dir = self.output_dir / "masks"
            self.overlays_dir = self.output_dir / "overlays"
            self.metadata_dir = self.output_dir / "metadata"
            
            self.masks_dir.mkdir(exist_ok=True)
            self.overlays_dir.mkdir(exist_ok=True)
            self.metadata_dir.mkdir(exist_ok=True)
        else:
            self.masks_dir = self.output_dir
            self.overlays_dir = self.output_dir
            self.metadata_dir = self.output_dir
        
        # Initialize run metadata
        self.run_metadata = {
            'model_name': model_name,
            'dataset_name': dataset_name,
            'output_directory': str(self.output_dir),
            'created_at': datetime.now().isoformat(),
            'processed_files': []
        }
        
        logging.info(f"Output manager initialized. Results will be saved to: {self.output_dir}")
    
    def save_prediction(
        self,
        masks: np.ndarray,
        metadata: Dict[str, Any],
        input_path: Union[str, Path],
        original_image: Optional[np.ndarray] = None,
        save_overlay: bool = True,
        save_metadata: bool = True
    ) -> Dict[str, Path]:
        """
        Save prediction results with organized file structure.
        
        Args:
            masks: Segmentation masks
            metadata: Prediction metadata
            input_path: Path to the original input file
            original_image: Original image for overlay visualization
            suffix: Suffix to add to output filename
            save_overlay: Whether to save overlay visualization
            save_metadata: Whether to save metadata JSON
            
        Returns:
            Dictionary mapping output type to saved file paths
        """
        input_path = Path(input_path)
        base_name = self._get_output_filename(input_path)
        suffix = self.pred_mask_suffix

        saved_files = {}

        mask_path = self.masks_dir / f"{base_name}{suffix}{self._label_ext}"
        if mask_path.exists() and not self.overwrite:
            logging.info(f"Skipping {input_path.name} — mask already exists at {mask_path}")
            return {}

        try:
            # Save masks
            self._save_masks(masks, mask_path)
            saved_files['masks'] = mask_path
            
            # Save overlay visualization if requested and possible
            if save_overlay and PLOTTING_AVAILABLE and original_image is not None:
                overlay_path = self.overlays_dir / f"{base_name}_overlay.png"
                self._save_overlay(original_image, masks, metadata, overlay_path)
                saved_files['overlay'] = overlay_path
            
            # Save metadata if requested
            if save_metadata:
                metadata_path = self.metadata_dir / f"{base_name}_metadata.json"
                self._save_metadata(metadata, input_path, metadata_path)
                saved_files['metadata'] = metadata_path
            
            # Update run metadata
            self.run_metadata['processed_files'].append({
                'input_file': str(input_path),
                'output_files': {k: str(v) for k, v in saved_files.items()},
                'processed_at': datetime.now().isoformat(),
                'num_cells': metadata.get('num_cells', 0)
            })
            
            logging.info(f"Saved prediction results for {input_path.name}")
            
        except Exception as e:
            logging.error(f"Failed to save prediction for {input_path.name}: {e}")
            raise
        
        return saved_files
    
    def save_z_stack_prediction(
        self,
        masks_stack: np.ndarray,
        metadata: Dict[str, Any],
        input_path: Union[str, Path],
        original_stack: Optional[np.ndarray] = None,
        save_individual_slices: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Save Z-stack prediction results.
        
        Args:
            masks_stack: 3D array of segmentation masks
            metadata: Prediction metadata
            input_path: Path to original input file
            original_stack: Original image stack for overlays
            save_individual_slices: Whether to save individual Z-slices
            **kwargs: Additional arguments for save_prediction
            
        Returns:
            Dictionary containing saved file information
        """
        input_path = Path(input_path)
        base_name = self._get_output_filename(input_path)

        stack_path = self.masks_dir / f"{base_name}_stack{self._label_ext}"
        if stack_path.exists() and not self.overwrite:
            logging.info(f"Skipping {input_path.name} — z-stack mask already exists at {stack_path}")
            return {'stack': {}, 'slices': []}

        saved_files = {'stack': {}, 'slices': []}

        # Save entire stack
        self._save_masks(masks_stack, stack_path)
        saved_files['stack']['masks'] = stack_path
        
        # Save stack metadata
        if kwargs.get('save_metadata', True):
            stack_metadata_path = self.metadata_dir / f"{base_name}_stack_metadata.json"
            self._save_metadata(metadata, input_path, stack_metadata_path)
            saved_files['stack']['metadata'] = stack_metadata_path
        
        # Save individual slices if requested
        if save_individual_slices and masks_stack.ndim == 3:
            for z_idx in range(masks_stack.shape[0]):
                slice_masks = masks_stack[z_idx]
                slice_name = f"{base_name}_z{z_idx:03d}"
                
                slice_path = self.masks_dir / f"{slice_name}_masks{self._label_ext}"
                self._save_masks(slice_masks, slice_path)
                
                slice_info = {'z_index': z_idx, 'masks': slice_path}
                
                # Save slice overlay if original stack is provided
                if (original_stack is not None and 
                    PLOTTING_AVAILABLE and 
                    kwargs.get('save_overlay', True)):
                    
                    slice_image = original_stack[z_idx]
                    slice_metadata = metadata.get('per_slice_metadata', [{}])[z_idx] if z_idx < len(metadata.get('per_slice_metadata', [])) else {}
                    
                    overlay_path = self.overlays_dir / f"{slice_name}_overlay.png"
                    self._save_overlay(slice_image, slice_masks, slice_metadata, overlay_path)
                    slice_info['overlay'] = overlay_path
                
                saved_files['slices'].append(slice_info)
        
        return saved_files
    
    def finalize_run(self) -> Path:
        """
        Finalize the prediction run and save summary metadata.
        
        Returns:
            Path to the saved run summary file
        """
        # Add summary statistics
        self.run_metadata['summary'] = {
            'total_files_processed': len(self.run_metadata['processed_files']),
            'total_cells_detected': sum(
                file_info.get('num_cells', 0) 
                for file_info in self.run_metadata['processed_files']
            ),
            'completed_at': datetime.now().isoformat()
        }
        
        # Save run summary
        summary_path = self.output_dir / "run_summary.json"
        with open(summary_path, 'w') as f:
            # Use the same JSON serialization fix for run metadata
            serializable_metadata = self._prepare_metadata_for_json(self.run_metadata)
            json.dump(serializable_metadata, f, indent=2)
        
        logging.info(f"Prediction run completed. Summary saved to: {summary_path}")
        return summary_path
    
    def _get_output_filename(self, input_path: Path) -> str:
        """
        Generate output filename from input path.
        
        Args:
            input_path: Path to input file
            
        Returns:
            Base name for output files
        """
        # Remove common suffixes like '_BF', '_Cells', etc.
        base_name = input_path.stem
        
        # Remove common microscopy suffixes
        suffixes_to_remove = ['_BF', '_Cells', '_DAPI', '_GFP', '_RFP']
        for suffix in suffixes_to_remove:
            if base_name.endswith(suffix):
                base_name = base_name[:-len(suffix)]
                break
        
        return base_name
    
    def _save_masks(self, masks: np.ndarray, output_path: Path) -> None:
        """Save segmentation masks using the configured label format."""
        try:
            save_labels(masks, output_path)

        except Exception as e:
            logging.error(f"Failed to save masks to {output_path}: {e}")
            raise

    @staticmethod
    def load_masks(path: Union[str, Path]) -> np.ndarray:
        """Load segmentation masks from tif, zarr, or hdf5 file."""
        return load_labels(Path(path))
    
    def _save_overlay(
        self,
        image: np.ndarray,
        masks: np.ndarray,
        metadata: Dict[str, Any],
        output_path: Path
    ) -> None:
        """Save overlay visualization of segmentation results."""
        if not PLOTTING_AVAILABLE:
            return
        
        try:
            # Create figure
            fig = plt.figure(figsize=(12, 8))
            
            # Get flows if available
            flows = metadata.get('flows', [None])[0] if metadata.get('flows') else None
            
            # Create overlay plot
            plot.show_segmentation(fig, image, masks, flows)
            
            # Add title with metadata
            num_cells = metadata.get('num_cells', 0)
            plt.suptitle(f'Segmentation Results - {num_cells} cells detected', fontsize=14)
            
            # Save figure
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            plt.close('all')
            
        except Exception as e:
            logging.error(f"Failed to save overlay to {output_path}: {e}")
            # Don't raise - overlay is optional
    
    def _save_metadata(
        self,
        metadata: Dict[str, Any],
        input_path: Path,
        output_path: Path
    ) -> None:
        """Save prediction metadata as JSON file."""
        try:
            # Prepare metadata for JSON serialization
            json_metadata = self._prepare_metadata_for_json(metadata)
            json_metadata['input_file'] = str(input_path)
            json_metadata['saved_at'] = datetime.now().isoformat()
            
            with open(output_path, 'w') as f:
                json.dump(json_metadata, f, indent=2)
                
        except Exception as e:
            logging.error(f"Failed to save metadata to {output_path}: {e}")
            raise
    
    def _prepare_metadata_for_json(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare metadata for JSON serialization by handling numpy arrays.
        
        Args:
            metadata: Original metadata dictionary
            
        Returns:
            JSON-serializable metadata dictionary
        """
        def convert_value(value):
            """Recursively convert values to JSON-serializable format."""
            if isinstance(value, np.ndarray):
                # Convert numpy arrays to lists, but only for small arrays
                if value.size < 1000:
                    return value.tolist()
                else:
                    return f"<numpy array of shape {value.shape}>"
            elif isinstance(value, (np.integer, np.floating)):
                return value.item()
            elif isinstance(value, dict):
                return {k: convert_value(v) for k, v in value.items()}
            elif isinstance(value, (list, tuple)):
                return [convert_value(item) for item in value]
            elif hasattr(value, '__dict__') and not isinstance(value, (str, int, float, bool, type(None))):
                # Handle objects by converting to dict representation
                return f"<{type(value).__name__} object>"
            else:
                return value
        
        return convert_value(metadata) # type: ignore
