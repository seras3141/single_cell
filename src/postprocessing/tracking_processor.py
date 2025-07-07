"""
3D Cell Tracking Processor - Complete postprocessing pipeline for 3D cell tracking with blur filtering.

This module provides a comprehensive postprocessing pipeline that combines:
- 3D cell tracking across z-stacks using trackpy
- Blur-based quality filtering using sharpness measurements
- Batch processing of multiple segmentation files
- Robust file handling and output management

Based on the track_cells.py implementation with improved modularity and testing.
"""

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
import pandas as pd
import tifffile
import trackpy as tp
from skimage.measure import regionprops
from tqdm import tqdm
from glob import glob

from ..preprocessing.blur_measure import measure_blur_heatmap
from .cell_tracking import CellTracker3D, TrackingConfig
from .blur_filtering import BlurFilter, FilterConfig, blur_intensity_metric

logger = logging.getLogger(__name__)


@dataclass
class TrackingProcessorConfig:
    """Configuration for the complete 3D tracking processor."""
    
    # File patterns and paths
    mask_pattern: str = "*_3d.tif"
    """Pattern to match mask files"""
    
    image_suffix_mapping: Dict[str, str] = field(default_factory=lambda: {
        "_3d.tif": "_BF_3d.tif"
    })
    """Mapping from mask suffix to image suffix"""
    
    blur_heatmap_suffix: str = "_blur_heatmap_32_8.tif"
    """Suffix for blur heatmap files"""
    
    output_suffix: str = "_tracked"
    """Suffix for output files"""
    
    # Processing parameters
    blur_threshold: float = 0.5
    """Threshold for blur filtering"""
    
    invert_blur_threshold: bool = False
    """Whether to invert blur threshold comparison"""
    
    # Tracking configuration
    tracking_config: TrackingConfig = field(default_factory=TrackingConfig)
    """Configuration for cell tracking"""
    
    # Blur filtering configuration  
    filter_config: FilterConfig = field(default_factory=FilterConfig)
    """Configuration for blur filtering"""
    
    # Output options
    create_output_dirs: bool = True
    """Whether to create output directories if they don't exist"""
    
    overwrite_existing: bool = False
    """Whether to overwrite existing output files"""
    
    save_tracking_data: bool = False
    """Whether to save intermediate tracking data as CSV"""


class TrackingProcessor:
    """
    Complete 3D cell tracking processor with blur-based filtering.
    
    This class provides a comprehensive pipeline for processing 3D segmentation
    masks, including cell tracking across z-stacks and quality filtering based
    on blur measurements.
    """
    
    def __init__(self, config: Optional[TrackingProcessorConfig] = None):
        """
        Initialize the tracking processor.
        
        Args:
            config: Processor configuration. If None, uses default config.
        """
        self.config = config or TrackingProcessorConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize sub-components
        self.tracker = CellTracker3D(self.config.tracking_config)
        self.blur_filter = BlurFilter(self.config.filter_config)
        
        # Processing state
        self.processing_stats: Dict[str, Any] = {}
        self.failed_files: List[str] = []
    
    def get_label_centers_with_blur_filtering(
        self, 
        segmentation_mask: np.ndarray,
        sharpness_image: Optional[np.ndarray] = None,
        blur_thresh: float = 0.5,
        inv: bool = False
    ) -> pd.DataFrame:
        """
        Extract cell centers with optional blur-based filtering.
        
        Args:
            segmentation_mask: 2D instance segmentation mask
            sharpness_image: Optional blur/sharpness heatmap
            blur_thresh: Threshold for blur filtering
            inv: Whether to invert the blur threshold comparison
            
        Returns:
            DataFrame with filtered cell properties
        """
        def comparison_func(x, y, inv_flag):
            """Comparison function for blur threshold.
            
            When inv_flag=False: keep cells where blur_value < threshold (sharper cells)
            When inv_flag=True: keep cells where blur_value > threshold (blurrier cells)
            """
            return x > y if inv_flag else x < y
        
        # Extract region properties
        if sharpness_image is not None:
            regions_unfiltered = regionprops(
                segmentation_mask, 
                intensity_image=sharpness_image,
                extra_properties=[blur_intensity_metric]
            )
            # Filter by blur threshold
            regions = [
                region for region in regions_unfiltered
                if comparison_func(region.blur_intensity_metric, blur_thresh, inv)
            ]
        else:
            regions = regionprops(segmentation_mask)
        
        # Filter by area (matching original implementation)
        regions_filtered = [
            region for region in regions 
            if self.config.tracking_config.min_area <= region.area <= self.config.tracking_config.max_area
        ]
        
        if not regions_filtered:
            return pd.DataFrame(columns=['x', 'y', 'label', 'area'])
        
        # Extract properties
        properties = {
            'x': [region.centroid[1] for region in regions_filtered],  # Note: x/y convention
            'y': [region.centroid[0] for region in regions_filtered],
            'label': [region.label for region in regions_filtered],
            'area': [region.area for region in regions_filtered]
        }
        
        # Add blur intensity if available
        if sharpness_image is not None:
            properties['blur_intensity'] = [
                region.blur_intensity_metric for region in regions_filtered
            ]
        
        return pd.DataFrame(properties)
    
    def extract_3d_centers_with_blur(
        self,
        segmentation_stack: np.ndarray,
        sharpness_image: Optional[np.ndarray] = None,
        **kwargs
    ) -> List[Tuple[int, pd.DataFrame]]:
        """
        Extract cell centers from 3D stack with blur filtering.
        
        Args:
            segmentation_stack: 3D segmentation array (z, y, x)
            sharpness_image: Optional 3D sharpness array or list of 2D arrays
            **kwargs: Additional arguments for blur filtering
            
        Returns:
            List of (z_index, properties_dataframe) tuples
        """
        centers_with_z = []
        
        # Handle sharpness image format
        if sharpness_image is None:
            sharpness_slices = [None] * len(segmentation_stack)
        elif isinstance(sharpness_image, np.ndarray):
            if sharpness_image.ndim == 3:
                sharpness_slices = [sharpness_image[z] for z in range(len(segmentation_stack))]
            else:
                # 2D sharpness image - use for all slices
                sharpness_slices = [sharpness_image] * len(segmentation_stack)
        else:
            # List of sharpness images
            sharpness_slices = sharpness_image
        
        # Process each z-slice
        for z, segmentation_mask in enumerate(segmentation_stack):
            sharpness_slice = sharpness_slices[z] if z < len(sharpness_slices) else None
            
            properties = self.get_label_centers_with_blur_filtering(
                segmentation_mask,
                sharpness_slice,
                blur_thresh=kwargs.get('blur_thresh', self.config.blur_threshold),
                inv=kwargs.get('inv', self.config.invert_blur_threshold)
            )
            
            centers_with_z.append((z, properties))
        
        return centers_with_z
    
    def track_3d_centers(
        self,
        segmentation_stack: np.ndarray,
        sharpness_image: Optional[np.ndarray] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Track cell centers across 3D stack with blur filtering.
        
        Args:
            segmentation_stack: 3D segmentation array
            sharpness_image: Optional sharpness array for blur filtering
            **kwargs: Additional tracking parameters
            
        Returns:
            3D array with tracked particle IDs
        """
        self.logger.info(f"Tracking cells in stack of shape {segmentation_stack.shape}")
        
        # Extract centers with blur filtering
        centers_with_z = self.extract_3d_centers_with_blur(
            segmentation_stack, sharpness_image, **kwargs
        )
        
        # Flatten centers into DataFrame for tracking
        all_data = []
        for z, properties in centers_with_z:
            if not properties.empty:
                properties_with_frame = properties.copy()
                properties_with_frame['frame'] = z
                all_data.append(properties_with_frame)
        
        if not all_data:
            self.logger.warning("No cells found for tracking")
            return np.zeros_like(segmentation_stack, dtype=np.int32)
        
        # Combine all data
        tracking_data = pd.concat(all_data, ignore_index=True)
        
        # Perform tracking using trackpy
        self.logger.info(f"Linking {len(tracking_data)} detections across {segmentation_stack.shape[0]} frames")
        tracked_data = tp.link_df(
            tracking_data,
            search_range=self.config.tracking_config.search_range,
            memory=self.config.tracking_config.memory
        )
        
        # Filter short tracks
        if self.config.tracking_config.min_track_length > 1:
            tracked_data = tp.filter_stubs(tracked_data, self.config.tracking_config.min_track_length)
        
        # Create output array with tracked IDs
        tracked_stack = np.zeros_like(segmentation_stack, dtype=np.int32)
        
        for _, row in tracked_data.iterrows():
            z = int(row['frame'])
            particle_id = int(row['particle']) + 1  # Start from 1
            label = int(row['label'])
            
            # Replace all pixels with original label with particle ID
            mask = segmentation_stack[z] == label
            tracked_stack[z][mask] = particle_id
        
        self.logger.info(f"Tracking completed: {tracked_data['particle'].nunique()} particles")
        
        return tracked_stack
    
    def get_or_create_blur_heatmap(
        self,
        image_path: Union[str, Path],
        blur_cache_dir: Union[str, Path],
        force_recompute: bool = False
    ) -> np.ndarray:
        """
        Get blur heatmap from cache or compute it.
        
        Args:
            image_path: Path to source image
            blur_cache_dir: Directory to store blur heatmaps
            force_recompute: Whether to force recomputation
            
        Returns:
            Blur heatmap array
        """
        image_path = Path(image_path)
        blur_cache_dir = Path(blur_cache_dir)
        
        # Construct blur heatmap filename
        blur_filename = image_path.stem + self.config.blur_heatmap_suffix
        blur_path = blur_cache_dir / blur_filename
        
        # Check if blur heatmap exists and should be used
        if blur_path.exists() and not force_recompute:
            self.logger.debug(f"Loading cached blur heatmap: {blur_path}")
            return tifffile.imread(str(blur_path))
        
        # Compute blur heatmap
        self.logger.info(f"Computing blur heatmap for: {image_path}")
        sharpness_image = measure_blur_heatmap(
            str(image_path),
            patch_size=self.config.filter_config.patch_size,
            stride_size=self.config.filter_config.stride_size
        )
        
        # Save blur heatmap
        os.makedirs(blur_cache_dir, exist_ok=True)
        tifffile.imwrite(str(blur_path), sharpness_image.astype(np.float32))
        self.logger.debug(f"Saved blur heatmap: {blur_path}")
        
        return sharpness_image
    
    def process_single_file(
        self,
        mask_path: Union[str, Path],
        image_directory: Union[str, Path],
        output_directory: Union[str, Path],
        blur_directory: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        Process a single mask file with tracking and blur filtering.
        
        Args:
            mask_path: Path to segmentation mask file
            image_directory: Directory containing source images
            output_directory: Directory for output files
            blur_directory: Directory for blur heatmaps
            
        Returns:
            Dictionary with processing results and statistics
        """
        mask_path = Path(mask_path)
        image_directory = Path(image_directory)
        output_directory = Path(output_directory)
        blur_directory = Path(blur_directory)
        
        try:
            # Create output filename
            output_filename = mask_path.stem.replace("_3d", f"_3d_filtered_{self.config.blur_threshold:.1f}")
            output_path = output_directory / f"{output_filename}.tif"
            
            # Skip if output exists and not overwriting
            if output_path.exists() and not self.config.overwrite_existing:
                self.logger.info(f"Skipping existing file: {output_path}")
                return {'status': 'skipped', 'reason': 'file_exists'}
            
            # Load segmentation stack
            self.logger.info(f"Processing: {mask_path}")
            segmentation_stack = tifffile.imread(str(mask_path)).astype(int)
            
            # Find corresponding image file
            image_filename = mask_path.name
            for mask_suffix, image_suffix in self.config.image_suffix_mapping.items():
                if mask_path.name.endswith(mask_suffix):
                    image_filename = mask_path.name.replace(mask_suffix, image_suffix)
                    break
            
            image_path = image_directory / image_filename
            
            # Get or compute blur heatmap
            sharpness_image = self.get_or_create_blur_heatmap(
                image_path, blur_directory
            )
            
            # Perform tracking with blur filtering
            tracked_stack = self.track_3d_centers(
                segmentation_stack,
                sharpness_image=sharpness_image,
                blur_thresh=self.config.blur_threshold,
                inv=self.config.invert_blur_threshold
            )
            
            # Save results
            if self.config.create_output_dirs:
                os.makedirs(output_directory, exist_ok=True)
            
            tifffile.imwrite(str(output_path), tracked_stack)
            
            # Save tracking data if requested
            if self.config.save_tracking_data:
                tracking_data_path = output_path.with_suffix('.csv')
                # Note: Would need to modify track_3d_centers to return tracking data
                # For now, just log the option
                self.logger.debug(f"Tracking data saving requested but not implemented")
            
            result = {
                'status': 'success',
                'input_path': str(mask_path),
                'output_path': str(output_path),
                'input_shape': segmentation_stack.shape,
                'output_shape': tracked_stack.shape,
                'blur_threshold': self.config.blur_threshold
            }
            
            self.logger.info(f"Successfully processed: {mask_path} -> {output_path}")
            return result
            
        except Exception as e:
            error_msg = f"Failed to process {mask_path}: {str(e)}"
            self.logger.error(error_msg)
            self.failed_files.append(str(mask_path))
            return {
                'status': 'error',
                'input_path': str(mask_path),
                'error': str(e)
            }
    
    def process_batch(
        self,
        mask_directory: Union[str, Path],
        image_directory: Union[str, Path], 
        output_directory: Union[str, Path],
        blur_directory: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        Process multiple mask files in batch.
        
        Args:
            mask_directory: Directory containing mask files
            image_directory: Directory containing source images
            output_directory: Directory for output files
            blur_directory: Directory for blur heatmaps
            
        Returns:
            Dictionary with batch processing results
        """
        mask_directory = Path(mask_directory)
        
        # Find mask files
        mask_files = list(mask_directory.glob(self.config.mask_pattern))
        
        if not mask_files:
            raise ValueError(f"No mask files found matching pattern {self.config.mask_pattern} in {mask_directory}")
        
        self.logger.info(f"Found {len(mask_files)} mask files to process")
        
        # Create output directories
        if self.config.create_output_dirs:
            for directory in [output_directory, blur_directory]:
                os.makedirs(directory, exist_ok=True)
        
        # Process files
        results = []
        self.failed_files = []
        
        for mask_file in tqdm(mask_files, desc="Processing masks"):
            result = self.process_single_file(
                mask_file, image_directory, output_directory, blur_directory
            )
            results.append(result)
        
        # Compute summary statistics
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] == 'error']
        skipped = [r for r in results if r['status'] == 'skipped']
        
        summary = {
            'total_files': len(mask_files),
            'successful': len(successful),
            'failed': len(failed),
            'skipped': len(skipped),
            'success_rate': len(successful) / len(mask_files) if mask_files else 0,
            'results': results,
            'failed_files': self.failed_files,
            'config': self.config
        }
        
        self.logger.info(f"Batch processing completed: {len(successful)}/{len(mask_files)} successful")
        if failed:
            self.logger.warning(f"Failed to process {len(failed)} files: {[r['input_path'] for r in failed]}")
        
        return summary


def run_tracking_pipeline(
    mask_directory: Union[str, Path],
    image_directory: Union[str, Path],
    output_directory: Union[str, Path],
    blur_directory: Union[str, Path],
    config: Optional[TrackingProcessorConfig] = None
) -> Dict[str, Any]:
    """
    Run the complete 3D tracking pipeline on a directory of files.
    
    Args:
        mask_directory: Directory containing 3D segmentation masks
        image_directory: Directory containing source images
        output_directory: Directory for tracked output files
        blur_directory: Directory for blur heatmap cache
        config: Processing configuration
        
    Returns:
        Dictionary with processing results and statistics
    """
    processor = TrackingProcessor(config)
    return processor.process_batch(
        mask_directory, image_directory, output_directory, blur_directory
    )


# Convenience function that matches the original main() interface
def main_compatible(
    image_directory: str = "data/BF+IF Experiments_3D_train_test_dataset/train",
    mask_directory: str = "data/BF+IF Experiments_2D_train_test_dataset/predictions_test/pretrained_cell2d_cyto3_FlowT0.4/3d_view",
    output_directory: str = "data/BF+IF Experiments_2D_train_test_dataset/predictions_test/pretrained_cell2d_cyto3_FlowT0.4/3d_view_tracked_SF",
    blur_directory: str = "data/BF+IF Experiments_3D_train_test_dataset/blur_heatmaps",
    blur_thresh: float = 0.5,
    inv: bool = False
) -> Dict[str, Any]:
    """
    Compatibility function that matches the original main() interface.
    
    This function provides the same interface as the original track_cells.py main()
    function while using the new modular architecture.
    """
    # Create configuration
    config = TrackingProcessorConfig(
        blur_threshold=blur_thresh,
        invert_blur_threshold=inv,
        create_output_dirs=True,
        overwrite_existing=True
    )
    
    # Run pipeline
    return run_tracking_pipeline(
        mask_directory=mask_directory,
        image_directory=image_directory,
        output_directory=output_directory,
        blur_directory=blur_directory,
        config=config
    )


if __name__ == "__main__":
    # Run with default parameters (matching original implementation)
    results = main_compatible()
    print(f"Processing completed: {results['successful']}/{results['total_files']} files successful")
