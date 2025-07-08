"""
Integrated postprocessing pipeline for 3D cell tracking with blur-based filtering.

This module provides a complete pipeline that combines cell tracking and blur-based
filtering for comprehensive postprocessing of segmentation results.
"""

import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm
import glob

from .cell_tracking import CellTracker3D, TrackingConfig
from .blur_filtering import BlurFilter, FilterConfig, blur_intensity_metric

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the complete postprocessing pipeline."""
    
    # Component configurations
    tracking_config: TrackingConfig = None
    filter_config: FilterConfig = None
    
    # Pipeline options
    enable_blur_filtering: bool = True
    """Whether to apply blur-based filtering"""
    
    # Processing order
    filter_before_tracking: bool = True
    """If True, filter cells before tracking; if False, track then filter"""
    
    # Output options
    save_intermediate_results: bool = True
    """Whether to save intermediate processing results"""
    
    output_format: str = "tiff"
    """Output format for results ('tiff', 'hdf5')"""
    
    # Parallel processing
    n_workers: int = 1
    """Number of parallel workers (not implemented yet)"""
    
    def __post_init__(self):
        """Initialize default configurations if not provided."""
        if self.tracking_config is None:
            self.tracking_config = TrackingConfig()
        if self.filter_config is None:
            self.filter_config = FilterConfig()


class CellTrackingPipeline:
    """
    Complete postprocessing pipeline for 3D cell tracking with quality filtering.
    
    This pipeline integrates:
    1. Blur-based quality assessment and filtering
    2. 3D cell tracking across z-stacks
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the postprocessing pipeline.
        
        Args:
            config: Pipeline configuration. If None, uses default config.
        """
        self.config = config or PipelineConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize components
        self.tracker = CellTracker3D(self.config.tracking_config)
        self.blur_filter = BlurFilter(self.config.filter_config) if self.config.enable_blur_filtering else None
        
        # Results storage
        self.last_results: Optional[Dict[str, Any]] = None
    
    def process_single_file(
        self,
        segmentation_path: Union[str, Path],
        image_path: Union[str, Path],
        output_dir: Union[str, Path],
        blur_cache_dir: Optional[Union[str, Path]] = None,
        filename_prefix: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a single segmentation file through the complete pipeline.
        
        Args:
            segmentation_path: Path to 3D segmentation mask
            image_path: Path to corresponding 3D image
            output_dir: Directory to save results
            blur_cache_dir: Directory for caching blur heatmaps
            filename_prefix: Optional prefix for output filenames
            
        Returns:
            Dictionary with processing results and statistics
        """
        segmentation_path = Path(segmentation_path)
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output filenames
        if filename_prefix is None:
            filename_prefix = segmentation_path.stem
        
        self.logger.info(f"Processing {segmentation_path.name}")
        
        # Load input data
        segmentation_stack = tifffile.imread(str(segmentation_path)).astype(int)
        
        results = {
            'input_segmentation': str(segmentation_path),
            'input_image': str(image_path),
            'input_shape': segmentation_stack.shape,
            'processing_steps': []
        }
        
        current_stack = segmentation_stack.copy()
        
        # Step 1: Get or compute blur heatmap if needed
        blur_heatmap = None
        if self.config.enable_blur_filtering:
            blur_heatmap = self.blur_filter.get_or_compute_blur_heatmap(
                image_path, blur_cache_dir
            )
            results['blur_heatmap_shape'] = blur_heatmap.shape
        
        # Step 2: Apply processing based on configuration
        if self.config.filter_before_tracking and self.config.enable_blur_filtering:
            # Filter first, then track
            current_stack, quality_stats = self._apply_blur_filtering(
                current_stack, blur_heatmap, output_dir, filename_prefix
            )
            results['blur_filtering'] = {
                'total_cells_before': sum(len(stats) for stats in quality_stats),
                'total_cells_after': sum(stats['passes_threshold'].sum() for stats in quality_stats),
                'output_shape': current_stack.shape
            }
            results['processing_steps'].append('blur_filtering')
            
            # Track filtered cells
            tracked_stack, tracking_stats = self._apply_tracking(
                current_stack, blur_heatmap, output_dir, filename_prefix + "_filtered"
            )
            results['tracking'] = tracking_stats
            results['processing_steps'].append('tracking')
            
        else:
            # Track first, then filter (or just track if filtering disabled)
            tracked_stack, tracking_stats = self._apply_tracking(
                current_stack, blur_heatmap, output_dir, filename_prefix
            )
            results['tracking'] = tracking_stats
            results['processing_steps'].append('tracking')
            
            if self.config.enable_blur_filtering:
                # Apply filtering to tracked results
                tracked_stack, quality_stats = self._apply_blur_filtering(
                    tracked_stack, blur_heatmap, output_dir, filename_prefix + "_tracked"
                )
                results['blur_filtering'] = {
                    'total_cells_before': sum(len(stats) for stats in quality_stats),
                    'total_cells_after': sum(stats['passes_threshold'].sum() for stats in quality_stats),
                    'output_shape': tracked_stack.shape
                }
                results['processing_steps'].append('blur_filtering')
        
        # Save final result
        final_output_path = output_dir / f"{filename_prefix}_final.tif"
        tifffile.imwrite(str(final_output_path), tracked_stack)
        results['final_output'] = str(final_output_path)
        results['final_shape'] = tracked_stack.shape
        
        self.last_results = results
        self.logger.info(f"Processing completed for {segmentation_path.name}")
        
        return results
    
    def _apply_blur_filtering(
        self,
        segmentation_stack: np.ndarray,
        blur_heatmap: np.ndarray,
        output_dir: Path,
        filename_prefix: str
    ) -> tuple[np.ndarray, List[pd.DataFrame]]:
        """Apply blur-based filtering to segmentation stack."""
        self.logger.info("Applying blur-based filtering")
        
        # Handle 3D vs 2D blur heatmap
        if segmentation_stack.ndim == 3:
            if blur_heatmap.ndim == 2:
                blur_heatmaps = [blur_heatmap] * segmentation_stack.shape[0]
            else:
                blur_heatmaps = [blur_heatmap[z] for z in range(blur_heatmap.shape[0])]
            
            filtered_stack, quality_stats = self.blur_filter.filter_3d_stack(
                segmentation_stack, blur_heatmaps
            )
        else:
            filtered_stack, quality_stats = self.blur_filter.filter_cells_by_blur(
                segmentation_stack, blur_heatmap
            )
            quality_stats = [quality_stats]
        
        # Save intermediate results if requested
        if self.config.save_intermediate_results:
            filtered_path = output_dir / f"{filename_prefix}_blur_filtered.tif"
            tifffile.imwrite(str(filtered_path), filtered_stack)
            
            # Save quality statistics
            combined_stats = pd.concat(quality_stats, ignore_index=True)
            stats_path = output_dir / f"{filename_prefix}_blur_stats.csv"
            combined_stats.to_csv(stats_path, index=False)
        
        return filtered_stack, quality_stats
    
    def _apply_tracking(
        self,
        segmentation_stack: np.ndarray,
        blur_heatmap: Optional[np.ndarray],
        output_dir: Path,
        filename_prefix: str
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        """Apply 3D cell tracking to segmentation stack."""
        self.logger.info("Applying 3D cell tracking")
        
        # Prepare extra properties for tracking
        extra_properties = []
        intensity_stack = None
        
        if blur_heatmap is not None:
            extra_properties = [blur_intensity_metric]
            if segmentation_stack.ndim == 3:
                if blur_heatmap.ndim == 2:
                    intensity_stack = np.stack([blur_heatmap] * segmentation_stack.shape[0])
                else:
                    intensity_stack = blur_heatmap
            else:
                intensity_stack = blur_heatmap
        
        # Perform tracking
        tracked_stack = self.tracker.track_cells(
            segmentation_stack,
            intensity_stack=intensity_stack,
            extra_properties=extra_properties
        )
        
        tracking_stats = self.tracker.get_tracking_summary()
        
        # Save intermediate results if requested
        if self.config.save_intermediate_results:
            tracked_path = output_dir / f"{filename_prefix}_tracked.tif"
            tifffile.imwrite(str(tracked_path), tracked_stack)
            
            # Save tracking data
            if self.tracker.last_tracking_data is not None:
                tracking_data_path = output_dir / f"{filename_prefix}_tracking_data.csv"
                self.tracker.last_tracking_data.to_csv(tracking_data_path, index=False)
        
        return tracked_stack, tracking_stats
    
    def process_batch(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        segmentation_pattern: str = "*_seg_3d.tif",
        image_pattern: str = "*_BF_3d.tif",
        blur_cache_dir: Optional[Union[str, Path]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of files through the pipeline.
        
        Args:
            input_dir: Directory containing input files
            output_dir: Directory to save results
            segmentation_pattern: Glob pattern for segmentation files
            image_pattern: Glob pattern for image files
            blur_cache_dir: Directory for caching blur heatmaps
            
        Returns:
            List of processing results for each file
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find matching segmentation and image files
        seg_files = list(input_dir.glob(segmentation_pattern))
        self.logger.info(f"Found {len(seg_files)} segmentation files to process")
        
        all_results = []
        
        for seg_file in tqdm(seg_files, desc="Processing files"):
            # Find corresponding image file
            image_file = self._find_corresponding_image(seg_file, input_dir, image_pattern)
            
            if image_file is None:
                self.logger.warning(f"No corresponding image found for {seg_file.name}")
                continue
            
            try:
                # Process single file
                result = self.process_single_file(
                    seg_file,
                    image_file,
                    output_dir,
                    blur_cache_dir,
                    seg_file.stem
                )
                all_results.append(result)
                
            except Exception as e:
                self.logger.error(f"Failed to process {seg_file.name}: {e}")
                all_results.append({
                    'input_segmentation': str(seg_file),
                    'error': str(e)
                })
        
        # Save batch summary
        self._save_batch_summary(all_results, output_dir)
        
        return all_results
    
    def _find_corresponding_image(
        self,
        seg_file: Path,
        input_dir: Path,
        image_pattern: str = "*_BF_3d.tif",
        segmentation_pattern: str = "*_seg_3d.tif",
    ) -> Optional[Path]:
        """Find the corresponding image file for a segmentation file."""
        # TODO : Implement more robust matching logic
        if True:
            # Remove wildcards from patterns if present
            if "*" in segmentation_pattern:
                segmentation_pattern = segmentation_pattern.replace("*", "")

            # Remove extension from segmentation_pattern if present
            if segmentation_pattern.endswith(".tif"):
                segmentation_pattern = segmentation_pattern[:-4]

        # Try to match based on filename patterns
        image_pattern = seg_file.stem.replace(segmentation_pattern, image_pattern)
        
        # # Look for image with similar name
        # possible_names = [
        #     f"{seg_stem}_BF_3d.tif",
        #     f"{seg_stem}_BF.tif",
        #     f"{seg_stem}.tif"
        # ]
        
        img_path = glob.glob(str(input_dir / image_pattern))
        if len(img_path) == 1:
            img_path = Path(img_path[0])
            if img_path.exists():
                return img_path
                
        return None
    
    def _save_batch_summary(self, results: List[Dict[str, Any]], output_dir: Path):
        """Save summary of batch processing results."""
        import json
        
        summary = {
            'total_files': len(results),
            'successful': len([r for r in results if 'error' not in r]),
            'failed': len([r for r in results if 'error' in r]),
            'config': {
                'tracking': self.config.tracking_config.__dict__,
                'filtering': self.config.filter_config.__dict__ if self.blur_filter else None,
                'pipeline': {
                    'enable_blur_filtering': self.config.enable_blur_filtering,
                    'filter_before_tracking': self.config.filter_before_tracking
                }
            }
        }
        
        summary_path = output_dir / "batch_processing_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Batch processing completed: {summary['successful']}/{summary['total_files']} files successful")


# High-level convenience functions

def process_single_stack(
    segmentation_path: Union[str, Path],
    image_path: Union[str, Path],
    output_dir: Union[str, Path],
    config: Optional[PipelineConfig] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Process a single 3D stack through the complete pipeline.
    
    Args:
        segmentation_path: Path to segmentation file
        image_path: Path to image file
        output_dir: Output directory
        config: Pipeline configuration
        **kwargs: Additional arguments
        
    Returns:
        Processing results
    """
    pipeline = CellTrackingPipeline(config)
    return pipeline.process_single_file(
        segmentation_path, image_path, output_dir, **kwargs
    )


def process_batch_stacks(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    config: Optional[PipelineConfig] = None,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Process a batch of 3D stacks through the complete pipeline.
    
    Args:
        input_dir: Input directory
        output_dir: Output directory
        config: Pipeline configuration
        **kwargs: Additional arguments
        
    Returns:
        List of processing results
    """
    pipeline = CellTrackingPipeline(config)
    return pipeline.process_batch(input_dir, output_dir, **kwargs)
