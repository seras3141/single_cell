"""
Blur-based filtering module for cell tracking postprocessing.

This module provides functionality to filter cells based on blur/sharpness
measurements, improving tracking quality by removing low-quality detections.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
import pandas as pd
from skimage.measure import regionprops
from scipy import ndimage as ndi

from joblib import Parallel, delayed

# Import the centralized configuration
from src.utils.config_schemas import FilterConfig

logger = logging.getLogger(__name__)


def blur_intensity_metric(regionmask: np.ndarray, intensity_image: np.ndarray) -> float:
    """
    Compute average blur intensity for a region.
    
    Args:
        regionmask: Boolean mask for the region
        intensity_image: Blur heatmap/intensity image
        
    Returns:
        Average blur intensity in the region
    """
    if not np.any(regionmask):
        return np.nan
    return np.mean(intensity_image[regionmask])


class BlurFilter:
    """
    Blur-based filter for cell segmentation quality assessment.
    
    This class provides functionality to filter cells based on sharpness
    measurements, helping to improve tracking quality by removing blurry
    or low-quality cell detections.
    """
    
    def __init__(self, config: Optional[FilterConfig] = None):
        """
        Initialize the blur filter.
        
        Args:
            config: Filter configuration. If None, uses default config.
        """
        self.config = config or FilterConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Cache for blur heatmaps
        self.blur_cache: Dict[str, np.ndarray] = {}
        
    def get_or_compute_blur_heatmap(
        self, 
        image_path: Union[str, Path],
        blur_cache_dir: Optional[Union[str, Path]] = None
    ) -> np.ndarray:
        """
        Get blur heatmap from cache or compute it.
        
        Args:
            image_path: Path to the image file
            blur_cache_dir: Directory to store/load cached blur maps
            
        Returns:
            Blur heatmap array
        """
        image_path = Path(image_path)
        cache_key = str(image_path)
        
        # Check memory cache first
        if cache_key in self.blur_cache:
            return self.blur_cache[cache_key]
        
        from src.utils.blur_measure import get_or_compute_blur_heatmap
        from src.utils.file_utils import BlurFileHandler

        if blur_cache_dir is not None:
            blur_file_handler = BlurFileHandler()
            blur_suffix = f"{self.config.blur_map_suffix}_{self.config.patch_size}_{self.config.stride_size}"
            blur_file_name = blur_file_handler.rename_image(image_path, blur_suffix)
            blur_path = Path(blur_cache_dir) / blur_file_name

        else:
            blur_path = None

            
        blur_map = get_or_compute_blur_heatmap(
            image_path,
            blur_path=blur_path,
            patch_size=self.config.patch_size,
            stride_size=self.config.stride_size,
            normalize=self.config.normalize_blur
        )

        print(f"Blur heatmap shape: {blur_map.shape}") # Debugging line to check shape
        
        # Check disk cache
        # blur_map = None
        # if blur_cache_dir is not None:
        #     blur_cache_dir = Path(blur_cache_dir)
        #     blur_filename = (image_path.stem + 
        #                    f"{self.config.blur_map_suffix}_{self.config.patch_size}_{self.config.stride_size}.tif")
        #     blur_path = blur_cache_dir / blur_filename
            
        #     if blur_path.exists():
        #         try:
        #             blur_map = tifffile.imread(str(blur_path))
        #             self.logger.debug(f"Loaded blur heatmap from cache: {blur_path}")
        #         except Exception as e:
        #             self.logger.warning(f"Failed to load cached blur map {blur_path}: {e}")
        
        # # Compute if not cached
        # if blur_map is None:
        #     self.logger.info(f"Computing blur heatmap for {image_path}")
        #     blur_map = measure_blur_heatmap(
        #         str(image_path),
        #         patch_size=self.config.patch_size,
        #         stride_size=self.config.stride_size,
        #         normalize=self.config.normalize_blur
        #     )
            
        #     # Save to disk cache if directory provided
        #     if blur_cache_dir is not None:
        #         raise (NotImplementedError("File renaming is not implemented yet."))
        #         try:
        #             blur_cache_dir.mkdir(parents=True, exist_ok=True)
        #             blur_filename = (image_path.stem + 
        #                            f"{self.config.blur_map_suffix}_{self.config.patch_size}_{self.config.stride_size}.tif")
        #             blur_path = blur_cache_dir / blur_filename
        #             tifffile.imwrite(str(blur_path), blur_map.astype(np.float32))
        #             self.logger.debug(f"Saved blur heatmap to cache: {blur_path}")
        #         except Exception as e:
        #             self.logger.warning(f"Failed to save blur map to cache: {e}")
        
        # Store in memory cache
        if self.config.cache_blur_maps:
            self.blur_cache[cache_key] = blur_map
            
        return blur_map
    
    def filter_cells_by_blur(
        self,
        segmentation_mask: np.ndarray,
        blur_heatmap: np.ndarray,
        blur_threshold: Optional[float] = None,
        invert_threshold: Optional[bool] = None
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Filter cells in a segmentation mask based on blur measurements.
        
        Args:
            segmentation_mask: 2D instance segmentation mask
            blur_heatmap: 2D blur heatmap
            blur_threshold: Blur threshold (uses config if None)
            invert_threshold: Whether to invert threshold (uses config if None)
            
        Returns:
            Tuple of (filtered_mask, quality_stats)
        """
        blur_threshold = blur_threshold or self.config.blur_threshold
        invert_threshold = invert_threshold if invert_threshold is not None else self.config.invert_threshold
        
        # Get region properties with blur intensity
        regions = regionprops(
            segmentation_mask,
            intensity_image=blur_heatmap,
            extra_properties=[blur_intensity_metric]
        )

        if len(regions) == 0:
            self.logger.warning("No regions found in segmentation mask.")
            filtered_mask = np.zeros_like(segmentation_mask)
            quality_df = pd.DataFrame(columns=[
                'label', 'area', 'centroid_x', 'centroid_y',
                'blur_intensity', 'passes_threshold'
            ])
            return filtered_mask, quality_df

        # print(np.unique(segmentation_mask))  # Debugging line to check unique labels
        # print(f"Number of regions found: {len(regions)}")  # Debugging line

        # Filter regions based on blur threshold
        def blur_passes_threshold(blur_value: float) -> bool:
            if np.isnan(blur_value):
                return False
            if invert_threshold:
                return blur_value > blur_threshold
            else:
                return blur_value < blur_threshold
        
        # Create filtered mask
        filtered_mask = np.zeros_like(segmentation_mask)
        quality_stats = []

        
        for region in regions:
            blur_value = region.blur_intensity_metric
            passes_threshold = blur_passes_threshold(blur_value)
            
            quality_stats.append({
                'label': region.label,
                'area': region.area,
                'centroid_x': region.centroid[1],
                'centroid_y': region.centroid[0],
                'blur_intensity': blur_value,
                'passes_threshold': passes_threshold
            })
            
            if passes_threshold:
                filtered_mask[segmentation_mask == region.label] = region.label
        
        quality_df = pd.DataFrame(quality_stats)

        # print(quality_df.head())  # Debugging line to check quality stats
        
        n_original = len(regions)
        n_filtered = quality_df['passes_threshold'].sum()
        self.logger.info(f"Blur filtering: {n_filtered}/{n_original} cells passed threshold {blur_threshold}")
        
        return filtered_mask, quality_df

    def filter_cells_by_blur_fast(self, segmentation_mask, blur_heatmap,
                                blur_threshold=None, invert_threshold=None):

        blur_threshold = blur_threshold or self.config.blur_threshold
        invert_threshold = invert_threshold if invert_threshold is not None else self.config.invert_threshold
        
        labels = np.unique(segmentation_mask)
        labels = labels[labels != 0]

        if len(labels) == 0:
            self.logger.warning("No regions found in segmentation mask.")
            filtered_mask = np.zeros_like(segmentation_mask)
            quality_df = pd.DataFrame(columns=[
                'label', 'area', 'centroid_x', 'centroid_y',
                'blur_intensity', 'passes_threshold'
            ])
            return filtered_mask, quality_df

        # Compute stats vectorized
        blur_means = ndi.mean(blur_heatmap, labels=segmentation_mask, index=labels)
        areas = ndi.sum(np.ones_like(segmentation_mask), labels=segmentation_mask, index=labels)
        centroids = np.array(ndi.center_of_mass(np.ones_like(segmentation_mask), labels=segmentation_mask, index=labels))

        # Apply threshold vectorized
        if invert_threshold:
            passes = blur_means > blur_threshold
        else:
            passes = blur_means < blur_threshold

        # Build stats table
        quality_df = pd.DataFrame({
            'label': labels,
            'area': areas,
            'centroid_x': centroids[:, 1],
            'centroid_y': centroids[:, 0],
            'blur_intensity': blur_means,
            'passes_threshold': passes
        })

        # Vectorized filtering
        keep_labels = labels[passes]
        filtered_mask = np.where(np.isin(segmentation_mask, keep_labels), segmentation_mask, 0)

        self.logger.info(f"Blur filtering: {passes.sum()}/{len(labels)} cells passed threshold {blur_threshold}")

        return filtered_mask, quality_df

    def filter_3d_stack(
        self,
        segmentation_stack: np.ndarray,
        blur_heatmaps: Union[np.ndarray, List[np.ndarray]],
        **kwargs
    ) -> Tuple[np.ndarray, List[pd.DataFrame]]:
        """
        Filter a 3D segmentation stack based on blur measurements.
        
        Args:
            segmentation_stack: 3D segmentation array (z, y, x)
            blur_heatmaps: 3D blur array or list of 2D arrays
            **kwargs: Additional arguments for filter_cells_by_blur
            
        Returns:
            Tuple of (filtered_stack, list_of_quality_stats)
        """
        if isinstance(blur_heatmaps, np.ndarray) and blur_heatmaps.ndim == 3:
            blur_list = [blur_heatmaps[z] for z in range(blur_heatmaps.shape[0])]
        else:
            blur_list = blur_heatmaps
        
        if len(blur_list) != segmentation_stack.shape[0]:
            raise ValueError(f"Number of blur heatmaps ({len(blur_list)}) must match "
                           f"number of z-slices ({segmentation_stack.shape[0]})")
        
        filtered_stack = np.zeros_like(segmentation_stack)
        all_quality_stats = []

        print(f"Filtering 3D stack with {segmentation_stack.shape[0]} slices")  # Debugging line
        print(f"Blur list length: {len(blur_list)}")  # Debugging line
        
        for z in range(segmentation_stack.shape[0]):
            filtered_slice, quality_stats = self.filter_cells_by_blur(
                segmentation_stack[z],
                blur_list[z],
                **kwargs
            )
            
            filtered_stack[z] = filtered_slice
            quality_stats['z'] = z
            all_quality_stats.append(quality_stats)
        
        return filtered_stack, all_quality_stats

    def filter_3d_stack_fast(
        self,
        segmentation_stack: np.ndarray,
        blur_heatmaps: Union[np.ndarray, List[np.ndarray]],
        n_jobs: int | None = None,
        **kwargs
    ) -> Tuple[np.ndarray, List[pd.DataFrame]]:
        """
        Filter a 3D segmentation stack based on blur measurements (parallelized and optimized).

        Args:
            segmentation_stack: 3D segmentation array (z, y, x)
            blur_heatmaps: 3D blur array or list of 2D arrays
            n_jobs: Number of parallel jobs to use (-1 = all cores)
            **kwargs: Additional arguments for filter_cells_by_blur_fast

        Returns:
            Tuple of (filtered_stack, list_of_quality_stats)
        """
        # Ensure blur_heatmaps matches the stack shape
        if isinstance(blur_heatmaps, np.ndarray) and blur_heatmaps.ndim == 3:
            blur_list = [blur_heatmaps[z] for z in range(blur_heatmaps.shape[0])]
        else:
            blur_list = blur_heatmaps

        if len(blur_list) != segmentation_stack.shape[0]:
            raise ValueError(
                f"Number of blur heatmaps ({len(blur_list)}) must match "
                f"number of z-slices ({segmentation_stack.shape[0]})"
            )

        n_slices = segmentation_stack.shape[0]
        # Patched up code for slurm parallel execution 
        if n_jobs is None:
            n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
        self.logger.info(f"Filtering 3D stack with {n_slices} slices using {n_jobs} parallel jobs.")

        # Parallel processing across z-slices
        results = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(self.filter_cells_by_blur_fast)(
                segmentation_stack[z],
                blur_list[z],
                **kwargs
            ) for z in range(n_slices)
        )

        # Reconstruct filtered stack and quality stats
        filtered_slices, quality_stats_list = zip(*results)
        filtered_stack = np.stack(filtered_slices, axis=0)

        # Add z-index to each DataFrame
        for z, df in enumerate(quality_stats_list):
            df['z'] = z

        return filtered_stack, list(quality_stats_list)

'''
def filter_cells_by_blur(
    segmentation_path: Union[str, Path],
    image_path: Union[str, Path],
    output_path: Union[str, Path],
    config: Optional[FilterConfig] = None,
    blur_cache_dir: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    High-level function to filter cells by blur quality.
    
    Args:
        segmentation_path: Path to segmentation mask file
        image_path: Path to corresponding image file
        output_path: Path to save filtered results
        config: Filter configuration
        blur_cache_dir: Directory for caching blur heatmaps
        
    Returns:
        Dictionary with filtering results and statistics
    """
    config = config or FilterConfig()
    blur_filter = BlurFilter(config)
    
    # Load segmentation
    logger.info(f"Loading segmentation from {segmentation_path}")
    segmentation_stack = tifffile.imread(str(segmentation_path)).astype(int)
    
    # Get or compute blur heatmap
    blur_heatmap = blur_filter.get_or_compute_blur_heatmap(
        image_path, blur_cache_dir
    )
    
    # Handle 3D vs 2D
    if segmentation_stack.ndim == 3:
        if blur_heatmap.ndim == 2:
            # Replicate 2D blur map for all z-slices
            blur_heatmaps = [blur_heatmap] * segmentation_stack.shape[0]
        else:
            blur_heatmaps = blur_heatmap
        
        filtered_stack, quality_stats = blur_filter.filter_3d_stack(
            segmentation_stack, blur_heatmaps
        )
        
        # Combine quality stats
        combined_stats = pd.concat(quality_stats, ignore_index=True)
        
    else:
        # 2D case
        filtered_stack, combined_stats = blur_filter.filter_cells_by_blur(
            segmentation_stack, blur_heatmap
        )
    
    # Save results
    logger.info(f"Saving filtered results to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tifffile.imwrite(str(output_path), filtered_stack)
    
    # Save quality stats
    stats_path = Path(output_path).with_suffix('.csv')
    combined_stats.to_csv(stats_path, index=False)
    
    # Compute summary statistics
    total_cells = len(combined_stats)
    passed_cells = combined_stats['passes_threshold'].sum()
    avg_blur = combined_stats['blur_intensity'].mean()
    
    results = {
        'segmentation_path': str(segmentation_path),
        'image_path': str(image_path),
        'output_path': str(output_path),
        'stats_path': str(stats_path),
        'input_shape': segmentation_stack.shape,
        'output_shape': filtered_stack.shape,
        'total_cells': total_cells,
        'passed_cells': passed_cells,
        'filter_rate': passed_cells / total_cells if total_cells > 0 else 0,
        'avg_blur_intensity': avg_blur,
        'config': config
    }
    
    logger.info(f"Blur filtering completed: {passed_cells}/{total_cells} cells passed "
               f"(rate: {results['filter_rate']:.2%})")
    
    return results
'''

'''
def assess_segmentation_quality(
    segmentation_path: Union[str, Path],
    image_path: Union[str, Path],
    config: Optional[FilterConfig] = None
) -> Dict[str, Any]:
    """
    Assess the quality of a segmentation based on blur measurements.
    
    Args:
        segmentation_path: Path to segmentation file
        image_path: Path to corresponding image
        config: Filter configuration
        
    Returns:
        Quality assessment results
    """
    config = config or FilterConfig()
    blur_filter = BlurFilter(config)
    
    # Load data
    segmentation = tifffile.imread(str(segmentation_path))
    blur_heatmap = blur_filter.get_or_compute_blur_heatmap(image_path)
    
    # Assess quality
    if segmentation.ndim == 3:
        _, quality_stats = blur_filter.filter_3d_stack(
            segmentation, [blur_heatmap] * segmentation.shape[0]
        )
        combined_stats = pd.concat(quality_stats, ignore_index=True)
    else:
        _, combined_stats = blur_filter.filter_cells_by_blur(segmentation, blur_heatmap)
    
    # Compute quality metrics
    total_cells = len(combined_stats)
    sharp_cells = combined_stats['passes_threshold'].sum()
    
    quality_metrics = {
        'total_cells': total_cells,
        'sharp_cells': sharp_cells,
        'blur_rate': 1 - (sharp_cells / total_cells) if total_cells > 0 else 0,
        'avg_blur_intensity': combined_stats['blur_intensity'].mean(),
        'median_blur_intensity': combined_stats['blur_intensity'].median(),
        'blur_std': combined_stats['blur_intensity'].std(),
        'blur_percentiles': {
            'p10': combined_stats['blur_intensity'].quantile(0.1),
            'p25': combined_stats['blur_intensity'].quantile(0.25),
            'p75': combined_stats['blur_intensity'].quantile(0.75),
            'p90': combined_stats['blur_intensity'].quantile(0.9)
        }
    }
    
    return quality_metrics
'''