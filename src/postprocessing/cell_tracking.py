"""
3D Cell tracking module for postprocessing segmentation results.

This module provides robust 3D cell tracking capabilities across z-stacks using trackpy,
with configurable parameters and quality assessment.
"""

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
import tifffile
import trackpy as tp
from skimage.measure import regionprops
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class TrackingConfig:
    """Configuration for 3D cell tracking."""
    
    # Tracking parameters
    search_range: float = 5.0
    """Maximum distance features can move between frames"""
    
    memory: int = 1
    """Number of frames to remember a particle"""
    
    min_track_length: int = 3
    """Minimum length of tracks to keep"""
    
    # Region filtering parameters
    min_area: int = 10
    """Minimum cell area to consider"""
    
    max_area: int = 5000
    """Maximum cell area to consider"""
    
    # Quality assessment
    area_percentiles: Tuple[float, float] = (0.1, 99.9)
    """Percentiles for area-based filtering"""
    
    # Output options
    save_intermediate: bool = False
    """Whether to save intermediate tracking data"""
    
    output_dtype: str = "int32"
    """Data type for output arrays"""


class CellTracker3D:
    """
    3D cell tracker for segmentation masks across z-stacks.
    
    This class provides robust cell tracking functionality with configurable
    parameters and quality assessment capabilities.
    """
    
    def __init__(self, config: Optional[TrackingConfig] = None):
        """
        Initialize the 3D cell tracker.
        
        Args:
            config: Tracking configuration. If None, uses default config.
        """
        self.config = config or TrackingConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Tracking state
        self.last_tracking_data: Optional[pd.DataFrame] = None
        self.tracking_stats: Dict[str, Any] = {}
    
    def extract_cell_properties(
        self, 
        segmentation_mask: np.ndarray,
        intensity_image: Optional[np.ndarray] = None,
        extra_properties: Optional[List[callable]] = None
    ) -> pd.DataFrame:
        """
        Extract cell properties from a single segmentation mask.
        
        Args:
            segmentation_mask: 2D instance segmentation mask
            intensity_image: Optional intensity image for additional properties
            extra_properties: Additional properties to compute
            
        Returns:
            DataFrame with cell properties (x, y, label, area, etc.)
        """
        # Get region properties
        props_kwargs = {}
        if intensity_image is not None:
            props_kwargs['intensity_image'] = intensity_image
        if extra_properties is not None:
            props_kwargs['extra_properties'] = extra_properties
            
        regions = regionprops(segmentation_mask, **props_kwargs)
        
        # Filter by area
        regions = [
            region for region in regions 
            if self.config.min_area <= region.area <= self.config.max_area
        ]
        
        if not regions:
            return pd.DataFrame(columns=['x', 'y', 'label', 'area'])
        
        # Extract basic properties
        properties = {
            'x': [region.centroid[1] for region in regions],  # Note: x/y convention
            'y': [region.centroid[0] for region in regions],
            'label': [region.label for region in regions],
            'area': [region.area for region in regions]
        }
        
        # Add extra properties if available
        if extra_properties:
            for prop_func in extra_properties:
                prop_name = prop_func.__name__
                try:
                    properties[prop_name] = [
                        getattr(region, prop_name) for region in regions
                    ]
                except AttributeError:
                    self.logger.warning(f"Property {prop_name} not found in regions")
        
        return pd.DataFrame(properties)
    
    def extract_3d_centers(
        self, 
        segmentation_stack: np.ndarray,
        intensity_stack: Optional[np.ndarray] = None,
        extra_properties: Optional[List[callable]] = None
    ) -> List[Tuple[int, pd.DataFrame]]:
        """
        Extract cell centers from a 3D segmentation stack.
        
        Args:
            segmentation_stack: 3D array (z, y, x)
            intensity_stack: Optional 3D intensity array
            extra_properties: Additional properties to compute
            
        Returns:
            List of (z_index, properties_dataframe) tuples
        """
        centers_with_z = []
        
        for z in range(segmentation_stack.shape[0]):
            segmentation_slice = segmentation_stack[z]
            intensity_slice = intensity_stack[z] if intensity_stack is not None else None
            
            properties = self.extract_cell_properties(
                segmentation_slice,
                intensity_slice,
                extra_properties
            )
            
            centers_with_z.append((z, properties))
        
        return centers_with_z
    
    def track_cells(self, segmentation_stack: np.ndarray, **kwargs) -> np.ndarray:
        """
        Track cells across a 3D segmentation stack.
        
        Args:
            segmentation_stack: 3D array with instance labels (z, y, x)
            **kwargs: Additional arguments passed to extract_3d_centers
            
        Returns:
            3D array with tracked cell IDs
        """
        self.logger.info(f"Tracking cells in stack of shape {segmentation_stack.shape}")
        
        # Extract cell centers from all z-slices
        centers_with_z = self.extract_3d_centers(segmentation_stack, **kwargs)
        
        # Combine all centers into a single DataFrame for tracking
        all_data = []
        for z, properties in centers_with_z:
            if not properties.empty:
                properties_with_frame = properties.copy()
                properties_with_frame['frame'] = z
                all_data.append(properties_with_frame)
        
        if not all_data:
            self.logger.warning("No cells found for tracking")
            return np.zeros_like(segmentation_stack, dtype=getattr(np, self.config.output_dtype))
        
        # Combine all data
        tracking_data = pd.concat(all_data, ignore_index=True)
        
        # Perform tracking
        self.logger.info(f"Linking {len(tracking_data)} detections across {segmentation_stack.shape[0]} frames")
        tracked_data = tp.link_df(
            tracking_data, 
            search_range=self.config.search_range,
            memory=self.config.memory
        )
        
        # Filter short tracks
        if self.config.min_track_length > 1:
            tracked_data = tp.filter_stubs(tracked_data, self.config.min_track_length)
        
        # Store tracking data for analysis
        self.last_tracking_data = tracked_data
        
        # Create output array with tracked IDs
        tracked_stack = self._create_tracked_stack(
            segmentation_stack, tracked_data
        )
        
        # Compute tracking statistics
        self._compute_tracking_stats(tracked_data)
        
        return tracked_stack
    
    def _create_tracked_stack(
        self, 
        segmentation_stack: np.ndarray, 
        tracked_data: pd.DataFrame
    ) -> np.ndarray:
        """Create a 3D stack with tracked particle IDs."""
        tracked_stack = np.zeros_like(
            segmentation_stack, 
            dtype=getattr(np, self.config.output_dtype)
        )
        
        # Map original labels to tracked particle IDs
        for _, row in tracked_data.iterrows():
            z = int(row['frame'])
            original_label = int(row['label'])
            particle_id = int(row['particle']) + 1  # Start from 1, not 0
            
            # Replace all pixels with original_label with particle_id
            mask = segmentation_stack[z] == original_label
            tracked_stack[z][mask] = particle_id
        
        return tracked_stack
    
    def _compute_tracking_stats(self, tracked_data: pd.DataFrame) -> None:
        """Compute tracking statistics."""
        n_particles = tracked_data['particle'].nunique()
        n_frames = tracked_data['frame'].nunique()
        avg_track_length = tracked_data.groupby('particle').size().mean()
        
        self.tracking_stats = {
            'n_particles': n_particles,
            'n_frames': n_frames,
            'n_detections': len(tracked_data),
            'avg_track_length': avg_track_length,
            'max_track_length': tracked_data.groupby('particle').size().max(),
            'min_track_length': tracked_data.groupby('particle').size().min()
        }
        
        self.logger.info(f"Tracking completed: {n_particles} particles, "
                        f"avg track length: {avg_track_length:.1f}")
    
    def get_tracking_summary(self) -> Dict[str, Any]:
        """Get summary statistics from the last tracking operation."""
        return self.tracking_stats.copy()
    
    def get_track_data(self) -> Optional[pd.DataFrame]:
        """Get the tracking data from the last operation."""
        return self.last_tracking_data.copy() if self.last_tracking_data is not None else None


def track_segmentation_masks(
    input_path: Union[str, Path],
    output_path: Union[str, Path], 
    config: Optional[TrackingConfig] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    High-level function to track cells in segmentation masks.
    
    Args:
        input_path: Path to 3D segmentation mask file
        output_path: Path to save tracked results
        config: Tracking configuration
        **kwargs: Additional arguments for tracking
        
    Returns:
        Dictionary with tracking results and statistics
    """
    config = config or TrackingConfig()
    tracker = CellTracker3D(config)
    
    # Load segmentation stack
    logger.info(f"Loading segmentation from {input_path}")
    segmentation_stack = tifffile.imread(str(input_path)).astype(int)
    
    # Perform tracking
    tracked_stack = tracker.track_cells(segmentation_stack, **kwargs)
    
    # Save results
    logger.info(f"Saving tracked results to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tifffile.imwrite(str(output_path), tracked_stack)
    
    # Return results
    return {
        'input_path': str(input_path),
        'output_path': str(output_path),
        'input_shape': segmentation_stack.shape,
        'output_shape': tracked_stack.shape,
        'tracking_stats': tracker.get_tracking_summary(),
        'config': config
    }


def filter_tracks_by_quality(
    tracking_data: pd.DataFrame,
    min_length: int = 3,
    min_area: Optional[float] = None,
    max_area: Optional[float] = None
) -> pd.DataFrame:
    """
    Filter tracking data by quality criteria.
    
    Args:
        tracking_data: DataFrame with tracking results
        min_length: Minimum track length to keep
        min_area: Minimum average area (if area column exists)
        max_area: Maximum average area (if area column exists)
        
    Returns:
        Filtered tracking data
    """
    # Filter by track length
    track_lengths = tracking_data.groupby('particle').size()
    valid_particles = track_lengths[track_lengths >= min_length].index
    filtered_data = tracking_data[tracking_data['particle'].isin(valid_particles)]
    
    # Filter by area if specified and available
    if 'area' in filtered_data.columns:
        if min_area is not None or max_area is not None:
            avg_areas = filtered_data.groupby('particle')['area'].mean()
            
            if min_area is not None:
                valid_particles = avg_areas[avg_areas >= min_area].index
                filtered_data = filtered_data[filtered_data['particle'].isin(valid_particles)]
            
            if max_area is not None:
                valid_particles = avg_areas[avg_areas <= max_area].index
                filtered_data = filtered_data[filtered_data['particle'].isin(valid_particles)]
    
    logger.info(f"Filtered tracks: {tracking_data['particle'].nunique()} -> "
               f"{filtered_data['particle'].nunique()} particles")
    
    return filtered_data
