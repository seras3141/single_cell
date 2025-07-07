"""
Postprocessing module for single cell analysis.

This module provides utilities for postprocessing segmentation results, including:
- 3D cell tracking across z-stacks
- Blur-based filtering and quality assessment
- Output formatting and processing pipelines
"""

from .cell_tracking import (
    CellTracker3D,
    TrackingConfig,
    track_segmentation_masks,
    filter_tracks_by_quality
)

from .blur_filtering import (
    BlurFilter,
    FilterConfig,
    filter_cells_by_blur,
    assess_segmentation_quality
)

from .tracking_processor import (
    TrackingProcessor,
    TrackingProcessorConfig,
    run_tracking_pipeline,
    main_compatible
)

__all__ = [
    # Cell tracking
    "CellTracker3D",
    "TrackingConfig", 
    "track_segmentation_masks",
    "filter_tracks_by_quality",
    # Blur filtering
    "BlurFilter",
    "FilterConfig",
    "filter_cells_by_blur",
    "assess_segmentation_quality",
    # Complete tracking processor
    "TrackingProcessor",
    "TrackingProcessorConfig", 
    "run_tracking_pipeline",
    "main_compatible"
]
