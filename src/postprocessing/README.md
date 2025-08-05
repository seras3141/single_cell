# Postprocessing Module

This module provides comprehensive postprocessing capabilities for 3D cell tracking and analysis, including blur-based quality filtering.

## Overview

The postprocessing module consists of three main components:

1. **Cell Tracking** (`cell_tracking.py`) - 3D cell tracking across z-stacks using trackpy
2. **Blur Filtering** (`blur_filtering.py`) - Quality assessment and filtering based on image sharpness  
3. **Postprocessing Pipeline** (`tracking_processor.py`) - Complete processing pipeline combining all components

## Quick Start

For a fast start, use the following command line examples:

```bash
# Quick single file run
python scripts/run_postprocessing.py \
    --segmentation-file path/to/segmentation_3d.tif \
    --image-file path/to/image_BF_3d.tif \
    --output-dir path/to/output

# Quick batch run
python scripts/run_postprocessing.py \
    --image-dir path/to/images \
    --seg-dir path/to/segmentations \
    --blur-dir path/to/blurmaps \
    --output-dir path/to/output
 
# Use config file
python scripts/run_postprocessing.py \
    --config config/postprocessing_config.yaml
```

These commands will process your data using default parameters and save results to the specified output directory.

## Features

### 3D Cell Tracking
- Robust tracking across z-stacks using trackpy
- Configurable search parameters and quality filters
- Support for intensity-based measurements
- Comprehensive tracking statistics

### Blur-Based Filtering
- Automatic blur heatmap generation and caching
- Configurable blur thresholds and metrics
- Support for 2D and 3D data
- Quality assessment reporting

### Postprocessing Pipeline
- Complete end-to-end processing
- Flexible processing order (filter→track or track→filter)
- Batch processing capabilities
- Comprehensive result reporting

## Detailed Postprocessing Pipeline

### Single File Processing

```python
from src.postprocessing.tracking_processor import CellTrackingPipeline

# Process a single 3D segmentation stack
pipeline = CellTrackingPipeline()
result = pipeline.process_single_file(
    segmentation_path="path/to/segmentation_3d.tif",
    image_path="path/to/image_BF_3d.tif", 
    output_dir="path/to/output"
)

print(f"Results saved to: {result['final_output']}")
```

### Batch Processing

```python
from src.postprocessing.tracking_processor import CellTrackingPipeline

pipeline = CellTrackingPipeline()
results = pipeline.process_batch(
    input_dir="path/to/input",
    output_dir="path/to/output"
)
print(f"Processed {len(results)} files")
```

### Command Line Interface

```bash
# Process single file
python scripts/run_tracking_pipeline.py \
    --segmentation-file path/to/seg_3d.tif \
    --image-file path/to/img_BF_3d.tif \
    --output-dir path/to/output

# Process batch
python scripts/run_tracking_pipeline.py \
    --input-dir path/to/input \
    --output-dir path/to/output \
    --blur-threshold 0.5 \
    --min-track-length 3
```

## Configuration

### Tracking Configuration

```python
from src.postprocessing.cell_tracking import TrackingConfig

config = TrackingConfig(
    search_range=5.0,        # Max distance between frames
    memory=1,                # Frames to remember lost particles
    min_track_length=3,      # Minimum track length to keep
    min_area=10,             # Minimum cell area
    max_area=5000           # Maximum cell area
)
```

### Blur Filtering Configuration

```python
from src.postprocessing.blur_filtering import FilterConfig

config = FilterConfig(
    patch_size=32,           # Blur measurement patch size
    stride_size=8,           # Stride for blur patches
    blur_threshold=0.5,      # Threshold for filtering
    invert_threshold=False,  # Keep sharp (False) or blurry (True) cells
    cache_blur_maps=True     # Cache blur heatmaps
)
```

### Unified Pipeline Configuration

```python
from src.postprocessing.tracking_processor import PostprocessingConfig

config = PostprocessingConfig(
    enable_blur_filtering=True,      # Enable blur filtering
    filter_before_tracking=True,     # Processing order
    save_intermediate_results=True   # Save intermediate files
)
```

## Advanced Usage

### Custom Tracking with Blur Filtering

```python
from src.postprocessing.cell_tracking import CellTracker3D
from src.postprocessing.blur_filtering import BlurFilter
import tifffile

# Initialize components
tracker = CellTracker3D()
blur_filter = BlurFilter()

# Load data
segmentation = tifffile.imread("segmentation_3d.tif")
image = tifffile.imread("image_3d.tif")

# Get blur heatmap
blur_heatmap = blur_filter.get_or_compute_blur_heatmap("image_3d.tif")

# Filter cells by blur quality
filtered_seg, quality_stats = blur_filter.filter_3d_stack(
    segmentation, [blur_heatmap] * segmentation.shape[0]
)

# Track filtered cells
tracked_result = tracker.track_cells(filtered_seg)

# Get tracking statistics
stats = tracker.get_tracking_summary()
print(f"Tracked {stats['n_particles']} cells")
```

## Output Files

The pipeline generates several output files:

### Main Output
- `*_final.tif` - Final tracked segmentation with particle IDs

### Intermediate Results (if enabled)
- `*_blur_filtered.tif` - Blur-filtered segmentation
- `*_tracked.tif` - Tracked segmentation before final filtering
- `*_blur_stats.csv` - Blur quality statistics
- `*_tracking_data.csv` - Raw tracking data
- `*_population_stats.json` - Population-level statistics

### Batch Processing
- `batch_processing_summary.json` - Summary of batch processing results

## Default Output Folder Structure

The default output directory contains the following files and structure after processing:

```
output/
├── blur_filtered/
├── tracked/
├── tracked_blur_filtered/
├── final/
│   ├── sample1_masks_3d.tif
│   └── sample2_masks_3d.tif
└── batch_processing_summary.json
```

## Quality Assessment

### Blur Quality Metrics
```python
from src.postprocessing.blur_filtering import assess_segmentation_quality

quality = assess_segmentation_quality(
    "segmentation.tif",
    "image.tif"
)

print(f"Blur rate: {quality['blur_rate']:.1%}")
print(f"Average blur intensity: {quality['avg_blur_intensity']:.3f}")
```

### Tracking Quality
```python
# Access tracking statistics
stats = tracker.get_tracking_summary()

print(f"Particles tracked: {stats['n_particles']}")
print(f"Average track length: {stats['avg_track_length']:.1f}")
print(f"Detection efficiency: {stats['n_detections']/stats['n_frames']:.1f} cells/frame")
```

