# Postprocessing Module

This module provides comprehensive postprocessing capabilities for 3D cell tracking and analysis, including blur-based quality filtering.

## Overview

The postprocessing module consists of three main components:

1. **Cell Tracking** (`cell_tracking.py`) - 3D cell tracking across z-stacks using trackpy
2. **Blur Filtering** (`blur_filtering.py`) - Quality assessment and filtering based on image sharpness  
3. **Integrated Pipeline** (`pipeline.py`) - Complete processing pipeline combining all components
4. **Tracking Processor** (`tracking_processor.py`) - Comprehensive 3D tracking with blur filtering

## Features

### 🔬 3D Cell Tracking
- Robust tracking across z-stacks using trackpy
- Configurable search parameters and quality filters
- Support for intensity-based measurements
- Comprehensive tracking statistics

### 🎯 Blur-Based Filtering
- Automatic blur heatmap generation and caching
- Configurable blur thresholds and metrics
- Support for 2D and 3D data
- Quality assessment reporting

###  Integrated Pipeline
- Complete end-to-end processing
- Flexible processing order (filter→track or track→filter)
- Batch processing capabilities
- Comprehensive result reporting

## Quick Start

### Single File Processing

```python
from src.postprocessing.pipeline import process_single_stack

# Process a single 3D segmentation stack
result = process_single_stack(
    segmentation_path="path/to/segmentation_3d.tif",
    image_path="path/to/image_BF_3d.tif", 
    output_dir="path/to/output"
)

print(f"Results saved to: {result['final_output']}")
```

### Batch Processing

```python
from src.postprocessing.pipeline import process_batch_stacks

# Process all files in a directory
results = process_batch_stacks(
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

### Pipeline Configuration

```python
from src.postprocessing.pipeline import PipelineConfig

config = PipelineConfig(
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

## Performance Optimization

### Blur Heatmap Caching
Enable caching to avoid recomputing blur heatmaps:

```python
# Cache blur heatmaps to disk
config = FilterConfig(cache_blur_maps=True)
blur_filter = BlurFilter(config)

# Specify cache directory for batch processing
results = process_batch_stacks(
    input_dir="input",
    output_dir="output", 
    blur_cache_dir="blur_cache"
)
```

### Memory Management
For large datasets, process files individually to manage memory:

```python
import glob

for seg_file in glob.glob("*.tif"):
    img_file = seg_file.replace("_3d.tif", "_BF_3d.tif")
    
    result = process_single_stack(
        seg_file, img_file, "output",
        config=PipelineConfig(save_intermediate_results=False)
    )
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

## Command Line Options

```bash
# Core options
--input-dir DIR              # Input directory
--output-dir DIR             # Output directory  
--segmentation-file FILE     # Single segmentation file
--image-file FILE           # Corresponding image file

# Blur filtering
--blur-threshold FLOAT      # Blur threshold (default: 0.5)
--blur-patch-size INT       # Patch size (default: 32)
--blur-cache-dir DIR        # Cache directory

# Tracking  
--search-range FLOAT        # Search range (default: 5.0)
--memory INT               # Memory frames (default: 1)
--min-track-length INT     # Min track length (default: 3)

# Pipeline options
--filter-before-tracking   # Filter then track (default)
--track-before-filtering   # Track then filter
--disable-blur-filtering   # Skip blur filtering

# Analysis
--pixel-size FLOAT         # Pixel size in μm (default: 1.0)
--time-interval FLOAT      # Time interval in min (default: 1.0)
```

## Testing

Run the test suite to verify functionality:

```bash
python -m pytest tests/postprocessing/ -v
```

The tests cover:
- 3D cell tracking with mock data
- Blur filtering and quality assessment  
- Integrated pipeline processing
- Batch processing workflows

## Best Practices

1. **Use blur filtering** to improve tracking quality
2. **Cache blur heatmaps** for repeated analysis
3. **Validate parameters** on a small subset first
4. **Save intermediate results** for debugging
5. **Monitor memory usage** with large datasets
6. **Check tracking statistics** for quality assessment

## Migration from Legacy Code

If migrating from the old `track_cells.py`:

```python
# Old approach
from src.utils.track_cells import track_3d_centers

# New approach  
from src.postprocessing.pipeline import process_single_stack

# The new pipeline provides the same functionality with:
# - Better error handling
# - Configurable parameters  
# - Comprehensive output
# - Quality assessment
```

## Dependencies

- `numpy` - Numerical operations
- `pandas` - Data manipulation
- `trackpy` - Particle tracking
- `tifffile` - TIFF file I/O
- `scikit-image` - Image processing
- `tqdm` - Progress bars
- `scipy` - Scientific computing

## Troubleshooting

### Common Issues

1. **ImportError**: Ensure all dependencies are installed
2. **Memory errors**: Process files individually or reduce image size
3. **No cells tracked**: Check segmentation quality and tracking parameters
4. **Poor tracking**: Adjust search_range and memory parameters
5. **Blur filtering too aggressive**: Increase blur_threshold or set invert_threshold=True

### Performance Tips

- Use smaller patch sizes for faster blur computation
- Disable intermediate file saving for batch processing
- Cache blur heatmaps when processing multiple files with same images
- Use appropriate tracking parameters for your cell movement patterns
