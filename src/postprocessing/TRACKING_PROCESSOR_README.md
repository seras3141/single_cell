# 3D Cell Tracking Processor

The 3D Cell Tracking Processor is a comprehensive postprocessing module for tracking cells across z-stacks with advanced blur-based filtering capabilities. This module replaces and enhances the functionality of the original `track_cells.py` script with a modern, modular, and testable design.

## Features

- **3D Cell Tracking**: Track cells across z-stacks using trackpy with configurable parameters
- **Blur-based Filtering**: Filter cells based on sharpness measurements to improve tracking quality
- **Batch Processing**: Process multiple files efficiently with progress tracking
- **Flexible Configuration**: Highly configurable parameters for different experimental conditions
- **CLI Interface**: Command-line interface matching the original track_cells.py workflow
- **Comprehensive Testing**: Full test coverage with mock data and integration tests
- **Compatibility Mode**: Drop-in replacement for the original track_cells.py main() function

## Architecture

The tracking processor consists of several modular components:

### Core Components

1. **TrackingProcessor**: Main processor class that orchestrates the entire pipeline
2. **TrackingProcessorConfig**: Configuration dataclass for all processing parameters  
3. **CellTracker3D**: Low-level 3D tracking using trackpy
4. **BlurFilter**: Blur-based quality assessment and filtering

### Key Features

- **Modular Design**: Each component can be used independently or as part of the full pipeline
- **Configurable Parameters**: Extensive configuration options for tracking, filtering, and output
- **Error Handling**: Robust error handling with detailed logging and progress reporting
- **Caching**: Automatic caching of blur heatmaps to avoid recomputation
- **Multiple Input Formats**: Support for various file naming conventions and patterns

## Usage

### Basic Usage

```python
from src.postprocessing import TrackingProcessor, TrackingProcessorConfig

# Create processor with default configuration
processor = TrackingProcessor()

# Track cells in a 3D segmentation stack
segmentation_stack = load_segmentation_data()  # Your 3D segmentation array
tracked_result = processor.track_3d_centers(segmentation_stack)
```

### Custom Configuration

```python
from src.postprocessing import (
    TrackingProcessor, 
    TrackingProcessorConfig,
    TrackingConfig,
    FilterConfig
)

# Create custom configurations
tracking_config = TrackingConfig(
    search_range=10.0,      # Larger search range for fast-moving cells
    memory=2,               # Remember particles for 2 frames
    min_track_length=4,     # Require longer tracks
    min_area=20,            # Larger minimum cell area
    max_area=3000           # Smaller maximum cell area
)

filter_config = FilterConfig(
    patch_size=64,          # Larger patches for blur measurement
    stride_size=16,         # Larger stride
    blur_threshold=0.3,     # More stringent blur threshold
    invert_threshold=False
)

config = TrackingProcessorConfig(
    blur_threshold=0.3,
    tracking_config=tracking_config,
    filter_config=filter_config,
    save_tracking_data=True  # Save intermediate data
)

processor = TrackingProcessor(config)
```

### Batch Processing

```python
from src.postprocessing import run_tracking_pipeline

# Process multiple files in batch
results = run_tracking_pipeline(
    mask_directory="path/to/masks",
    image_directory="path/to/images",
    output_directory="path/to/output",
    blur_directory="path/to/blur_cache",
    config=config
)

print(f"Processed {results['successful']}/{results['total_files']} files successfully")
```

### Command Line Interface

```bash
# Basic usage
python scripts/track_3d_cells.py \
    --image-directory "data/images" \
    --mask-directory "data/masks" \
    --output-directory "data/tracked" \
    --blur-directory "data/blur_cache"

# With custom parameters
python scripts/track_3d_cells.py \
    --image-directory "data/images" \
    --mask-directory "data/masks" \
    --output-directory "data/tracked" \
    --blur-directory "data/blur_cache" \
    --blur-threshold 0.3 \
    --search-range 10.0 \
    --memory 2 \
    --min-track-length 4 \
    --overwrite \
    --verbose
```

### Compatibility Mode

For existing workflows using the original track_cells.py, use the compatibility function:

```python
from src.postprocessing import main_compatible

# Drop-in replacement for the original main() function
results = main_compatible(
    image_directory="data/BF+IF Experiments_3D_train_test_dataset/train",
    mask_directory="data/BF+IF Experiments_2D_train_test_dataset/predictions_test/pretrained_cell2d_cyto3_FlowT0.4/3d_view",
    output_directory="data/BF+IF Experiments_2D_train_test_dataset/predictions_test/pretrained_cell2d_cyto3_FlowT0.4/3d_view_tracked_SF",
    blur_directory="data/BF+IF Experiments_3D_train_test_dataset/blur_heatmaps",
    blur_thresh=0.5,
    inv=False
)
```

## Configuration Parameters

### Tracking Parameters

- `search_range`: Maximum distance features can move between frames (default: 5.0)
- `memory`: Number of frames to remember a particle (default: 1)  
- `min_track_length`: Minimum length of tracks to keep (default: 3)
- `min_area`: Minimum cell area to consider (default: 10)
- `max_area`: Maximum cell area to consider (default: 5000)

### Blur Filtering Parameters

- `blur_threshold`: Threshold for blur filtering - lower values are sharper (default: 0.5)
- `invert_threshold`: Invert blur threshold comparison (default: False)
- `patch_size`: Size of patches for blur measurement (default: 32)
- `stride_size`: Stride size for blur measurement (default: 8)

### File Processing Parameters

- `mask_pattern`: Pattern to match mask files (default: "*_3d.tif")
- `blur_heatmap_suffix`: Suffix for blur heatmap files (default: "_blur_heatmap_32_8.tif")
- `create_output_dirs`: Whether to create output directories (default: True)
- `overwrite_existing`: Whether to overwrite existing files (default: False)
- `save_tracking_data`: Whether to save intermediate tracking data (default: False)

## Output

The processor generates:

1. **Tracked 3D TIFF files**: Segmentation masks with particle IDs instead of original labels
2. **Blur heatmaps**: Cached blur measurements for faster subsequent processing
3. **Optional tracking data**: CSV files with detailed tracking information (if enabled)
4. **Processing logs**: Detailed logs with timing and success/failure information

## File Naming Conventions

The processor follows these conventions:

- **Input masks**: `*_3d.tif` (configurable via `mask_pattern`)
- **Input images**: `*_BF_3d.tif` (derived from mask names using `image_suffix_mapping`)
- **Blur heatmaps**: `*_blur_heatmap_32_8.tif` (configurable via `blur_heatmap_suffix`)
- **Output files**: `*_3d_filtered_0.5.tif` (includes blur threshold in filename)

## Performance Considerations

- **Blur Heatmap Caching**: Blur heatmaps are automatically cached to avoid recomputation
- **Memory Usage**: Processing is done one file at a time to manage memory usage
- **Parallel Processing**: Future versions may support parallel processing of multiple files
- **Large Datasets**: Tested with datasets containing hundreds of files

## Error Handling

The processor includes comprehensive error handling:

- **File Missing**: Graceful handling of missing input files
- **Invalid Data**: Robust handling of empty or corrupted segmentation data  
- **Processing Failures**: Detailed error logging with file-specific failure tracking
- **Resource Limits**: Automatic memory management and timeout handling

## Testing

The module includes comprehensive tests:

- **Unit Tests**: Test individual components and functions
- **Integration Tests**: Test the complete pipeline with realistic data
- **Mock Tests**: Fast tests using generated test data
- **CLI Tests**: Verify command-line interface functionality

Run tests with:
```bash
python -m pytest tests/postprocessing/test_tracking_processor.py -v
```

## Migration from track_cells.py

To migrate from the original `track_cells.py`:

1. **Replace imports**:
   ```python
   # Old
   from src.utils.track_cells import track_3d_centers, main
   
   # New
   from src.postprocessing import TrackingProcessor, main_compatible
   ```

2. **Update function calls**:
   ```python
   # Old
   main()
   
   # New (exact same interface)
   main_compatible()
   ```

3. **Use new configuration options** (optional):
   ```python
   # New modular approach
   config = TrackingProcessorConfig(blur_threshold=0.3)
   processor = TrackingProcessor(config)
   ```

## Examples

See `examples/tracking_processor_examples.py` for comprehensive usage examples including:

- Basic 3D cell tracking
- Custom configuration setup
- Blur-based filtering demonstration
- Batch processing workflows
- Error handling scenarios
- Compatibility mode usage

## Dependencies

- numpy: Array processing
- pandas: Data manipulation and tracking
- trackpy: Particle tracking algorithms
- tifffile: TIFF file I/O
- skimage: Image processing and region properties
- tqdm: Progress bars
- pathlib: File path handling

## Future Enhancements

Planned improvements include:

- **Parallel Processing**: Multi-threaded batch processing
- **GPU Acceleration**: CUDA support for large datasets
- **Advanced Filtering**: Additional quality metrics beyond blur
- **Export Formats**: Support for additional output formats (HDF5, CSV, etc.)
- **Interactive Interface**: Jupyter notebook widgets for parameter tuning
