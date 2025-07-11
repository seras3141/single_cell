# Visualization Module - Advanced Documentation

This document provides detailed technical documentation for the visualization module, including API references, advanced usage patterns, development guidelines, and troubleshooting.

## Detailed API Documentation

### 3D Visualization (`view_3d_tiff.py`)

#### Key Functions

```python
def view_3d_data(input_path, viewer=None, **kwargs):
    """
    View 3D TIFF data in Napari
    
    Args:
        input_path: Path to 3D TIFF file or directory
        viewer: Existing Napari viewer (optional)
        **kwargs: Additional visualization parameters
    
    Returns:
        Napari viewer instance
    """

def load_3d_stack(filepath):
    """Load 3D TIFF stack with metadata"""

def apply_3d_rendering(viewer, layer, **kwargs):
    """Apply 3D rendering settings"""
```

#### Visualization Options

- **Colormap**: Various color schemes (viridis, plasma, etc.)
- **Opacity**: Layer transparency control
- **Contrast**: Brightness and contrast adjustment
- **Rendering**: Volume rendering vs. maximum intensity projection
- **Scale**: Anisotropic scaling for proper aspect ratios

### 4D Visualization (`view_4d_tiff.py`)

#### Key Functions

```python
def view_4d_data(input_path, time_axis=0, **kwargs):
    """
    View 4D TIFF data with time series
    
    Args:
        input_path: Path to 4D TIFF file
        time_axis: Time axis dimension (0, 1, 2, or 3)
        **kwargs: Additional visualization parameters
    """

def setup_time_controls(viewer, n_timepoints):
    """Setup time series navigation controls"""

def animate_timeseries(viewer, fps=1):
    """Animate time series playback"""
```

#### Time Series Features

- **Navigation**: Frame-by-frame navigation
- **Animation**: Automatic playback with adjustable speed
- **Synchronization**: Multi-channel time series synchronization
- **Export**: Export animations as videos or image sequences

### Prediction Visualization (`visualize_prediction.py`)

#### Key Functions

```python
def visualize_segmentation(image_path, mask_path, output_path=None, **kwargs):
    """
    Visualize segmentation overlays
    
    Args:
        image_path: Path to original image
        mask_path: Path to segmentation mask
        output_path: Path to save overlay (optional)
        **kwargs: Visualization parameters
    """

def create_overlay(image, mask, alpha=0.5, colormap="tab10"):
    """Create segmentation overlay"""

def generate_contours(mask, thickness=1):
    """Generate contour lines from mask"""

def create_enhanced_napari_viewer(brightfield=None, ground_truth=None, 
                                 blur_heatmap=None, inference_prediction=None,
                                 final_segmentation=None, layer_config=None):
    """
    Create enhanced napari viewer with multiple layers
    
    Args:
        brightfield: Path to brightfield image
        ground_truth: Path to ground truth segmentation
        blur_heatmap: Path to blur heatmap
        inference_prediction: Path to inference prediction
        final_segmentation: Path to final segmentation
        layer_config: Dictionary with layer-specific configurations
    
    Returns:
        Configured napari viewer
    """
```

#### Enhanced Visualization Features

- **Better Colormaps**: 
  - I Blue colormap for blur heatmaps (better visibility)
  - Additive blending for heat maps
- **Enhanced 2D Settings**:
  - 2D display mode with z-slider navigation
  - Automatic middle z-slice starting position
  - Better axis labeling (Z, Y, X)
- **Modular Design**: 
  - Reusable `create_enhanced_napari_viewer()` function
  - Support for programmatic layer configuration

#### Overlay Options

- **Alpha Blending**: Adjustable transparency
- **Color Schemes**: Various colormaps for different labels
- **Contour Mode**: Solid masks vs. contour lines
- **Label Display**: Show/hide cell labels
- **Comparison Views**: Side-by-side or overlay modes

### Interactive GUI (`segmentation_gui.py`)

#### Key Components

```python
class SegmentationVisualizationGUI(QMainWindow):
    def __init__(self, data_dir=None):
        """Initialize GUI with optional data directory"""
    
    def setup_ui(self):
        """Setup user interface components"""
    
    def change_data_directory(self):
        """Change the data directory"""
    
    def update_file_list(self):
        """Update the list of available files"""
    
    def update_file_info(self):
        """Update file information panel"""
    
    def visualize_selected(self):
        """Launch napari with selected layers"""

class FileSearcher:
    def __init__(self, data_dir):
        """Initialize file searcher for given directory"""
    
    def get_available_base_names(self):
        """Get list of available base image names"""
    
    def find_related_files(self, base_name):
        """Find all related files for a base name"""
```

#### GUI Features

- **Parameter Adjustment**: Real-time parameter tuning
- **Preview**: Live preview of segmentation results
- **Batch Processing**: Process multiple files
- **Export Options**: Various output formats
- **History**: Parameter history and presets
- **File Browser**: Lists all available segmentation images
- **Multi-layer Support**: Simultaneously display multiple data types
- **Interactive Controls**: Toggle visibility and adjust opacity
- **2D Navigation**: Z-slider for easy slice navigation

#### GUI Components

1. **Data Directory Selection**: 
   - Change the data directory if needed
   - GUI will automatically scan for available images

2. **File List**: 
   - Shows all available base images (e.g., "p2126_J03")
   - Tooltip shows how many files are available for each image

3. **Visualization Options**:
   - Checkboxes to enable/disable each layer type
   - Opacity sliders for fine-tuning transparency

4. **File Information Panel**:
   - Shows which files are available for the selected image
   - Displays file status and paths

5. **Visualize Button**:
   - Launches napari with all selected layers
   - Only enabled when at least a brightfield image is available

### Qt Configuration (`qt_config.py`)

#### Dependency Management

```python
def check_qt_dependencies():
    """Check available Qt backends"""

def configure_qt_backend(backend="auto"):
    """Configure Qt backend"""

def resolve_qt_conflicts():
    """Resolve Qt backend conflicts"""

def safe_napari_import():
    """Safely import Napari with proper Qt configuration"""
```

## Advanced Usage Patterns

### Programmatic Visualization

```python
from src.visualize import create_enhanced_napari_viewer

# Create visualization with custom settings
viewer = create_enhanced_napari_viewer(
    brightfield="path/to/bf_image.tif",
    ground_truth="path/to/gt_segmentation.tif",
    inference_prediction="path/to/prediction.tif",
    layer_config={
        'ground_truth': {'opacity': 0.8},
        'blur_heatmap': {'colormap': 'viridis', 'opacity': 0.4}
    }
)

napari.run()
```

### Batch Processing

```python
from src.visualize import SegmentationVisualizationGUI, FileSearcher

# Find all available images
searcher = FileSearcher("path/to/data")
base_names = searcher.get_available_base_names()

for base_name in base_names:
    files = searcher.find_related_files(base_name)
    if files['brightfield']:
        print(f"Can visualize: {base_name}")
```

### Custom Layer Configuration

```python
# Advanced layer configuration
layer_config = {
    'brightfield': {
        'colormap': 'gray',
        'opacity': 1.0,
        'contrast_limits': [0, 65535]
    },
    'ground_truth': {
        'colormap': 'tab10',
        'opacity': 0.7,
        'blending': 'translucent'
    },
    'blur_heatmap': {
        'colormap': 'I Blue',
        'opacity': 0.5,
        'blending': 'additive'
    },
    'inference_prediction': {
        'colormap': 'tab20',
        'opacity': 0.6,
        'blending': 'translucent'
    }
}

viewer = create_enhanced_napari_viewer(
    brightfield="image.tif",
    ground_truth="gt.tif",
    blur_heatmap="blur.tif",
    inference_prediction="pred.tif",
    layer_config=layer_config
)
```

## Best Practices

### Performance Optimization

1. **Memory Management**: Use memory mapping for large files
2. **Chunked Loading**: Load data in chunks for large datasets
3. **GPU Acceleration**: Use GPU-accelerated rendering when available
4. **Caching**: Cache frequently accessed visualizations
5. **Layer Management**: Hide unused layers to improve performance
6. **2D Mode**: Use 2D mode with z-slider for better performance than full 3D

### User Experience

1. **Responsive UI**: Keep UI responsive during processing
2. **Progress Indicators**: Show progress for long operations
3. **Error Handling**: Graceful error handling with user feedback
4. **Keyboard Shortcuts**: Implement common keyboard shortcuts
5. **Threading**: Napari runs in main thread using QTimer to avoid threading issues

### Visual Quality

1. **Color Schemes**: Use perceptually uniform colormaps
2. **Contrast**: Optimize contrast for different data types
3. **Scaling**: Use appropriate scaling for different magnifications
4. **Annotations**: Add clear labels and legends
5. **Colormap Selection**: Use "I Blue" for better heat map visibility

### Suggested Visualization Improvements

1. **Colormap Enhancements**:
   - Use "I Blue" for better heat map visibility
   - Consider "viridis" or "plasma" for scientific accuracy

2. **Layer Management**:
   - Additive blending for overlay layers
   - Better default opacity values
   - Smart auto-contrast

3. **2D Navigation**:
   - Z-slider for easy slice navigation
   - Middle slice as starting position
   - Better performance than 3D mode

## Development Guidelines

### Adding New Features

To extend the GUI:

1. **New Layer Types**: Add to `FileSearcher.find_related_files()`
2. **Visualization Options**: Extend the configuration panel
3. **File Formats**: Update loading functions to support new formats

### Code Structure

- `segmentation_gui.py`: Main GUI application
- `visualize_prediction.py`: Enhanced napari visualization functions
- `launch_gui_safe.py`: Safe launcher script with dependency checking

### Testing New Features

```python
# Test new visualization features
from src.visualize import SegmentationVisualizationGUI
import sys
from PyQt5.QtWidgets import QApplication

# Create test application
app = QApplication(sys.argv)
gui = SegmentationVisualizationGUI("path/to/test/data")
gui.show()

# Test programmatically
gui.file_list.setCurrentRow(0)
gui.visualize_selected()

app.exec_()
```

## Comprehensive Troubleshooting

### Common Issues and Solutions

1. **PyQt5 not found**: 
   ```bash
   pip install PyQt5
   ```

2. **Napari not working**:
   ```bash
   pip install napari[all]
   ```

3. **Threading errors**:
   - Fixed in current version
   - Napari now runs in main thread using QTimer

4. **Qt backend conflicts**:
   ```bash
   # Set environment variable
   export QT_API=pyqt5
   python scripts/launch_gui_safe.py
   ```

5. **Memory Issues**: 
   - 3D visualization can be memory-intensive
   - Use 2D mode with z-slider navigation
   - Reduce file sizes or use chunked loading

6. **Performance Issues**: 
   - Hide unused layers to improve performance
   - Use 2D mode instead of 3D
   - Optimize layer opacity and blending modes

### Qt Backend Problems

```python
# Check Qt availability
from src.visualize.qt_config import check_qt_dependencies
dependencies = check_qt_dependencies()
print(dependencies)

# Configure backend manually
from src.visualize.qt_config import configure_qt_backend
configure_qt_backend("pyqt5")
```

### Performance Tuning

```python
# Optimize for large datasets
viewer = napari.Viewer()
viewer.layers.selection.active.multiscale = True
viewer.layers.selection.active.cache = True

# Use 2D mode for better performance
viewer.dims.ndisplay = 2
```

### Napari Visualization Details

The enhanced visualization includes:

- **Brightfield**: Gray colormap, base layer
- **Ground Truth**: Label layer with default coloring
- **Blur Heatmap**: I Blue colormap, additive blending
- **Inference Prediction**: Label layer with transparency
- **Final Segmentation**: Label layer showing final results

**2D Mode with Z-Slider**: All data is displayed in 2D mode with a slider at the bottom to navigate through z-slices. This provides better performance and easier navigation compared to 3D mode.

### Debug Mode

```python
# Enable debug mode for troubleshooting
import logging
logging.basicConfig(level=logging.DEBUG)

# Launch GUI with debug information
python scripts/launch_gui_safe.py --debug
```

### Performance Monitoring

```python
# Monitor memory usage
import psutil
import os

def monitor_memory():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")

# Call during visualization
monitor_memory()
```

## Testing and Validation

### Test Suite

```bash
# Run all visualization tests
python -m pytest tests/visualize/ -v

# Test GUI functionality
python -m pytest tests/visualize/test_gui.py -v

# Test Qt configuration
python -m pytest tests/visualize/test_qt_config.py -v

# Test visualization functions
python -m pytest tests/visualize/test_visualize_prediction.py -v
```

### Manual Testing Checklist

1. **GUI Launch**: Test safe GUI launch with dependency checking
2. **File Loading**: Test loading different file types and sizes
3. **Layer Management**: Test visibility toggles and opacity controls
4. **Navigation**: Test z-slider navigation in 2D mode
5. **Export**: Test export functionality for different formats
6. **Error Handling**: Test error handling for missing files
7. **Performance**: Test with large datasets and multiple layers

### Integration Testing

```python
# Test integration with pipeline components
from src.inference.inference_pipeline import InferencePipeline
from src.visualize.visualize_prediction import visualize_segmentation

# Run inference and visualize results
pipeline = InferencePipeline(predictor, output_manager)
results = pipeline.run_inference("data/test")
visualize_segmentation(results)
```

---

*This advanced documentation covers the technical details for developers and advanced users of the visualization module.*
