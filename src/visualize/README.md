# Visualization Module

The `src/visualize` module provides comprehensive visualization tools for 3D/4D TIFF stacks, prediction overlays, and interactive GUI interfaces using Napari.

## Overview

This module consists of:

1. **3D Visualization** (`view_3d_tiff.py`) - 3D TIFF stack visualization
2. **4D Visualization** (`view_4d_tiff.py`) - 4D TIFF stack visualization with time series
3. **Prediction Visualization** (`visualize_prediction.py`) - Segmentation overlay visualization
4. **Interactive GUI** (`segmentation_gui.py`) - Interactive segmentation GUI
5. **Qt Configuration** (`qt_config.py`) - Qt backend configuration and dependency management

## Features

### 3D/4D Visualization
- Interactive 3D volume rendering
- Time series visualization for 4D data
- Multi-channel support
- Zoom, pan, and rotation controls
- Layer management and opacity control

### Prediction Overlays
- Segmentation mask overlays
- Contour visualization
- Color-coded cell labels
- Transparency and blending options
- Side-by-side comparison views

### Interactive GUI
- User-friendly interface for segmentation
- Real-time parameter adjustment
- Batch processing capabilities
- Export functionality
- Dependency checking and safe launching
- 2D visualization with z-stack navigation
- Multi-layer support for brightfield, ground truth, blur heatmaps, and predictions

### Qt Backend Management
- Automatic Qt backend detection
- Dependency conflict resolution
- Safe GUI launching with error handling
- Cross-platform compatibility

## Quick Start

### Interactive GUI

```bash
# Launch GUI with dependency checking
python scripts/launch_gui_safe.py

# Launch with specific backend
python scripts/launch_gui_safe.py --backend pyqt5

# Or launch directly
python src/visualize/segmentation_gui.py
```

### 3D Visualization

```python
from src.visualize.view_3d_tiff import view_3d_data
import napari

# View 3D TIFF stack
view_3d_data("data/3d_stack.tif")

# View directory of 3D files
view_3d_data("results/3d_predictions/")

# Advanced 3D visualization
viewer = napari.Viewer()
viewer = view_3d_data(
    "data/3d_stack.tif",
    viewer=viewer,
    colormap="viridis",
    opacity=0.8
)
```

### 4D Visualization

```python
from src.visualize.view_4d_tiff import view_4d_data

# View 4D TIFF stack with time series
view_4d_data("data/4d_timeseries.tif")

# View with custom settings
view_4d_data(
    "data/4d_timeseries.tif",
    time_axis=0,
    colormap="plasma",
    fps=2
)
```

### Prediction Visualization

```python
from src.visualize.visualize_prediction import visualize_segmentation

# Visualize segmentation results
visualize_segmentation(
    image_path="data/test/sample_BF.tif",
    mask_path="results/segmentation/sample_mask.tif",
    output_path="results/visualization/sample_overlay.png"
)

# Interactive visualization
visualize_segmentation(
    image_path="data/test/sample_BF.tif",
    mask_path="results/segmentation/sample_mask.tif",
    interactive=True
)
```

## Interactive GUI

The module includes a comprehensive GUI for visualizing 3D segmentation results with multi-layer support.

### GUI Features

- **File Browser**: Lists all available segmentation images in your data directory
- **Multi-layer Visualization**: Simultaneously displays:
  - Brightfield images
  - Ground truth segmentation
  - Blur heatmaps
  - Inference predictions
  - Final postprocessed segmentation
- **Interactive Controls**: 
  - Toggle layer visibility
  - Adjust opacity for each layer
  - Change data directory
- **2D Visualization with Z-Slider**: View 3D data in 2D mode with a slider to navigate through z-slices

### Expected Data Structure

The GUI expects the following directory structure:

```
data/
├── sample_plates_processed/
│   ├── split_3d/                    # Brightfield and ground truth
│   │   ├── p2126_J03_BF_3d.tif
│   │   ├── p2126_J03_Cells_3d.tif
│   │   └── ...
│   ├── blur_heatmaps/               # Blur analysis results
│   │   ├── p2126_J03_BF_3d_blur_heatmap.tif
│   │   └── ...
│   └── segmentation_2d/cyto3/test/
│       ├── masks_3d/                # Inference predictions
│       │   ├── p2126_J03_masks_3d.tif
│       │   └── ...
│       └── tracking/final/          # Final postprocessed results
│           ├── p2126_J03_masks_3d.tif
│           └── ...
```

### File Naming Conventions

- **Base name**: `p2126_J03` (plate_well format)
- **Brightfield**: `{base_name}_BF_3d.tif`
- **Ground truth**: `{base_name}_Cells_3d.tif`
- **Blur heatmap**: `{base_name}_BF_3d_blur_heatmap.tif`
- **Inference**: `{base_name}_masks_3d.tif`
- **Final result**: `{base_name}_masks_3d.tif` (in tracking/final/)

## Command Line Usage

### Safe GUI Launch

```bash
# Launch GUI with dependency checking
python scripts/launch_gui_safe.py

# Launch with specific backend
python scripts/launch_gui_safe.py --backend pyqt5

# Check dependencies only
python scripts/launch_gui_safe.py --check-only
```

### Visualization Scripts

```bash
# View 3D data
python -m src.visualize.view_3d_tiff --input data/3d_stack.tif

# View 4D data
python -m src.visualize.view_4d_tiff --input data/4d_timeseries.tif

# Generate prediction overlays
python -m src.visualize.visualize_prediction \
    --image data/test/sample_BF.tif \
    --mask results/segmentation/sample_mask.tif \
    --output results/visualization/overlay.png
```

## Installation

### Prerequisites

Make sure you have the following packages installed:

```bash
pip install PyQt5 napari[all] tifffile numpy scikit-image
```

**Important**: Use PyQt5, not PySide6, for better napari compatibility and to avoid threading issues.

## Configuration

### GUI Configuration

```yaml
# config/gui_config.yaml
gui:
  default_backend: "pyqt5"
  window_size: [1200, 800]
  auto_save: true
  
visualization:
  default_colormap: "viridis"
  default_opacity: 0.8
  max_layers: 10
  
export:
  default_format: "png"
  default_dpi: 300
  compression: "lzw"
```

### Visualization Presets

```yaml
# Visualization presets
presets:
  segmentation:
    colormap: "tab10"
    opacity: 0.6
    show_labels: true
    
  tracking:
    colormap: "hsv"
    opacity: 0.8
    show_tracks: true
    
  quality:
    colormap: "RdYlBu"
    opacity: 0.7
    show_contours: true
```

## Integration with Pipeline

The visualization module integrates seamlessly with other pipeline components:

```python
# Visualize preprocessing results
from src.preprocessing.blur_analysis import measure_dataset_blur_heatmaps
from src.visualize.visualize_prediction import visualize_blur_heatmap

# Generate and visualize blur heatmaps
blur_results = measure_dataset_blur_heatmaps("data/raw")
visualize_blur_heatmap(blur_results)

# Visualize inference results
from src.inference.inference_pipeline import InferencePipeline
from src.visualize.visualize_prediction import visualize_segmentation

# Run inference and visualize
pipeline = InferencePipeline(predictor, output_manager)
results = pipeline.run_inference("data/test")
visualize_segmentation(results)
```

## Testing

```bash
# Run visualization tests
python -m pytest tests/visualize/ -v
```

## Troubleshooting

### Common Issues

1. **PyQt5 not found**: 
   ```bash
   pip install PyQt5
   ```

2. **Napari not working**:
   ```bash
   pip install napari[all]
   ```

3. **Qt backend conflicts**:
   ```bash
   # Set environment variable
   export QT_API=pyqt5
   python scripts/launch_gui_safe.py
   ```

4. **Memory Issues**: 3D visualization can be memory-intensive
5. **Performance Issues**: Hide unused layers to improve performance

---

*For detailed documentation, advanced usage, and development information, see `README_ADVANCED.md`.*
