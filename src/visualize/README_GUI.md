# 3D Segmentation Visualization GUI

A PyQt5-based graphical user interface for visualizing 3D segmentation results using napari in 2D mode with z-stack navigation.

## Features

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
- **Threading Fixed**: No more Qt threading issues - napari runs in main thread

## Installation

### Prerequisites

Make sure you have the following packages installed:

```bash
pip install PyQt5 napari[all] tifffile numpy scikit-image
```

**Important**: Use PyQt5, not PySide6, for better napari compatibility and to avoid threading issues.

### Quick Start

1. **Launch the GUI**:
   ```bash
   python launch_gui_safe.py
   ```

2. **Or run directly**:
   ```bash
   python src/visualize/segmentation_gui.py
   ```

3. **Or import in Python**:
   ```python
   from src.visualize import SegmentationVisualizationGUI
   from PyQt5.QtWidgets import QApplication
   
   app = QApplication([])
   gui = SegmentationVisualizationGUI()
   gui.show()
   app.exec_()
   ```

## Usage

### Data Directory Structure

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

### GUI Components

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

### Napari Visualization

The visualization includes:

- **Brightfield**: Gray colormap, base layer
- **Ground Truth**: Label layer with default coloring
- **Blur Heatmap**: I Blue colormap, additive blending
- **Inference Prediction**: Label layer with transparency
- **Final Segmentation**: Label layer showing final results

**2D Mode with Z-Slider**: All data is displayed in 2D mode with a slider at the bottom to navigate through z-slices. This provides better performance and easier navigation compared to 3D mode.

## Enhanced Features

### Improved Napari Visualization

The updated `visualize_prediction.py` includes:

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

## File Naming Conventions

The GUI expects files to follow this naming pattern:

- **Base name**: `p2126_J03` (plate_well format)
- **Brightfield**: `{base_name}_BF_3d.tif`
- **Ground truth**: `{base_name}_Cells_3d.tif`
- **Blur heatmap**: `{base_name}_BF_3d_blur_heatmap.tif`
- **Inference**: `{base_name}_masks_3d.tif`
- **Final result**: `{base_name}_masks_3d.tif` (in tracking/final/)

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

3. **Threading errors**:
   - This has been fixed in the new version
   - Napari now runs in the main thread using QTimer

4. **Qt backend conflicts**:
   ```bash
   # Set environment variable
   export QT_API=pyqt5
   python launch_gui_safe.py
   ```

### Performance Tips

- **Memory**: 3D visualization can be memory-intensive
- **Layer Management**: Hide unused layers to improve performance
- **File Size**: Large TIFF files may take time to load

## Development

### Adding New Features

To extend the GUI:

1. **New Layer Types**: Add to `FileSearcher.find_related_files()`
2. **Visualization Options**: Extend the configuration panel
3. **File Formats**: Update loading functions to support new formats

### Code Structure

- `segmentation_gui.py`: Main GUI application
- `visualize_prediction.py`: Enhanced napari visualization functions
- `launch_visualization_gui.py`: Simple launcher script

## Examples

### Programmatic Usage

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
