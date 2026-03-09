"""
Enhanced 3D Segmentation Visualization using Napari

This module provides improved functions for visualizing 3D segmentation results
with better layer management, colormaps, and viewing options.
"""

import os

# Fix Qt backend conflict - must be set before importing napari
os.environ['QT_API'] = 'pyqt5'

import napari
import numpy as np
import tifffile as tiff
from pathlib import Path
from typing import Dict, Optional, Union, List, Tuple
from skimage.measure import label
from skimage.morphology import closing, square, remove_small_objects


def create_enhanced_napari_viewer(
    brightfield: Union[str, Path, np.ndarray],
    ground_truth: Optional[Union[str, Path, np.ndarray]] = None,
    inference_prediction: Optional[Union[str, Path, np.ndarray]] = None,
    blur_heatmap: Optional[Union[str, Path, np.ndarray]] = None,
    final_segmentation: Optional[Union[str, Path, np.ndarray]] = None,
    layer_config: Optional[Dict] = None,
    viewer_title: str = "3D Segmentation Visualization"
) -> napari.Viewer:
    """
    Create an enhanced napari viewer with improved visualization settings.
    
    Args:
        brightfield: Brightfield image data or path
        ground_truth: Ground truth segmentation data or path
        inference_prediction: Inference prediction data or path
        blur_heatmap: Blur heatmap data or path
        final_segmentation: Final segmentation data or path
        layer_config: Configuration dictionary for layer properties
        viewer_title: Title for the napari viewer window
        
    Returns:
        napari.Viewer: Configured viewer with all layers
    """
    # Default layer configuration
    default_config = {
        'brightfield': {
            'colormap': 'gray',
            'opacity': 0.8,
            'contrast_limits': None,
            'gamma': 1.0
        },
        'ground_truth': {
            'opacity': 0.6,
            'visible': True
        },
        'inference': {
            'opacity': 0.5,
            'visible': True
        },
        'blur_heatmap': {
            'colormap': 'I Blue',
            'opacity': 0.3,
            'blending': 'additive',
            'visible': True
        },
        'final_segmentation': {
            'opacity': 0.7,
            'visible': True
        }
    }
    
    # Merge with user config
    if layer_config:
        for layer_type, config in layer_config.items():
            if layer_type in default_config:
                default_config[layer_type].update(config)
    
    # Create viewer
    viewer = napari.Viewer(title=viewer_title)
    
    # Helper function to load data
    def load_data(data_source):
        if isinstance(data_source, (str, Path)):
            return tiff.imread(str(data_source))
        return data_source
    
    # Add brightfield layer (base layer)
    try:
        bf_data = load_data(brightfield)
        bf_config = default_config['brightfield']
        
        viewer.add_image(
            bf_data,
            name="Brightfield",
            colormap=bf_config['colormap'],
            opacity=bf_config['opacity'],
            gamma=bf_config['gamma']
        )
        
        # Auto-adjust contrast if not specified
        if bf_config['contrast_limits'] is None:
            viewer.layers[-1].reset_contrast_limits()
        else:
            viewer.layers[-1].contrast_limits = bf_config['contrast_limits']
            
    except Exception as e:
        print(f"Warning: Could not load brightfield image: {e}")
        return None
    
    # Add ground truth segmentation
    if ground_truth is not None:
        try:
            gt_data = load_data(ground_truth).astype(np.uint32)
            gt_config = default_config['ground_truth']
            
            viewer.add_labels(
                gt_data,
                name="Ground Truth",
                opacity=gt_config['opacity'],
                visible=gt_config['visible']
            )
        except Exception as e:
            print(f"Warning: Could not load ground truth: {e}")
    
    # Add inference prediction
    if inference_prediction is not None:
        try:
            inf_data = load_data(inference_prediction).astype(np.uint32)
            inf_config = default_config['inference']
            
            viewer.add_labels(
                inf_data,
                name="Inference Prediction",
                opacity=inf_config['opacity'],
                visible=inf_config['visible']
            )
        except Exception as e:
            print(f"Warning: Could not load inference prediction: {e}")
    
    # Add blur heatmap
    if blur_heatmap is not None:
        try:
            blur_data = load_data(blur_heatmap)
            blur_config = default_config['blur_heatmap']
            
            viewer.add_image(
                blur_data,
                name="Blur Heatmap",
                colormap=blur_config['colormap'],
                opacity=blur_config['opacity'],
                blending=blur_config['blending'],
                visible=blur_config['visible']
            )
        except Exception as e:
            print(f"Warning: Could not load blur heatmap: {e}")
    
    # Add final segmentation
    if final_segmentation is not None:
        try:
            final_data = load_data(final_segmentation).astype(np.uint32)
            final_config = default_config['final_segmentation']
            
            viewer.add_labels(
                final_data,
                name="Final Segmentation",
                opacity=final_config['opacity'],
                visible=final_config['visible']
            )
        except Exception as e:
            print(f"Warning: Could not load final segmentation: {e}")
    
    # Enhanced 2D viewing settings with z-slider
    viewer.dims.ndisplay = 2  # Set to 2D display mode
    viewer.dims.axis_labels = ['Z', 'Y', 'X']  # Label the dimensions
    
    # Set initial slice to middle of z-stack
    if len(viewer.layers) > 0:
        first_layer = viewer.layers[0]
        if hasattr(first_layer, 'data') and len(first_layer.data.shape) >= 3:
            z_max = first_layer.data.shape[0]
            viewer.dims.current_step = (z_max // 2, 0, 0)  # Start at middle z-slice
    
    # Add scale bar if possible
    try:
        viewer.scale_bar.visible = True
        viewer.scale_bar.unit = "pixels"
    except:
        pass  # Scale bar not available in older versions
    
    return viewer


def quick_visualize_example():
    """
    Quick example function showing how to use the enhanced visualization.
    This replaces the hardcoded example from the original file.
    """
    # Example file paths (update these to your actual data)
    data_dir = Path(__file__).parent.parent.parent / "data"
    
    # Example for p2126_L11 (update base_name as needed)
    base_name = "p2126_L11"
    
    file_paths = {
        'brightfield': data_dir / "sample_plates_processed" / "split_3d" / f"{base_name}_BF_3d.tif",
        'ground_truth': data_dir / "sample_plates_processed" / "split_3d" / f"{base_name}_Cells_3d.tif",
        'blur_heatmap': data_dir / "sample_plates_processed" / "blur_heatmaps" / f"{base_name}_BF_3d_blur_heatmap.tif",
        'inference': data_dir / "sample_plates_processed" / "segmentation_2d" / "cyto3" / "test" / "masks_3d" / f"{base_name}_masks_3d.tif",
        'final': data_dir / "sample_plates_processed" / "segmentation_2d" / "cyto3" / "test" / "tracking" / "final" / f"{base_name}_masks_3d.tif"
    }
    
    # Check which files exist
    existing_files = {}
    for key, path in file_paths.items():
        if path.exists():
            existing_files[key] = path
            print(f"Found: {key} -> {path}")
        else:
            print(f"Missing: {key} -> {path}")
    
    if 'brightfield' not in existing_files:
        print("Error: Brightfield image not found. Cannot create visualization.")
        return None
    
    # Create enhanced visualization
    viewer = create_enhanced_napari_viewer(
        brightfield=existing_files['brightfield'],
        ground_truth=existing_files.get('ground_truth'),
        inference_prediction=existing_files.get('inference'),
        blur_heatmap=existing_files.get('blur_heatmap'),
        final_segmentation=existing_files.get('final'),
        viewer_title=f"3D Visualization - {base_name}"
    )
    
    return viewer


# Legacy compatibility - keep the original structure for existing code
data_dir = Path(__file__).parent.parent.parent / "data"

# Update paths to use the new structure
base_name = "p2126_L11"  # Change this to visualize different images

zstack_file = data_dir / "sample_plates_processed" / "split_3d" / f"{base_name}_BF_3d.tif"
segmentation_label_file = data_dir / "sample_plates_processed" / "split_3d" / f"{base_name}_Cells_3d.tif" 
prediction_file = data_dir / "sample_plates_processed" / "segmentation_2d" / "cyto3" / "test" / "masks_3d" / f"{base_name}_masks_3d.tif"
blur_map_file = data_dir / "sample_plates_processed" / "blur_heatmaps" / f"{base_name}_BF_3d_blur_heatmap.tif"
filtered_prediction_file = data_dir / "sample_plates_processed" / "segmentation_2d" / "cyto3" / "test" / "tracking" / "final" / f"{base_name}_masks_3d.tif"


if __name__ == "__main__":
    # Use the enhanced visualization function
    print("Starting enhanced 3D visualization...")
    
    # Check if files exist and create visualization
    viewer = quick_visualize_example()
    
    if viewer is not None:
        print("Napari viewer created successfully!")
        print("Available layers:")
        for layer in viewer.layers:
            print(f"  - {layer.name} ({type(layer).__name__})")
        
        # Run napari
        napari.run()
    else:
        print("Could not create visualization. Please check file paths.")
        
        # Fallback to original method if new paths don't work
        print("\nTrying legacy file paths...")
        try:
            # Load data from the legacy file paths (if they exist)
            if all(os.path.exists(str(f)) for f in [zstack_file, segmentation_label_file]):
                zstack = tiff.imread(str(zstack_file))
                segmentation_label = tiff.imread(str(segmentation_label_file)).astype(np.uint32)
                
                # Start napari viewer
                viewer = napari.Viewer()
                
                # Add layers to the viewer
                viewer.add_image(zstack, name="Z-Stacks", colormap="gray")
                viewer.add_labels(segmentation_label, name="Segmentation Label")
                
                # Add other layers if they exist
                if os.path.exists(str(prediction_file)):
                    prediction = tiff.imread(str(prediction_file)).astype(np.uint32)
                    viewer.add_labels(prediction, name="Prediction", opacity=0.5)
                
                if os.path.exists(str(blur_map_file)):
                    blur_map = tiff.imread(str(blur_map_file))
                    viewer.add_image(blur_map, name="Blur Map", colormap="I Blue", opacity=0.3)
                
                if os.path.exists(str(filtered_prediction_file)):
                    filtered_prediction = tiff.imread(str(filtered_prediction_file)).astype(np.uint32)
                    viewer.add_labels(filtered_prediction, name="Filtered Prediction", opacity=0.5)
                
                # Enhanced 2D settings with z-slider
                viewer.dims.ndisplay = 2  # Set to 2D display mode
                viewer.dims.axis_labels = ['Z', 'Y', 'X']  # Label the dimensions
                
                # Set initial slice to middle of z-stack
                if len(zstack.shape) >= 3:
                    z_max = zstack.shape[0]
                    viewer.dims.current_step = (z_max // 2, 0, 0)  # Start at middle z-slice
                
                # Run napari
                napari.run()
            else:
                print("Legacy files not found either. Please update file paths.")
                
        except Exception as e:
            print(f"Error with legacy visualization: {e}")