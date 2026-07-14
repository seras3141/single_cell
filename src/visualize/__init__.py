"""Visualization tools for single cell analysis."""

# from .view_3d_tiff import view_3d_data, parse_filename
# from .view_4d_tiff import view_4d_data
from .visualize_prediction import create_enhanced_napari_viewer, quick_visualize_example
from .paths import resolve_related_paths
from .headless_layers import render_layers, build_layer_controls, normalize_for_display
from .browse import build_index, resolve_selection, build_browser_widget

try:
    # PyQt5 is only installed via the optional 'gui' extra; qt_config.configure_qt_backend()
    # calls sys.exit(1) when it's missing, which would otherwise crash any import of this
    # package (even headless-safe submodules like .paths) on HPC/headless environments.
    from .segmentation_gui import SegmentationVisualizationGUI
except (ImportError, SystemExit):
    SegmentationVisualizationGUI = None

__all__ = [
    # "view_3d_data",
    # "parse_filename",
    # "view_4d_data",
    "create_enhanced_napari_viewer",
    "quick_visualize_example",
    "SegmentationVisualizationGUI",
    "resolve_related_paths",
    "render_layers",
    "build_layer_controls",
    "normalize_for_display",
    "build_index",
    "resolve_selection",
    "build_browser_widget",
]
