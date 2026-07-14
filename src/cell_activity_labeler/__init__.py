"""Cell Activity Labeler - threshold-based cell activity labeling tool

A labeling tool for threshold-based cell activity classification and analysis,
with optional Napari visualization support. Provides core labeling logic,
configuration management, and flexible frontend options (CLI, Voila, Napari).

Main Components:
    - ThresholdClassifier: Core processing class
    - ThresholdConfig: Configuration management
    - ThresholdWidget: Optional Napari visualization widget
    - Utilities for file I/O and validation

Quick Start:
    ```python
    from cell_activity_labeler import ThresholdClassifier, ThresholdConfig

    # Create classifier with default config
    classifier = ThresholdClassifier()

    # Process an image
    mask, threshold = classifier.process(image)
    ```

For Napari visualization:
    ```python
    import cell_activity_labeler as tac
    tac.register_for_napari()  # Add widget to active viewer
    ```
"""
from __future__ import annotations

# Core functionality
from .core import (
    ThresholdClassifier,
    ThresholdInstanceLabeler,
    ImagePreprocessor,
    ThresholdComputer,
    create_activity_labeled_image,
    create_classifier,
    get_labeling_strategy,
)

# Frontends
from .frontends import AbstractFrontend, CLIFrontend, VoilaFrontend

# Configuration
from .config import (
    ThresholdConfig,
    LabelingConfig,
    PreprocessingConfig,
    ThresholdParams,
    Method,
    Metric,
    NormalizeMode,
    get_default_config,
    validate_method_params
)

# Utilities
from .utils import (
    list_tif_files,
    save_config,
    load_config,
    validate_folder
)

# Version and metadata
__version__ = "0.2.0"
__author__ = "Serena Sritharan"
__email__ = "serena.sritharan@helmholtz-munich.de"

# Public API
__all__ = [
    # Core classes
    "ThresholdClassifier",
    "ThresholdInstanceLabeler",
    "ImagePreprocessor", 
    "ThresholdComputer",
    "create_activity_labeled_image",
    "create_classifier",
    "get_labeling_strategy",

    # Frontends
    "AbstractFrontend",
    "CLIFrontend",
    "VoilaFrontend",
    
    # Configuration
    "ThresholdConfig",
    "LabelingConfig",
    "PreprocessingConfig",
    "ThresholdParams",
    "Method",
    "Metric", 
    "NormalizeMode",
    "get_default_config",
    "validate_method_params",
    
    # UI
    "ThresholdWidget",
    "create_threshold_widget",
    "threshold_classifier_widget_factory",
    "threshold_classifier_widget",
    
    # Utilities
    "list_tif_files",
    "save_config",
    "load_config", 
    "validate_folder",
    
    # Napari integration
    "register_for_napari",
]


def __getattr__(name):
    """Load optional UI integrations only when explicitly requested."""
    if name in {
        "ThresholdWidget",
        "create_threshold_widget",
        "threshold_classifier_widget_factory",
        "threshold_classifier_widget",
    }:
        from . import ui

        return getattr(ui, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# Napari visualization hooks
def __napari_experimental_provide_dock_widget():
    """Fallback hook for Napari to provide the dock widget directly.

    This helps when entry-point/manifest discovery doesn't surface the
    MagicGUI widget automatically (useful during editable installs).
    Returns a widget factory or a list of (callable, kwargs) pairs.
    """
    try:
        from .ui import threshold_classifier_widget_factory

        # Return a list of (callable, kwargs) so napari can register the
        # widget with a display name. This is a forgiving format accepted
        # by various napari/plugin-manager versions.
        return [(threshold_classifier_widget_factory, {"name": "Cell Activity Labeler"})]
    except Exception:
        # If import fails for any reason, return nothing and allow normal
        # discovery to proceed. Napari will ignore None.
        return None

# Some napari/plugin-manager variants look for the non-dunder name as well;
# provide a fallback alias so discovery is more robust across versions.
napari_experimental_provide_dock_widget = __napari_experimental_provide_dock_widget


def register_for_napari(viewer=None):
    """Convenience helper to register or add the widget to a running Napari viewer.

    Usage (in Napari's Python console):
        import cell_activity_labeler as tac
        tac.register_for_napari()  # will add the widget to the active viewer

    Args:
        viewer: Napari viewer instance. If None, uses current active viewer.

    Returns:
        True on success

    Raises:
        RuntimeError: If no viewer is available
        Exception: If widget creation or registration fails
    """
    try:
        import napari
        from .ui import threshold_classifier_widget_factory

        widget = threshold_classifier_widget_factory()
        if viewer is None:
            # napari.current_viewer() is available when running inside Napari
            viewer = napari.current_viewer()
        if viewer is None:
            raise RuntimeError('No Napari viewer found; pass a Viewer instance to register_for_napari')

        viewer.window.add_dock_widget(widget, name='Cell Activity Labeler')
        return True
    except Exception:
        raise


# Convenience functions for quick usage
def process_image(image, method: Method = 'otsu', **kwargs):
    """Quick function to process a single image with default settings.
    
    Args:
        image: Input image array
        method: Thresholding method to use
        **kwargs: Additional configuration parameters
        
    Returns:
        Tuple of (binary_mask, threshold_value)
    """
    classifier = create_classifier(method=method, **kwargs)
    return classifier.process(image)


def create_config(**kwargs):
    """Create a ThresholdConfig with custom parameters.
    
    Args:
        **kwargs: Configuration parameters
        
    Returns:
        ThresholdConfig instance
    """
    return ThresholdConfig(**kwargs)
