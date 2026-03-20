"""User interface components for the threshold activity classifier.

This module provides Napari widgets and other UI components for interactive
threshold-based image analysis.
"""
from __future__ import annotations

from typing import Optional, TYPE_CHECKING, Any

import napari
import numpy as np
from magicgui import magicgui
from magicgui.widgets import Container, PushButton, ComboBox, SpinBox, FloatSpinBox

from threshold_activity_classifier.config import ThresholdConfig

if TYPE_CHECKING:
    import napari.layers
from threshold_activity_classifier.core import ThresholdClassifier


class ThresholdWidget:
    """Interactive threshold classification widget for Napari.
    
    This widget provides a user-friendly interface for adjusting threshold
    parameters and seeing real-time results.
    """
    
    def __init__(self, viewer: napari.Viewer):
        """Initialize the threshold widget.
        
        Args:
            viewer: Napari viewer instance
        """
        self.viewer = viewer
        self.classifier: Optional[ThresholdClassifier] = None
        self.current_image: Optional[np.ndarray] = None
        self._setup_widget()
        
    def _setup_widget(self) -> None:
        """Set up the widget components."""
        # Method selection
        self.method_combo = ComboBox(
            label="Method",
            choices=['otsu', 'yen', 'li', 'triangle', 'percentile', 'manual', 'local'],
            value='otsu',
            tooltip="Thresholding method to use"
        )
        
        # Metric selection  
        self.metric_combo = ComboBox(
            label="Metric",
            choices=['mean_intensity', 'max_intensity', 'percentile_90'],
            value='mean_intensity',
            tooltip="Image metric for analysis"
        )
        
        # Method-specific parameters
        self.percentile_spin = FloatSpinBox(
            label="Percentile",
            min=0.0,
            max=100.0,
            value=90.0,
            step=1.0,
            visible=False,
            tooltip="Percentile for percentile-based thresholding"
        )
        
        self.manual_spin = FloatSpinBox(
            label="Manual Value",
            min=0.0,
            max=1.0,
            value=0.5,
            step=0.01,
            visible=False,
            tooltip="Manual threshold value"
        )
        
        self.block_size_spin = SpinBox(
            label="Block Size",
            min=3,
            max=501,
            value=51,
            step=2,
            visible=False,
            tooltip="Block size for local thresholding (must be odd)"
        )
        
        # Preprocessing options
        self.gaussian_spin = FloatSpinBox(
            label="Gaussian σ",
            min=0.0,
            max=10.0,
            value=0.0,
            step=0.1,
            tooltip="Gaussian blur standard deviation"
        )
        
        self.median_spin = SpinBox(
            label="Median Radius",
            min=0,
            max=20,
            value=0,
            tooltip="Median filter radius"
        )
        
        self.bg_subtract_spin = SpinBox(
            label="Background Radius",
            min=0,
            max=50,
            value=0,
            tooltip="Background subtraction radius"
        )
        
        # Action buttons
        self.preview_button = PushButton(
            text="Preview",
            tooltip="Generate threshold preview"
        )
        
        self.apply_button = PushButton(
            text="Apply",
            tooltip="Apply threshold and create mask layer"
        )
        
        # Connect callbacks
        self.method_combo.changed.connect(self._on_method_changed)
        self.preview_button.clicked.connect(self._on_preview_clicked)
        self.apply_button.clicked.connect(self._on_apply_clicked)
        
        # Create container
        self.container = Container(
            widgets=[
                self.method_combo,
                self.metric_combo,
                self.percentile_spin,
                self.manual_spin,
                self.block_size_spin,
                self.gaussian_spin,
                self.median_spin,
                self.bg_subtract_spin,
                self.preview_button,
                self.apply_button,
            ],
            labels=True,
            layout='vertical'
        )
        
        # Initial setup
        self._on_method_changed()
    
    def _on_method_changed(self) -> None:
        """Handle method selection change."""
        method = self.method_combo.value
        
        # Hide all method-specific widgets
        self.percentile_spin.visible = False
        self.manual_spin.visible = False  
        self.block_size_spin.visible = False
        
        # Show relevant widgets for selected method
        if method == 'percentile':
            self.percentile_spin.visible = True
        elif method == 'manual':
            self.manual_spin.visible = True
        elif method in ('local', 'adaptive'):
            self.block_size_spin.visible = True
    
    def _get_current_config(self) -> ThresholdConfig:
        """Get current configuration from widget state."""
        from ..config import ThresholdParams, PreprocessingConfig
        
        params = ThresholdParams(
            percentile=self.percentile_spin.value if self.percentile_spin.visible else None,
            manual_value=self.manual_spin.value if self.manual_spin.visible else None,
            block_size=self.block_size_spin.value
        )
        
        preprocessing = PreprocessingConfig(
            gaussian_sigma=self.gaussian_spin.value,
            median_footprint=self.median_spin.value,
            background_subtract_radius=self.bg_subtract_spin.value
        )
        
        return ThresholdConfig(
            method=self.method_combo.value,
            metric=self.metric_combo.value,
            params=params,
            preprocessing=preprocessing
        )
    
    def _get_active_image(self) -> Optional[np.ndarray]:
        """Get the currently active image layer."""
        if not self.viewer.layers:
            return None
            
        active_layer = self.viewer.layers.selection.active
        if active_layer and hasattr(active_layer, 'data') and hasattr(active_layer, '__class__') and 'Image' in str(active_layer.__class__):
            return np.asarray(active_layer.data)
        
        # Fallback to first image layer
        for layer in self.viewer.layers:
            if hasattr(layer, 'data') and hasattr(layer, '__class__') and 'Image' in str(layer.__class__):
                return np.asarray(layer.data)
                
        return None
    
    def _on_preview_clicked(self) -> None:
        """Handle preview button click."""
        image = self._get_active_image()
        if image is None:
            return
            
        config = self._get_current_config()
        self.classifier = ThresholdClassifier(config)
        
        try:
            mask, threshold = self.classifier.process(image)
            self._update_preview_layer(mask, f"Preview ({config.method})")
            print(f"Threshold value: {threshold}")
        except Exception as e:
            print(f"Preview failed: {e}")
    
    def _on_apply_clicked(self) -> None:
        """Handle apply button click."""
        image = self._get_active_image()
        if image is None:
            return
            
        config = self._get_current_config()
        classifier = ThresholdClassifier(config)
        
        try:
            mask, threshold = classifier.process(image)
            self._add_result_layer(mask, f"Threshold Mask ({config.method})")
            print(f"Applied threshold: {threshold}")
        except Exception as e:
            print(f"Apply failed: {e}")
    
    def _update_preview_layer(self, mask: np.ndarray, name: str) -> None:
        """Update or create preview layer."""
        # Remove existing preview
        for layer in list(self.viewer.layers):
            if layer.name.startswith("Preview"):
                self.viewer.layers.remove(layer)
        
        # Add new preview as labels layer
        self.viewer.add_labels(
            mask.astype(int),
            name=name,
            opacity=0.7,
            blending='additive'
        )
    
    def _add_result_layer(self, mask: np.ndarray, name: str) -> None:
        """Add result as new labels layer."""
        self.viewer.add_labels(
            mask.astype(int),
            name=name,
            opacity=0.8
        )


# Legacy magicgui function for backward compatibility
@magicgui(
    call_button='Preview',
    auto_call=False,
    method={'widget_type': 'ComboBox', 'choices': ['otsu', 'yen', 'li', 'triangle', 'percentile', 'manual', 'local']},
    metric={'widget_type': 'ComboBox', 'choices': ['mean_intensity', 'max_intensity', 'percentile_90']}
)
def threshold_classifier_widget(
    viewer: napari.Viewer,
    image_layer: Optional[Any] = None,
    method: str = 'otsu',
    metric: str = 'mean_intensity',
    percentile: float = 90.0,
    manual_value: float = 0.5,
    gaussian_sigma: float = 0.0
) -> None:
    """Legacy threshold classifier widget (magicgui version).
    
    This function is kept for backward compatibility with existing code.
    Consider using ThresholdWidget class for new implementations.
    """
    if image_layer is None:
        return
    
    # Import here to avoid circular imports
    from threshold_activity_classifier.config import ThresholdConfig, ThresholdParams, PreprocessingConfig
    from threshold_activity_classifier.core import ThresholdClassifier
    
    # Create configuration
    config = ThresholdConfig(
        method=method,  # type: ignore
        metric=metric,  # type: ignore
        params=ThresholdParams(
            percentile=percentile if method == 'percentile' else None,
            manual_value=manual_value if method == 'manual' else None
        ),
        preprocessing=PreprocessingConfig(
            gaussian_sigma=gaussian_sigma
        )
    )
    
    # Process image
    classifier = ThresholdClassifier(config)
    try:
        mask, threshold = classifier.process(np.asarray(image_layer.data))
        
        # Update or create mask layer
        mask_layer_name = "threshold_mask"
        existing_layer = None
        for layer in viewer.layers:
            if layer.name == mask_layer_name:
                existing_layer = layer
                break
        
        if existing_layer is not None:
            existing_layer.data = mask.astype(int)
        else:
            viewer.add_labels(mask.astype(int), name=mask_layer_name)
            
    except Exception as e:
        print(f"Threshold processing failed: {e}")


def threshold_classifier_widget_factory():
    """Factory function for creating the threshold classifier widget.
    
    Returns:
        The magicgui threshold classifier widget
    """
    return threshold_classifier_widget


def create_threshold_widget(viewer: napari.Viewer) -> ThresholdWidget:
    """Create a new threshold widget instance.
    
    Args:
        viewer: Napari viewer instance
        
    Returns:
        ThresholdWidget instance
    """
    return ThresholdWidget(viewer)

