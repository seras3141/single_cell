__all__ = [
    'ThresholdWidget',
    'threshold_classifier_widget',
    'threshold_classifier_widget_factory',
    'create_threshold_widget',
    'NapariImageViewer',
    'visualize_image_with_napari',
    'visualize_sample_zstack_napari',
    'show_sample_images_napari',
    'create_and_display_napari_viewer',
    'ImageViewer',
]


def __getattr__(name):
    """Load UI objects lazily so backend imports do not initialize Napari."""
    if name in {
        'ThresholdWidget',
        'threshold_classifier_widget',
        'threshold_classifier_widget_factory',
        'create_threshold_widget',
    }:
        from . import widget

        return getattr(widget, name)

    if name in {
        'NapariImageViewer',
        'visualize_image_with_napari',
        'visualize_sample_zstack_napari',
        'show_sample_images_napari',
    }:
        from . import napari_viewer

        return getattr(napari_viewer, name)

    if name in {'ImageViewer', 'create_and_display_napari_viewer'}:
        from . import image_viewer_widget

        return getattr(image_viewer_widget, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
