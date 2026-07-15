"""Render BF / inference-mask / mCherry as independent matplotlib layers.

Headless-safe (matplotlib Agg backend) equivalent of napari's layer panel: each layer
is an individually toggleable, opacity-adjustable AxesImage artist.
"""

from typing import Optional

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from skimage.color import label2rgb


def normalize_for_display(
    img: np.ndarray, p_low: float = 1, p_high: float = 99
) -> np.ndarray:
    """Percentile-rescale an image to [0, 1] float for display only.

    Deliberately not using mcherry_metrics.core.preprocessing.ImagePreprocessor, which is
    coupled to ExtractionConfig and metric-extraction correctness — a different concern
    from display rescaling.
    """
    p_lo, p_hi = np.percentile(img, (p_low, p_high))
    if p_hi <= p_lo:
        return np.zeros_like(img, dtype=float)
    return np.clip((img.astype(float) - p_lo) / (p_hi - p_lo), 0, 1)


def _label_to_rgba(mask: np.ndarray) -> np.ndarray:
    """Colored label overlay with background (label 0) fully transparent."""
    rgb = label2rgb(mask, bg_label=0, bg_color=(0, 0, 0))
    alpha = (mask != 0).astype(float)
    return np.dstack([rgb, alpha])


def render_layers(
    bf: np.ndarray,
    mask: Optional[np.ndarray] = None,
    mcherry: Optional[np.ndarray] = None,
    ax=None,
    mask_alpha: float = 0.5,
    mcherry_alpha: float = 0.5,
    mcherry_cmap: str = "inferno",
):
    """Render BF (base) + optional mask/mCherry layers on one Axes.

    Returns (fig, layer_artists) where layer_artists is
    {"bf": AxesImage, "mask": AxesImage | None, "mcherry": AxesImage | None} — callers
    (or build_layer_controls) toggle visibility/opacity via the returned artists.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for render_layers")

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    layer_artists = {}
    layer_artists["bf"] = ax.imshow(normalize_for_display(bf), cmap="gray")

    layer_artists["mask"] = (
        ax.imshow(_label_to_rgba(mask), alpha=mask_alpha) if mask is not None else None
    )
    layer_artists["mcherry"] = (
        ax.imshow(
            normalize_for_display(mcherry), cmap=mcherry_cmap, alpha=mcherry_alpha
        )
        if mcherry is not None
        else None
    )

    ax.set_axis_off()
    return fig, layer_artists


def build_layer_controls(layer_artists: dict, fig):
    """ipywidgets Checkbox+FloatSlider per non-BF layer, wired to visibility/opacity."""
    import ipywidgets as widgets

    rows = []
    for name, artist in layer_artists.items():
        if name == "bf" or artist is None:
            continue

        checkbox = widgets.Checkbox(value=True, description=f"{name} visible")
        slider = widgets.FloatSlider(
            value=artist.get_alpha() or 1.0,
            min=0,
            max=1,
            step=0.05,
            description=f"{name} opacity",
        )

        def _make_visibility_cb(artist=artist):
            def _cb(change):
                artist.set_visible(change["new"])
                fig.canvas.draw_idle()

            return _cb

        def _make_alpha_cb(artist=artist):
            def _cb(change):
                artist.set_alpha(change["new"])
                fig.canvas.draw_idle()

            return _cb

        checkbox.observe(_make_visibility_cb(), names="value")
        slider.observe(_make_alpha_cb(), names="value")
        rows.append(widgets.HBox([checkbox, slider]))

    return widgets.VBox(rows)
