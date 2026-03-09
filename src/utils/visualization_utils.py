from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def plot_blur_heatmap(heatmap_img, title):
    """
    Plots a heatmap image with a given title.

    Args:
        heatmap_img: 2D array-like, the heatmap image to display.
        title: str, the title for the plot.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap_img, cmap='hot', interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    plt.axis('off')
    plt.show()

def plot_blur_heatmap_3D(heatmap_img, title):
    """
    Plots a 3D heatmap with a scrollbar to scroll through z-slices.

    Args:
        heatmap_img: 3D numpy array (z, y, x)
        title: str, the title for the plot
    """
    if heatmap_img.ndim != 3:
        raise ValueError("heatmap_img must be a 3D array (z, y, x)")

    # Initial slice
    z_slices = heatmap_img.shape[0]
    current_slice = 0

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.2)  # make space for slider

    img_display = ax.imshow(heatmap_img[current_slice], cmap='hot', interpolation='nearest')
    ax.set_title(f"{title}\nSlice {current_slice + 1}/{z_slices}")
    plt.axis('off')
    _cbar = plt.colorbar(img_display, ax=ax)

    # Slider axis and object
    ax_slider = plt.axes((0.2, 0.05, 0.6, 0.03))
    slider = Slider(ax_slider, 'Slice', 0, z_slices - 1, valinit=current_slice, valstep=1)

    # Update function for slider
    def update(val):
        slice_idx = int(val)
        img_display.set_data(heatmap_img[slice_idx])
        ax.set_title(f"{title}\nSlice {slice_idx + 1}/{z_slices}")
        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()

    # # If running with a non-interactive backend (e.g., Agg), save the figure instead of showing it.
    # if plt.get_backend().lower().endswith('agg'):
    #     safe_title = "".join(c if c.isalnum() or c in (" ", "_") else "_" for c in title).strip().replace(" ", "_")
    #     out_path = Path.cwd() / f"{safe_title}_slices.png"
    #     fig.savefig(out_path, bbox_inches='tight')
    #     plt.close(fig)
    # else:
    #     plt.show()
