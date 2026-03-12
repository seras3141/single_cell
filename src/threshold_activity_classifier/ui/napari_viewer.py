# Napari visualization module for single-cell analysis
from pathlib import Path
import ipywidgets as widgets
from IPython.display import display
import tifffile as tiff

try:
    import napari
    NAPARI_AVAILABLE = True
except ImportError:
    NAPARI_AVAILABLE = False

import numpy as np

from utils.io import find_label_for_image, ensure_2d
from core.classifier import create_activity_labeled_image


class NapariImageViewer:
    """
    Unified napari viewer for single-cell analysis.
    
    Supports both direct array input and path-based loading with interactive widgets.
    Can visualize 2D single images and 3D z-stacks.
    
    Usage:
        # Direct array mode (2D)
        viewer = NapariImageViewer()
        viewer.open(img, lbl, activity_labels, metrics_df, "image_name")
        
        # Path-based mode with widgets
        viewer = NapariImageViewer(metrics_df, root_path)
        viewer.display()
    """
    
    def __init__(self, metrics_df=None, root_path=None):
        """
        Initialize the viewer.
        
        Parameters
        ----------
        metrics_df : pd.DataFrame, optional
            Classification results DataFrame. Required for widget-based interface.
        root_path : Path or str, optional
            Root path for image files. Required for widget-based interface.
        """
        self.metrics_df = metrics_df
        self.root_path = Path(root_path) if root_path is not None else None
        self.current_viewer = None
        self._widgets_initialized = False
        
        # Initialize widgets only if metrics_df is provided
        if metrics_df is not None and root_path is not None:
            self._setup_widgets()

    # ------------------ Direct array mode methods -------------------------
    
    def open(self, img, lbl=None, activity_labels=None, metrics_df=None, 
             image_name="image", activity_is_bin=False):
        """
        Open napari viewer with provided image arrays.
        
        Parameters
        ----------
        img : np.ndarray
            Intensity image (2D or 3D).
        lbl : np.ndarray, optional
            Instance segmentation label image.
        activity_labels : np.ndarray, optional
            Activity classification labels. Positive values = active, negative = dead.
        metrics_df : pd.DataFrame, optional
            Classification results for statistics display.
        image_name : str
            Display name for the image.
        activity_is_bin : bool
            If True, activity_labels already contains binary values (0=bg, 1=active, 2=dead).
            If False, will convert from signed labels.
            
        Returns
        -------
        napari.Viewer
            The napari viewer instance.
        """
        if not NAPARI_AVAILABLE:
            print("Napari is not available. Install with: pip install napari[all]")
            return None
            
        title = f"Image Browser: {image_name}"
        self.current_viewer = napari.Viewer(title=title)
        
        # Process intensity image
        img = ensure_2d(img).astype(float)
        lo, hi = np.percentile(img, [0.1, 99.9])
        img = np.clip(img, lo, hi)
        
        if metrics_df is not None:
            self.metrics_df = metrics_df
        
        # Add original intensity image
        self.current_viewer.add_image(img, name=image_name, colormap="yellow", opacity=0.9)
        
        # Add instance segmentation labels
        if lbl is not None:
            if not np.issubdtype(lbl.dtype, np.integer):
                lbl = lbl.astype(np.int32)
            self.current_viewer.add_labels(lbl, name=f"{image_name}_labels", opacity=0.4)
        
        # Add activity classification overlay
        if activity_labels is not None:
            self._add_activity_layer(activity_labels, image_name, activity_is_bin)
        else:
            print(f"No activity classification data for {image_name}")
        
        self.current_viewer.window.resize(1200, 800)
        return self.current_viewer
    
    def open_3d(self, image_volume, label_volume=None, activity_volume=None,
                title="3D Volume", show_original=True, show_labels=True, 
                show_activity=True):
        """
        Open napari viewer with 3D volume data.
        
        Parameters
        ----------
        image_volume : np.ndarray
            3D intensity volume (Z, Y, X).
        label_volume : np.ndarray, optional
            3D instance segmentation labels.
        activity_volume : np.ndarray, optional
            3D activity classification labels (signed or binary).
        title : str
            Window title.
        show_original : bool
            Whether to show original intensity volume.
        show_labels : bool
            Whether to show instance labels.
        show_activity : bool
            Whether to show activity classification.
            
        Returns
        -------
        napari.Viewer
            The napari viewer instance.
        """
        if not NAPARI_AVAILABLE:
            print("Napari is not available. Install with: pip install napari[all]")
            return None
        
        self.current_viewer = napari.Viewer(title=title)

        lo, hi = np.percentile(image_volume, [0.1, 99.9])
        image_volume = np.clip(image_volume, lo, hi)

        if show_original:
            self.current_viewer.add_image(
                image_volume, name='Original Images',
                colormap='yellow', opacity=0.7, scale=(1, 1, 1)
            )
        
        if show_labels and label_volume is not None:
            self.current_viewer.add_labels(label_volume, name='Instance Labels', opacity=0.5)
        
        if show_activity and activity_volume is not None:
            self._add_activity_layer(activity_volume, 'Activity Classification', activity_is_bin=True)
        
        self.current_viewer.window.resize(1400, 1000)
        return self.current_viewer
    
    def _add_activity_layer(self, activity_labels, image_name, activity_is_bin=False):
        """Add activity classification layer to viewer."""
        if self.current_viewer is None:
            raise RuntimeError("Viewer not initialized. Call open() first.")
        if not np.issubdtype(activity_labels.dtype, np.integer):
            activity_labels = activity_labels.astype(np.int32)
        
        # Convert to binary format: 0=bg, 1=active, 2=dead
        if activity_is_bin:
            activity_binary = activity_labels.astype(np.uint8)
        else:
            activity_binary = np.zeros_like(activity_labels, dtype=np.uint8)
            activity_binary[activity_labels > 0] = 1
            activity_binary[activity_labels < 0] = 2
        
        layer = self.current_viewer.add_labels(
            activity_binary,
            name=f"{image_name}_activity",
            opacity=0.6,
        )
        self._set_activity_colors(layer)
    
    def _set_activity_colors(self, layer):
        """Set activity classification colors on a labels layer."""
        try:
            layer.color = {0: "transparent", 1: "green", 2: "magenta"}
        except Exception:
            try:
                layer.colors = ['black', 'green', 'magenta']
            except Exception:
                try:
                    layer.colormap = 'viridis'
                    layer.opacity = 0.6
                except Exception:
                    pass
    
    def close(self):
        """Close the napari viewer if open."""
        if self.current_viewer is not None:
            try:
                self.current_viewer.close()
            except Exception:
                pass
            self.current_viewer = None

    # -------------------------------------------------------------------------
    # Path-based widget interface methods
    # -------------------------------------------------------------------------
    
    def _setup_widgets(self):
        """Create interactive widgets for image selection."""
        if self.metrics_df is None:
            return
            
        # Sample selector
        available_samples = sorted(self.metrics_df['sample'].dropna().unique())
        self.sample_selector = widgets.Dropdown(
            options=['All'] + list(available_samples),
            value='All',
            description='Sample:',
            layout=widgets.Layout(width='200px')
        )
        
        # Image selector
        self.image_selector = widgets.Dropdown(
            options=[],
            description='Image:',
            layout=widgets.Layout(width='300px')
        )
        
        # Display options
        self.show_original = widgets.Checkbox(value=True, description='Original Image')
        self.show_labels = widgets.Checkbox(value=True, description='Instance Labels')
        self.show_activity = widgets.Checkbox(value=True, description='Activity Classification')
        
        # Z-stack options
        self.max_z_slices = widgets.IntSlider(
            value=20, min=5, max=50, step=5,
            description='Max Z-slices:',
            layout=widgets.Layout(width='300px')
        )
        
        # Buttons
        self.open_napari_btn = widgets.Button(
            description='Open Single Image',
            button_style='primary',
            icon='eye',
            layout=widgets.Layout(width='150px')
        )
        
        self.open_zstack_btn = widgets.Button(
            description='Open Z-Stack 3D',
            button_style='success',
            icon='cube',
            layout=widgets.Layout(width='150px')
        )
        
        self.close_napari_btn = widgets.Button(
            description='Close Napari',
            button_style='warning',
            icon='times',
            layout=widgets.Layout(width='150px')
        )
        
        self.show_multiple_btn = widgets.Button(
            description='Show Sample (5 images)',
            button_style='info',
            icon='images',
            layout=widgets.Layout(width='180px')
        )
        
        # Output area
        self.info_output = widgets.Output()
        
        # Event handlers
        self.sample_selector.observe(self._update_image_list, names='value')
        self.open_napari_btn.on_click(self._on_open_napari)
        self.open_zstack_btn.on_click(self._on_open_zstack)
        self.close_napari_btn.on_click(self._on_close_napari)
        self.show_multiple_btn.on_click(self._on_show_multiple)
        
        # Initialize image list
        self._update_image_list({'new': 'All'})
        self._widgets_initialized = True
    
    def _update_image_list(self, change):
        """Update available images based on selected sample."""
        sample = change['new']
        if self.metrics_df is None:
            raise RuntimeError("Metrics DataFrame not set. Cannot update image list.")
        
        if sample == 'All':
            available_images = sorted(self.metrics_df['image'].unique())
        else:
            available_images = sorted(
                self.metrics_df[self.metrics_df['sample'] == sample]['image'].unique()
            )
        
        options = []
        for img in available_images:
            img_data = self.metrics_df[self.metrics_df['image'] == img]
            n_cells = len(img_data)
            n_active = img_data['is_active'].sum()
            activity_rate = (n_active / n_cells) * 100 if n_cells > 0 else 0
            z_idx = img_data['z_index'].iloc[0] if len(img_data) > 0 else 'N/A'
            
            display_name = f"{img} (z={z_idx}, {n_cells}c, {activity_rate:.0f}% active)"
            options.append((display_name, img))
        
        self.image_selector.options = options
        if options:
            self.image_selector.value = options[0][1]
    
    def _on_open_napari(self, button):
        """Handle single image open button click."""
        with self.info_output:
            self.info_output.clear_output(wait=True)
            
            if not self.image_selector.value:
                print("Please select an image first.")
                return
            
            self.close()
            
            print(f"Opening {self.image_selector.value} in napari...")
            self.current_viewer = visualize_image_with_napari(
                self.image_selector.value,
                self.metrics_df,
                self.root_path,
                show_original=self.show_original.value,
                show_labels=self.show_labels.value,
                show_activity=self.show_activity.value
            )
    
    def _on_open_zstack(self, button):
        """Handle z-stack open button click."""
        with self.info_output:
            self.info_output.clear_output(wait=True)
            
            sample = self.sample_selector.value
            if sample == 'All':
                print("Please select a specific sample to view z-stack.")
                return
            
            self.close()
            
            print(f"Opening z-stack for sample {sample} in napari...")
            self.current_viewer = visualize_sample_zstack_napari(
                sample,
                self.metrics_df,
                self.root_path,
                show_original=self.show_original.value,
                show_labels=self.show_labels.value,
                show_activity=self.show_activity.value,
                max_z_slices=self.max_z_slices.value
            )
    
    def _on_close_napari(self, button):
        """Handle close button click."""
        with self.info_output:
            self.info_output.clear_output(wait=True)
            
            if self.current_viewer is not None:
                self.close()
                print("Napari viewer closed.")
            else:
                print("No napari viewer is currently open.")
    
    def _on_show_multiple(self, button):
        """Handle show multiple images button click."""
        with self.info_output:
            self.info_output.clear_output(wait=True)
            
            sample = self.sample_selector.value
            if sample == 'All':
                print("Please select a specific sample to show multiple images.")
                return
            
            print(f"Opening multiple images from sample {sample}...")
            viewers = show_sample_images_napari(
                self.metrics_df, sample, self.root_path, max_images=5
            )
            if viewers:
                print(f"Opened {len(viewers)} napari viewers.")
    
    def display(self):
        """Display the interactive widget interface."""
        if not self._widgets_initialized:
            if self.metrics_df is None or self.root_path is None:
                print("Cannot display widgets: metrics_df and root_path required.")
                return
            self._setup_widgets()
        
        controls_row = widgets.HBox([self.sample_selector, self.image_selector])
        options_row = widgets.HBox([
            self.show_original, self.show_labels, self.show_activity
        ])
        zstack_row = widgets.HBox([self.max_z_slices])
        buttons_row1 = widgets.HBox([
            self.open_napari_btn, self.open_zstack_btn, self.close_napari_btn
        ])
        buttons_row2 = widgets.HBox([self.show_multiple_btn])
        
        interface = widgets.VBox([
            widgets.HTML("<h3>Interactive Napari Image Viewer</h3>"),
            widgets.HTML("<b>Sample and Image Selection:</b>"),
            controls_row,
            widgets.HTML("<b>Display Options:</b>"),
            options_row,
            widgets.HTML("<b>Z-Stack Options:</b>"),
            zstack_row,
            widgets.HTML("<b>Viewer Actions:</b>"),
            buttons_row1,
            widgets.HTML("<b>Additional Actions:</b>"),
            buttons_row2,
            widgets.HTML("<b>Information:</b>"),
            self.info_output
        ])
        
        display(interface)



# -------------------------------------------------------------------------
# Standalone visualization functions
# -------------------------------------------------------------------------

def visualize_image_with_napari(image_name, metrics_df, root_path,
                                show_original=True, show_labels=True, show_activity=True):
    """
    Open napari with the specified image and activity-coded labels.
    
    Parameters
    ----------
    image_name : str
        Name of the image file to visualize.
    metrics_df : pd.DataFrame
        DataFrame with classification results.
    root_path : Path
        Root path to search for images.
    show_original : bool
        Whether to show the original intensity image.
    show_labels : bool
        Whether to show the original instance labels.
    show_activity : bool
        Whether to show the activity-coded labels.
        
    Returns
    -------
    napari.Viewer or None
        The napari viewer instance, or None on error.
    """
    if not NAPARI_AVAILABLE:
        print("Napari is not available. Install with: pip install napari[all]")
        return None
    
    image_files = list(Path(root_path).rglob(image_name))
    if not image_files:
        print(f"Image {image_name} not found under {root_path}")
        return None
    
    img_path = image_files[0]
    lbl_path = find_label_for_image(img_path)
    
    if lbl_path is None or not lbl_path.exists():
        print(f"Label file not found for {image_name}")
        return None
    
    try:
        img = tiff.imread(str(img_path))
        lbl = tiff.imread(str(lbl_path))
        
        img = ensure_2d(img).astype(float)
        lbl = ensure_2d(lbl).astype(np.int32)
        
        image_classification = metrics_df[metrics_df['image'] == image_name].copy()
        
        if len(image_classification) == 0:
            print(f"No classification data found for {image_name}")
            return None
        
        activity_labels = create_activity_labeled_image(lbl, image_classification)
        
        viewer = napari.Viewer(title=f"Cell Activity: {image_name}")
        
        if show_original:
            viewer.add_image(img, name='Original Image', colormap='yellow', opacity=0.7)
        
        if show_labels:
            viewer.add_labels(lbl, name='Instance Labels', opacity=0.5)
        
        if show_activity:
            activity_layer = viewer.add_labels(
                activity_labels, name='Activity Classification', opacity=0.8
            )
            try:
                activity_layer.colormap = 'viridis' # type: ignore
                activity_layer.opacity = 0.6
            except Exception:
                pass
        
        n_total = len(image_classification)
        n_active = image_classification['is_active'].sum()
        n_dead = n_total - n_active
        activity_rate = (n_active / n_total) * 100
        
        print(f"Image: {image_name}")
        print(f"   Total cells: {n_total}")
        print(f"   Active: {n_active} ({activity_rate:.1f}%)")
        print(f"   Dead: {n_dead} ({100-activity_rate:.1f}%)")
        print(f"   Threshold used: {image_classification['threshold'].iloc[0]:.1f}")
        
        viewer.window.resize(1200, 800)
        return viewer
        
    except Exception as e:
        print(f"Error loading image {image_name}: {e}")
        return None


def show_sample_images_napari(metrics_df, sample_id, root, z_indices=None, max_images=5):
    """
    Show multiple images from a sample in separate napari windows.
    
    Parameters
    ----------
    metrics_df : pd.DataFrame
        DataFrame with classification results.
    sample_id : str
        Sample identifier (e.g., 'B02').
    root : Path
        Root path to search for images.
    z_indices : list, optional
        Specific z-indices to show, or None for automatic selection.
    max_images : int
        Maximum number of images to open.
        
    Returns
    -------
    list
        List of napari viewer instances.
    """
    if not NAPARI_AVAILABLE:
        print("Napari is not available. Install with: pip install napari[all]")
        return []
    
    sample_images = metrics_df[metrics_df['sample'] == sample_id]['image'].unique()
    
    if len(sample_images) == 0:
        print(f"No images found for sample {sample_id}")
        return []
    
    if z_indices is not None:
        sample_data = metrics_df[
            (metrics_df['sample'] == sample_id) & 
            (metrics_df['z_index'].isin(z_indices))
        ]
        filtered_images = sample_data['image'].unique()
        sample_images = [img for img in sample_images if img in filtered_images]
    
    if len(sample_images) > max_images:
        step = len(sample_images) // max_images
        sample_images = sample_images[::step][:max_images]
    
    print(f"Opening {len(sample_images)} images for sample {sample_id}")
    
    viewers = []
    for i, image_name in enumerate(sample_images):
        print(f"Opening image {i+1}/{len(sample_images)}: {image_name}")
        viewer = visualize_image_with_napari(image_name, metrics_df, root)
        if viewer is not None:
            viewer.window.move(50 * i, 50 * i)
            viewers.append(viewer)
    
    return viewers


def visualize_sample_zstack_napari(sample_id, metrics_df, root_path,
                                   show_original=True, show_labels=True, show_activity=True,
                                   max_z_slices=None):
    """
    Open napari with all z-stack images from a sample in a single 3D viewer.
    
    Parameters
    ----------
    sample_id : str
        Sample identifier (e.g., 'B02').
    metrics_df : pd.DataFrame
        DataFrame with classification results.
    root_path : Path
        Root path to search for images.
    show_original : bool
        Whether to show the original intensity images.
    show_labels : bool
        Whether to show the original instance labels.
    show_activity : bool
        Whether to show the activity-coded labels.
    max_z_slices : int, optional
        Maximum number of z-slices to load (None for all).
        
    Returns
    -------
    napari.Viewer or None
        The napari viewer instance, or None on error.
    """
    if not NAPARI_AVAILABLE:
        print("Napari is not available. Install with: pip install napari[all]")
        return None
    
    # Get all images for this sample, sorted by z-index
    sample_data = metrics_df[metrics_df['sample'] == sample_id].copy()
    if len(sample_data) == 0:
        print(f"No images found for sample {sample_id}")
        return None
    
    # Sort by z-index and optionally limit
    sample_images = sample_data.sort_values('z_index')['image'].unique()
    z_indices = sorted(sample_data['z_index'].unique())
    
    if max_z_slices is not None and len(sample_images) > max_z_slices:
        step = len(sample_images) // max_z_slices
        sample_images = sample_images[::step][:max_z_slices]
        z_indices = z_indices[::step][:max_z_slices]
    
    print(f"Loading {len(sample_images)} z-slices for sample {sample_id}")
    
    image_stack = []
    label_stack = []
    activity_stack = []
    valid_z_indices = []
    
    for i, image_name in enumerate(sample_images):
        try:
            image_files = list(Path(root_path).rglob(image_name))
            if not image_files:
                print(f"Warning: Image {image_name} not found, skipping")
                continue
            
            img_path = image_files[0]
            lbl_path = find_label_for_image(img_path)
            
            if lbl_path is None or not lbl_path.exists():
                print(f"Warning: Label file not found for {image_name}, skipping")
                continue
            
            img = tiff.imread(str(img_path))
            lbl = tiff.imread(str(lbl_path))
            
            img = ensure_2d(img).astype(float)
            lbl = ensure_2d(lbl).astype(np.int32)
            
            image_classification = metrics_df[metrics_df['image'] == image_name].copy()
            
            if len(image_classification) == 0:
                print(f"Warning: No classification data found for {image_name}, skipping")
                continue
            
            activity_labels = create_activity_labeled_image(lbl, image_classification)
            
            image_stack.append(img)
            label_stack.append(lbl)
            activity_stack.append(activity_labels)
            valid_z_indices.append(z_indices[i] if i < len(z_indices) else i)
            
        except Exception as e:
            print(f"Error processing {image_name}: {e}")
            continue
    
    if len(image_stack) == 0:
        print("No valid images were loaded")
        return None
    
    image_volume = np.stack(image_stack, axis=0)
    label_volume = np.stack(label_stack, axis=0)
    activity_volume = np.stack(activity_stack, axis=0)
    
    activity_volume_binary = np.zeros_like(activity_volume, dtype=np.uint8)
    activity_volume_binary[activity_volume > 0] = 1
    activity_volume_binary[activity_volume < 0] = 2
    
    print(f"Created 3D volumes with shape: {image_volume.shape}")
    
    viewer = napari.Viewer(title=f"Sample {sample_id} - Z-Stack ({len(image_stack)} slices)")
    
    if show_original:
        viewer.add_image(
            image_volume, name='Original Images',
            colormap='yellow', opacity=0.7, scale=(1, 1, 1)
        )
    
    if show_labels:
        viewer.add_labels(label_volume, name='Instance Labels', opacity=0.5)
    
    if show_activity:
        activity_layer = viewer.add_labels(
            activity_volume_binary, name='Activity Classification', opacity=0.8
        )
        try:
            # Modern napari: assign a dict mapping label index -> color string
            activity_layer.color = {0: 'black', 1: 'green', 2: 'magenta'} # type: ignore
        except Exception:
            try:
                # Older/other napari: assign to .colors (numpy array of RGBA) or .color_cycle
                activity_layer.colors = ['black', 'green', 'magenta'] # type: ignore
            except Exception as e:
                print(f"Could not set colors: {e}")
                try:
                    activity_layer.colormap = 'viridis' # type: ignore
                    activity_layer.opacity = 0.6
                except Exception:
                    pass
    
    sample_stats = sample_data.groupby('z_index').agg({
        'is_active': ['count', 'sum'],
        'metric_value': ['mean', 'median'],
        'threshold': 'first'
    }).round(2)
    
    total_cells = sample_data.shape[0]
    total_active = sample_stats['is_active', 'sum'].sum()
    overall_activity_rate = (total_active / total_cells) * 100 if total_cells > 0 else 0
    
    print(f"\nSample {sample_id} Z-Stack Statistics:")
    print(f"   Total z-slices loaded: {len(image_stack)}")
    print(f"   Total cells across all slices: {total_cells}")
    print(f"   Active cells: {total_active} ({overall_activity_rate:.1f}%)")
    print(f"   Dead cells: {total_cells - total_active} ({100-overall_activity_rate:.1f}%)")
    
    viewer.window.resize(1400, 1000)
    return viewer


def create_and_display_napari_viewer(metrics_df, root):
    """
    Create and display the interactive napari viewer interface.
    
    Parameters
    ----------
    metrics_df : pd.DataFrame
        DataFrame with classification results.
    root : Path
        Root path for image files.
    """
    if len(metrics_df) > 0 and NAPARI_AVAILABLE:
        print("Creating Interactive Napari Image Viewer...")
        viewer = NapariImageViewer(metrics_df, root)
        viewer.display()
        print("\nInteractive viewer ready. Use the controls above to:")
        print("   - Select sample and image")
        print("   - Choose display options")
        print("   - Open single images or complete z-stacks in napari")
        print("   - View 3D z-stacks with interactive navigation")
    elif not NAPARI_AVAILABLE:
        print("Napari not available. Install with: pip install napari[all]")
    else:
        print("No classification data available. Please run previous cells first.")

