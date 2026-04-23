# Napari visualization module for single-cell analysis
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff

try:
    import napari
    NAPARI_AVAILABLE = True
except ImportError:
    NAPARI_AVAILABLE = False

from threshold_activity_classifier.utils.io import find_label_from_mcherry_path, ensure_2d, get_images_from_mcherry
from threshold_activity_classifier.core.classifier import create_activity_labeled_image
from threshold_activity_classifier.utils.df_utils import filter_images_by_sample_id_and_z_index


class NapariImageViewer:
    """
    Napari-only viewer for 2D images and 3D z-stacks.
    """

    def __init__(self):
        self.current_viewer = None

    def open(self, img, lbl=None,
             activity_labels=None,
             brightfield=None,
             metrics_df=None, 
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
        brightfield : np.ndarray, optional
            Brightfield image for overlay (if available).
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
        
        # Add brightfield image if available
        if brightfield is not None:
            self.current_viewer.add_image(brightfield, name=f"{image_name}_brightfield", colormap="gray", opacity=0.5)
        
        self.current_viewer.window.resize(1200, 800)
        return self.current_viewer
    
    def open_3d(self, image_volume, label_volume=None, activity_volume=None, brightfield_volume=None,
                title="3D Volume", activity_is_bin=True,
                show_original=True, show_labels=True, show_activity=True, show_brightfield=False):
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
        brightfield_volume : np.ndarray, optional
            3D brightfield volume for overlay.
        title : str
            Window title.
        show_original : bool
            Whether to show original intensity volume.
        show_labels : bool
            Whether to show instance labels.
        show_activity : bool
            Whether to show activity classification.
        show_brightfield : bool
            Whether to show brightfield image.
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

        if brightfield_volume is not None:
            lo, hi = np.percentile(brightfield_volume, [0.1, 99.9])
            brightfield_volume = np.clip(brightfield_volume, lo, hi)

        if show_original:
            self.current_viewer.add_image(
                image_volume, name='Original Images',
                colormap='yellow', opacity=0.7, scale=(1, 1, 1)
            )
        
        if show_labels and label_volume is not None:
            self.current_viewer.add_labels(label_volume, name='Instance Labels', opacity=0.5)
        
        if show_activity and activity_volume is not None:
            self._add_activity_layer(activity_volume, 'Activity Classification', activity_is_bin=activity_is_bin)
        
        if show_brightfield and brightfield_volume is not None:
            self.current_viewer.add_image(brightfield_volume, name='Brightfield', colormap='gray', opacity=0.5)

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
    lbl_path = find_label_from_mcherry_path(img_path)
    
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
    
    sample_images = metrics_df[metrics_df['ID'] == sample_id]['image'].unique()
    
    if len(sample_images) == 0:
        print(f"No images found for sample {sample_id}")
        return []
    
    if z_indices is not None:
        sample_data = metrics_df[
            (metrics_df['ID'] == sample_id) & 
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

def _load_optional(path_val: str | Path | None, label: str) -> np.ndarray | None:
    """
    Load a TIFF file, returning ``None`` gracefully if the path is absent.

    Parameters
    ----------
    path_val : str, Path, or None
        Path to the TIFF file, or a value that evaluates as missing
        (``None``, ``NaN``, empty string).
    label : str
        Human-readable description used in the warning message when the
        file is not found.

    Returns
    -------
    np.ndarray or None
        Loaded image array, or ``None`` if the file is missing.
    """
    if path_val is None:
        return None
    try:
        if pd.isna(path_val):
            return None
    except (TypeError, ValueError):
        pass
    if not path_val:
        return None
    p = Path(str(path_val))
    if not p.exists():
        print(f"  Warning: {label} not found at {p}")
        return None
    return tiff.imread(str(p))

def create_napari_3d_from_summary_df(
    metrics_df: pd.DataFrame,
    sample_id: str,
    z_indices: list | None = None,
) -> NapariImageViewer | None:
    """
    Build 3D volumes from paths stored in ``metrics_df`` and open them in napari.

    Reads the ``mcherry_path``, ``bf_path``, ``label_path``, and
    ``activity_path`` columns for each row that belongs to ``sample_id``.
    Channels where every slice is missing are silently omitted; channels where
    only *some* slices are missing are filled with zero arrays of the same
    shape as the first valid slice.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        DataFrame produced by the activity classifier pipeline.  Must contain
        at least the columns ``ID``, ``z_index``, and ``mcherry_path``.
    sample_id : str
        Sample identifier used to filter ``metrics_df`` (matches the ``ID``
        column).
    z_indices : list of int, optional
        Restrict the volume to these z-indices.  ``None`` (default) includes
        all z-slices for the sample.

    Returns
    -------
    NapariImageViewer or None
        The open napari viewer, or ``None`` if no images could be loaded.
    """
    filtered_df = filter_images_by_sample_id_and_z_index(metrics_df, sample_id=sample_id)
    if z_indices is not None and 'z_index' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['z_index'].isin(z_indices)]
    filtered_df = filtered_df.sort_values('z_index')

    if len(filtered_df) == 0:
        print(f"No images found for sample {sample_id} with specified z-indices.")
        return None

    mcherry_stack: list[np.ndarray | None] = []
    bf_stack:      list[np.ndarray | None] = []
    label_stack:   list[np.ndarray | None] = []
    activity_stack: list[np.ndarray | None] = []

    for _, row in filtered_df.iterrows():
        mcherry_stack.append(_load_optional(row.get('mcherry_path'),  "mCherry image"))
        bf_stack.append(     _load_optional(row.get('bf_path'),       "Brightfield image"))
        label_stack.append(  _load_optional(row.get('label_path'),    "Label image"))
        activity_stack.append(_load_optional(row.get('activity_path'), "Activity labels"))

    def _stack_channel(slices: list[np.ndarray | None], name: str) -> np.ndarray | None:
        """Stack a list of 2-D arrays, filling missing slices with zeros."""
        valid = [s for s in slices if s is not None]
        if not valid:
            return None
        if len(valid) < len(slices):
            ref_shape = valid[0].shape
            slices = [s if s is not None else np.zeros(ref_shape, dtype=valid[0].dtype) for s in slices]
            print(f"  Warning: {len(slices) - len(valid)} missing {name} slice(s) filled with zeros.")
        return np.stack(slices, axis=0)

    mcherry_volume  = _stack_channel(mcherry_stack,  "mCherry")
    bf_volume       = _stack_channel(bf_stack,        "brightfield")
    label_volume    = _stack_channel(label_stack,     "label")
    activity_volume = _stack_channel(activity_stack,  "activity")

    if mcherry_volume is None:
        print(f"No mCherry images could be loaded for sample {sample_id}.")
        return None

    # create_sample_zstack_statistics(filtered_df, sample_id)

    try:
        viewer = NapariImageViewer()
        viewer.open_3d(
            image_volume=mcherry_volume,
            label_volume=label_volume,
            activity_volume=activity_volume,
            brightfield_volume=bf_volume,
            title=f"3D Z-Stack: Sample {sample_id}",
        )
        return viewer

    except Exception as e:
        print(f"Error creating napari viewer for sample {sample_id}: {e}")
        return None


def visualize_sample_zstack_napari(sample_id, metrics_df, root_path, activity_dir, label_dir=None,
                                   show_original=True, show_labels=True, show_activity=True, show_brightfield=True,
                                   ) -> NapariImageViewer | None:
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
    label_dir : Path, optional
        Directory to search for label files. If None, looks in the same folder as each image.
    show_original : bool
        Whether to show the original intensity images.
    show_labels : bool
        Whether to show the original instance labels.
    show_activity : bool
        Whether to show the activity-coded labels.
    show_brightfield : bool
        Whether to show the brightfield image.
        
    Returns
    -------
    NapariImageViewer or None
        The napari viewer instance, or None on error.
    """
    if not NAPARI_AVAILABLE:
        print("Napari is not available. Install with: pip install napari[all]")
        return None

    filtered_images = filter_images_by_sample_id_and_z_index(metrics_df, sample_id=sample_id)
    if len(filtered_images) == 0:
        print(f"No images found for sample {sample_id}")
        return None

    missing_labels = []
    for image_name in filtered_images['image']:
        image_files = list(Path(root_path).rglob(image_name))
        if not image_files:
            continue
        img_path = image_files[0]
        lbl_path = find_label_from_mcherry_path(img_path, label_dir=label_dir)
        if lbl_path is None or not lbl_path.exists():
            missing_labels.append(image_name)

    if missing_labels:
        print(f"Cannot open 3D viewer: label files missing for {len(missing_labels)} image(s):")
        for name in missing_labels:
            print(f"  - {name}")
        return None

    image_stack = []
    label_stack = []
    activity_stack = []
    brightfield_stack = []
    activity_is_bin = True

    for image_name in filtered_images['image']:
        try:
            image_files = list(Path(root_path).rglob(image_name))
            if not image_files:
                print(f"Warning: Image {image_name} not found, skipping")
                continue

            img_path = image_files[0]
            img_dict = get_images_from_mcherry(
                img_path,
                activity_dir=activity_dir,
                label_dir=label_dir,
            )

            img = img_dict.get('img')
            lbl = img_dict.get('lbl')
            activity_labels = img_dict.get('activity_labels')
            brightfield = img_dict.get('brightfield')

            if activity_labels is None:
                print(f"Activity labels not found for {image_name}, creating from metrics_df...")
                activity_is_bin = False
                image_classification = metrics_df[metrics_df['image'] == image_name].copy()
                if len(image_classification) == 0:
                    print(f"Warning: No classification data found for {image_name}, skipping")
                    continue
                try:
                    activity_labels = create_activity_labeled_image(lbl, image_classification)
                except Exception as e:
                    print(f"Warning: Could not create activity labels for {image_name}: {e}")
                    activity_labels = None

            img = ensure_2d(img).astype(float)
            lbl = ensure_2d(lbl).astype(np.int32) if lbl is not None else None
            brightfield = ensure_2d(brightfield).astype(float) if brightfield is not None else None
            activity_labels = ensure_2d(activity_labels).astype(np.int32) if activity_labels is not None else None

            image_stack.append(img)
            if lbl is not None:
                label_stack.append(lbl)
            if activity_labels is not None:
                activity_stack.append(activity_labels)
            if brightfield is not None:
                brightfield_stack.append(brightfield)
        except Exception as e:
            print(f"Error processing {image_name}: {e}")
            continue

    if len(image_stack) == 0:
        print("No valid images were loaded")
        return None

    image_volume = np.stack(image_stack, axis=0)
    label_volume = np.stack(label_stack, axis=0) if label_stack else None
    activity_volume = np.stack(activity_stack, axis=0) if activity_stack else None
    brightfield_volume = np.stack(brightfield_stack, axis=0) if brightfield_stack else None

    print(f"Created 3D volumes with shape: {image_volume.shape}")

    viewer = NapariImageViewer()
    viewer.open_3d(
        image_volume=image_volume,
        label_volume=label_volume,
        activity_volume=activity_volume,
        brightfield_volume=brightfield_volume,
        title=f"3D Z-Stack: Sample {sample_id}",
        show_original=show_original,
        show_labels=show_labels,
        show_activity=show_activity,
        show_brightfield=show_brightfield,
        activity_is_bin=activity_is_bin,
    )
    return viewer
