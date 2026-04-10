"""Instance metrics extraction for threshold activity classifier.

This module provides a class for extracting various metrics from labeled instances
in microscopy images, including intensity statistics and percentiles.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Tuple, Union
from tqdm import tqdm
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import tifffile as tiff
from skimage import measure
import logging

# Handle imports that work both in package and Jupyter contexts
from threshold_activity_classifier.utils.io import find_label_from_mcherry_path, ensure_2d

class InstanceMetricsExtractor:
    """Extract metrics from labeled instances in images.
    
    This class processes images and their corresponding label masks to compute
    various intensity-based metrics for each labeled instance (cell, object, etc.).
    
    Attributes:
        percentiles: List of percentile values to compute (default: [95, 90, 75, 50])
        file_handler: File handler for managing file operations
    """
    
    def __init__(
        self,
        percentiles: Optional[List[int]] = None,
    ):
        """Initialize the metrics extractor.
        
        Args:
            percentiles: List of percentile values to compute (0-100).
                        Default: [95, 90, 75, 50]
            file_handler: Custom file handler. Default: StandardFileHandler()
        """
        self.percentiles = percentiles if percentiles is not None else [95, 90, 75, 50]
        

    def compute_basic_metrics(
        self,
        label_array: np.ndarray,
        intensity_image: np.ndarray
    ) -> pd.DataFrame:
        """Compute basic intensity metrics for each labeled region.
        
        Args:
            label_array: Labeled array where each unique value represents an instance
            intensity_image: Intensity image to measure
            
        Returns:
            DataFrame with basic metrics (label, area, mean/max/min intensity)
        """
        props = measure.regionprops_table(
            label_array,
            intensity_image=intensity_image,
            properties=('label', 'area', 'mean_intensity', 'max_intensity', 'min_intensity')
        )
        return pd.DataFrame(props)

    def compute_sum_intensity(
        self,
        df: pd.DataFrame,
        label_array: np.ndarray,
        intensity_image: np.ndarray
    ) -> pd.DataFrame:
        """Compute sum intensity for each labeled region using bincount.
        
        Args:
            df: DataFrame with basic metrics (must have 'label' column)
            label_array: Labeled array
            intensity_image: Intensity image
            
        Returns:
            DataFrame with added 'sum_intensity' column
        """
        labels_in_df = df['label'].to_numpy()
        
        # Sum intensity per label via bincount (single pass)
        # Background label 0 included; we'll pick only needed labels after
        sums = np.bincount(
            label_array.ravel(),
            weights=intensity_image.ravel(),
            minlength=labels_in_df.max() + 1
        )
        df['sum_intensity'] = sums[labels_in_df]
        
        return df

    def compute_percentiles(
        self,
        df: pd.DataFrame,
        label_array: np.ndarray,
        intensity_image: np.ndarray
    ) -> pd.DataFrame:
        """Compute percentile values for each labeled region.
        
        Args:
            df: DataFrame with basic metrics (must have 'label' column)
            label_array: Labeled array
            intensity_image: Intensity image
            
        Returns:
            DataFrame with added percentile columns
        """
        labels_in_df = df['label'].to_numpy()
        
        # Compute percentiles per label using vectorized groupby.quantile
        mask = label_array > 0
        
        if np.any(mask):
            lab = label_array[mask].ravel()
            val = intensity_image[mask].ravel()

            s = pd.Series(val, index=lab)  # index=label ids
            q = s.groupby(level=0).quantile(np.array(self.percentiles) / 100.0)  # MultiIndex (label, q)
            q = q.unstack(level=1)  # columns are percentiles as floats

            # Align to df labels and insert columns
            q = q.reindex(labels_in_df)  # ensure same order as df
            for perc in self.percentiles:
                df[f'percentile_{perc}'] = q[perc / 100.0].to_numpy()
        else:
            # No foreground
            for perc in self.percentiles:
                df[f'percentile_{perc}'] = 0.0
        
        return df

    def process_single_image(
        self,
        img_path: str | Path,
        lbl_path: str | Path | None,
    ) -> Tuple[Optional[str], Optional[pd.DataFrame]]:
        """Process a single image and extract metrics for all instances.
        
        Args:
            img_path: Path to the image file
            
        Returns:
            Tuple of (warning_message, metrics_dataframe)
            - warning_message: String if there was an issue, None otherwise
            - metrics_dataframe: DataFrame with metrics, or None if processing failed
        """
        img_path = Path(img_path)
        if lbl_path is None:
            return f"Warning: No label path provided for image {img_path.name}", None
        
        # --- I/O (lean dtypes, avoid extra copies) ---
        img = tiff.imread(str(img_path))
        lbl = tiff.imread(str(lbl_path))

        img = ensure_2d(img)
        lbl = ensure_2d(lbl)

        # Cast with minimal copies
        if img.dtype != np.float32:
            img = img.astype(np.float32, copy=False)
        if lbl.dtype != np.int32:
            lbl = lbl.astype(np.int32, copy=False)

        if img.shape != lbl.shape:
            return (f"warning: shape mismatch ({img.shape} vs {lbl.shape}) for "
                    f"{img_path.name} / {Path(lbl_path).name}, skipping"), None

        # --- Compute metrics ---
        df = self.compute_basic_metrics(lbl, img)

        if df.empty:
            # Nothing to do (no labels)
            df['image'] = img_path.name
            return None, df

        # Add sum intensity
        df = self.compute_sum_intensity(df, lbl, img)
        
        # Add percentiles
        df = self.compute_percentiles(df, lbl, img)

        df['image'] = img_path.name
        return None, df

    def process_batch_images(
            self,
            img_paths: List[Union[str, Path]],
            lbl_paths: List[Union[str, Path]] | None = None,
            show_progress: bool = True,
            n_jobs: int = -1,
    ) -> pd.DataFrame:
        """Process a batch of images and extract metrics for all instances.
        
        Args:
            img_paths: List of paths to image files
            lbl_paths: Optional list of paths to label files (must correspond to img_paths)
            show_progress: Whether to show progress bar
            n_jobs: Number of parallel jobs (-1 uses all available CPUs)
        returns:
            DataFrame with metrics for all instances from all images
        """

        if lbl_paths is None:
            lbl_paths = [find_label_from_mcherry_path(Path(p)) for p in img_paths]  # type: ignore
        if lbl_paths is None or len(lbl_paths) != len(img_paths):
            raise ValueError("Label paths must be provided and correspond to image paths")

        pairs = tqdm(zip(img_paths, lbl_paths), total=len(img_paths), disable=not show_progress)
        results = Parallel(n_jobs=n_jobs)(
            delayed(self.process_single_image)(img_path, lbl_path)
            for img_path, lbl_path in pairs
        )

        records = []
        for warn, out in results:
            if warn is not None:
                print(warn)
            if out is not None:
                records.append(out)

        return pd.concat(records, ignore_index=True) if records else pd.DataFrame()

    def process_directory(
        self,
        directory: Union[str, Path],
        pattern: str = "*.tif",
        show_progress: bool = True,
        verbose: bool = True
    ) -> pd.DataFrame:
        """Process all images in a directory matching a pattern.
        
        Args:
            directory: Directory containing images
            pattern: Glob pattern for finding images (default: "*.tif")
            show_progress: Whether to show progress bar
            verbose: Whether to print warnings
            
        Returns:
            DataFrame with metrics for all instances from all images
        """
        directory = Path(directory)
        image_paths: List[Union[str, Path]] = list(sorted(directory.glob(pattern)))
        
        if not image_paths:
            logging.warning(f"Warning: No images found matching pattern '{pattern}' in {directory}")
            return pd.DataFrame()
        
        return self.process_batch_images(image_paths, show_progress=show_progress, n_jobs=-1)


__all__ = [
    'InstanceMetricsExtractor',
]
