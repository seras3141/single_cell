"""Validation utilities for folder processing."""
from __future__ import annotations

from pathlib import Path
from typing import Union, Optional

import numpy as np
import pandas as pd
from tifffile import TiffFile

from config import ThresholdConfig
from core import ThresholdClassifier
from .io import list_tif_files


def validate_folder(
    folder_path: Union[str, Path],
    config: ThresholdConfig,
    output_path: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """Validate a folder of TIFF images using threshold classifier.
    
    Args:
        folder_path: Path to folder containing TIFF files
        config: Threshold configuration to use
        output_path: Optional path to save results CSV
        
    Returns:
        DataFrame with validation results
    """
    folder_path = Path(folder_path)
    if not folder_path.exists():
        raise ValueError(f"Folder does not exist: {folder_path}")
    
    # Find all TIFF files
    tiff_files = list_tif_files(folder_path)
    if not tiff_files:
        raise ValueError(f"No TIFF files found in {folder_path}")
    
    # Initialize classifier
    classifier = ThresholdClassifier(config)
    
    # Process each file
    results = []
    for tiff_path in tiff_files:
        try:
            # Load image
            with TiffFile(tiff_path) as tif:
                image = tif.asarray()
            
            # Process with classifier
            mask, threshold = classifier.process(image)
            
            # Compute metrics
            total_pixels = image.size
            active_pixels = np.sum(mask)
            active_fraction = active_pixels / total_pixels if total_pixels > 0 else 0.0
            
            # Compute intensity statistics
            mean_intensity = float(np.mean(image))
            std_intensity = float(np.std(image))
            min_intensity = float(np.min(image))
            max_intensity = float(np.max(image))
            
            # Active region statistics
            if active_pixels > 0:
                active_mean = float(np.mean(image[mask]))
                active_std = float(np.std(image[mask]))
            else:
                active_mean = 0.0
                active_std = 0.0
            
            results.append({
                'filename': tiff_path.name,
                'path': str(tiff_path),
                'image_shape': str(image.shape),
                'threshold_method': config.method,
                'threshold_value': np.asarray(threshold, dtype=float).item() if np.isscalar(threshold) else 'adaptive',
                'total_pixels': total_pixels,
                'active_pixels': active_pixels,
                'active_fraction': active_fraction,
                'mean_intensity': mean_intensity,
                'std_intensity': std_intensity,
                'min_intensity': min_intensity,
                'max_intensity': max_intensity,
                'active_mean_intensity': active_mean,
                'active_std_intensity': active_std,
                'processing_success': True,
                'error_message': None
            })
            
        except Exception as e:
            results.append({
                'filename': tiff_path.name,
                'path': str(tiff_path),
                'image_shape': None,
                'threshold_method': config.method,
                'threshold_value': None,
                'total_pixels': None,
                'active_pixels': None,
                'active_fraction': None,
                'mean_intensity': None,
                'std_intensity': None,
                'min_intensity': None,
                'max_intensity': None,
                'active_mean_intensity': None,
                'active_std_intensity': None,
                'processing_success': False,
                'error_message': str(e)
            })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save if requested
    if output_path is not None:
        output_path = Path(output_path)
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
    
    return df


__all__ = ['validate_folder']
