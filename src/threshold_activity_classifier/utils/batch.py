"""Batch processing utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from tifffile import TiffFile
import tifffile

from config import ThresholdConfig
from core import ThresholdClassifier
from .validation import validate_folder


def batch_process(
    input_folder: Union[str, Path],
    output_folder: Union[str, Path],
    config: ThresholdConfig,
    save_masks: bool = True,
    save_overlays: bool = False
) -> pd.DataFrame:
    """Batch process a folder of images.
    
    Args:
        input_folder: Input directory with TIFF files
        output_folder: Output directory for results
        config: Processing configuration
        save_masks: Whether to save binary masks
        save_overlays: Whether to save overlay images
        
    Returns:
        DataFrame with processing results
    """
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Get validation results
    results_df = validate_folder(input_folder, config)
    
    # Save processing results
    results_path = output_folder / "processing_results.csv"
    results_df.to_csv(results_path, index=False)
    
    if save_masks or save_overlays:
        classifier = ThresholdClassifier(config)
        masks_dir = output_folder / "masks"
        overlays_dir = output_folder / "overlays"
        
        if save_masks:
            masks_dir.mkdir(exist_ok=True)
        if save_overlays:
            overlays_dir.mkdir(exist_ok=True)
        
        # Process successful files
        successful_files = results_df[results_df['processing_success']]
        
        for _, row in successful_files.iterrows():
            try:
                # Load image
                image_path = Path(row['path'])
                with TiffFile(image_path) as tif:
                    image = tif.asarray()
                
                # Process
                mask, _ = classifier.process(image)
                
                # Save mask
                if save_masks:
                    mask_path = masks_dir / f"{image_path.stem}_mask.tif"
                    tifffile.imwrite(mask_path, mask.astype(np.uint8))
                
                # Save overlay
                if save_overlays:
                    # Create simple overlay (original + colored mask)
                    overlay = np.zeros((*image.shape, 3), dtype=np.uint8)
                    # Normalize original image to 0-255
                    img_norm = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
                    overlay[..., 0] = img_norm  # Red channel
                    overlay[..., 1] = img_norm  # Green channel 
                    overlay[..., 2] = img_norm  # Blue channel
                    
                    # Add mask in red
                    overlay[mask, 0] = 255
                    overlay[mask, 1] = 0
                    overlay[mask, 2] = 0
                    
                    overlay_path = overlays_dir / f"{image_path.stem}_overlay.tif"
                    tifffile.imwrite(overlay_path, overlay)
                    
            except Exception as e:
                print(f"Failed to save outputs for {row['filename']}: {e}")
    
    return results_df


__all__ = ['batch_process']
