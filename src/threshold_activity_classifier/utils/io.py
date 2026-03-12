"""File I/O utilities for TIFF images and configurations."""
from __future__ import annotations

from pathlib import Path
from typing import List, Union
import numpy as np

from threshold_activity_classifier.config import ThresholdConfig

import tifffile as tiff



def list_tif_files(directory: Union[str, Path], pattern: str = '*.tif', recursive: bool = True) -> List[Path]:
    """List all TIFF files in a directory.
    
    Args:
        directory: Directory to search
        recursive: Whether to search subdirectories
        
    Returns:
        List of TIFF file paths
    """
    directory = Path(directory)
    if not directory.exists():
        raise ValueError(f"Directory does not exist: {directory}")
    
    pattern = f'**/{pattern}' if recursive else pattern
    
    return list(directory.glob(pattern))


def save_config(config: ThresholdConfig, path: Union[str, Path]) -> None:
    """Save configuration to JSON file.
    
    Args:
        config: Configuration to save
        path: Output file path
    """
    config.save(path)


def load_config(path: Union[str, Path]) -> ThresholdConfig:
    """Load configuration from JSON file.
    
    Args:
        path: Input file path
        
    Returns:
        Loaded configuration
    """
    return ThresholdConfig.load(path)


def find_label_for_image(img_path: Path, img_suffix:str = "_w2", mask_suffix:str = "_Cells") -> Path | None:
    """Find the corresponding label/mask file for an image.
    
    Args:
        img_path: Path to the image file
        
    Returns:
        Path to the label file, or None if not found
        
    Raises:
        NotImplementedError: For special cases that need custom logic
    """
    # Try exact replacement of "_w2" -> "_Cells"
    stem = img_path.name

    target_name = stem.replace(img_suffix, mask_suffix)

    candidate = img_path.parent / target_name
    if candidate.exists():
        return candidate
    else:
        raise NotImplementedError("Method needs to be updated")
        
    # Special case: if "_w2_" in name, try prefix before that + mask_suffix
    if "_w2_" in stem:
        prefix = stem.split("_w2_")[0]
        candidate = img_path.parent / (prefix + mask_suffix)
        if candidate.exists():
            return candidate
        # Fallback: any file starting with prefix and containing "Cells"
        for p in img_path.parent.glob(prefix + "*Cells*.tif"):
            return p
            
    # Generic fallback: look for any "*Cells*.tif" in same folder 
    # whose name shares prefix before first underscore
    base_prefix = stem.split("_")[0]
    for p in img_path.parent.glob(base_prefix + "*Cells*.tif"):
        return p
        
    # Global search
    for p in img_path.parent.rglob("*Cells*.tif"):
        return p
        
    return None

def find_activity_for_image(img_path: Path, img_suffix:str = "_w2", activity_suffix:str = "_activity", activity_dir:Path|None = None, must_exist: bool = False) -> Path | None:
    """Find the corresponding activity file for an image.
    
    Args:
        img_path: Path to the image file

    Returns:
        Path to the activity file, or None if not found
    """

    stem = img_path.name
    target_name = stem.replace(img_suffix, activity_suffix)

    if activity_dir:
        candidate = activity_dir / target_name
    else:
        candidate = img_path.parent / target_name

    if candidate.exists() or not must_exist:
        return candidate
    
    return None    

def get_img_lbl_activity(img_path: Path, activity_dir: Path):
    # Load the image
    img = tiff.imread(str(img_path))
                    
    # Try to find and load the corresponding label image
    metrics_df = None

    lbl_path = find_label_for_image(img_path)
    if lbl_path is not None and lbl_path.exists():
        lbl = tiff.imread(str(lbl_path))
        print(f"Label found: {lbl_path.name}")
    else:
        lbl = None
        print("No label image found for this image.")

    activity_path = find_activity_for_image(img_path, activity_suffix="_activity_bin", activity_dir=activity_dir)
    # print(activity_path)
    if activity_path is not None and activity_path.exists():
        activity_labels = tiff.imread(str(activity_path))
        activity_is_bin = True
        print(f"Activity classification found: {activity_path.name}")
    else:
        activity_path = find_activity_for_image(img_path, activity_dir=activity_dir)
        # print(activity_path)
        activity_is_bin = False
        if activity_path is not None and activity_path.exists():
            activity_labels = tiff.imread(str(activity_path))
            print(f"Activity classification found: {activity_path.name}")
        else:
            activity_labels = None
            print("No activity classification found for this image.")

    return img, lbl, activity_labels



def ensure_2d(arr: np.ndarray) -> np.ndarray:
    """Ensure array is 2D by taking the first plane if multi-dimensional.
    
    Args:
        arr: Input array (2D, 3D, or higher dimensional)
        
    Returns:
        2D array
    """
    if arr.ndim == 2:
        return arr
    return arr[0]

__all__ = ['list_tif_files', 'save_config', 'load_config', 'find_label_for_image', 'ensure_2d']