"""File I/O utilities for TIFF images and configurations."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union
import numpy as np

# Handle imports that work both in package and Jupyter contexts
try:
    from ..config import ThresholdConfig
except ImportError:
    from config import ThresholdConfig

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

def find_brightfield_from_mcherry_path(img_path, mcherry_suffix:str = "_mCherry", brightfield_suffix:str = "_BF") -> Path | None:
    """Find brightfield (BF) image corresponding to input image."""
    img_path = Path(img_path)
    # Replace the filename with _BF version
    # Pattern: p2426_B02_z10_mCherry.tif -> p2426_B02_z10_BF.tif
    stem = img_path.name
    target_name = stem.replace(mcherry_suffix, brightfield_suffix)
    bf_path = img_path.parent / target_name
    
    if bf_path.exists():
        return bf_path
    else:
        raise NotImplementedError("Method needs to be updated")
    return None


def find_label_from_mcherry_path(img_path: Path, mcherry_suffix:str = "_mCherry", mask_suffix:str = "_Cells") -> Path | None:
    """Find the corresponding label/mask file for an image.
    
    Args:
        img_path: Path to the image file
        mcherry_suffix: Suffix to identify mCherry images (default: "_mCherry")
        mask_suffix: Suffix for label/mask files (default: "_Cells")
        
    Returns:
        Path to the label file, or None if not found
    """
    stem = img_path.name
    
    # If the input file is already a label file (ends with _Cells), return it as-is
    if stem.endswith(mask_suffix + ".tif"):
        return img_path
    
    # Strategy 1: Try exact suffix replacement with provided mcherry_suffix
    target_name = stem.replace(mcherry_suffix, mask_suffix)
    candidate = img_path.parent / target_name
    if candidate.exists():
        return candidate
    else:
        raise NotImplementedError("Method needs to be updated")
    
    # # Strategy 2: If "_w2" in name, try prefix before that + mask_suffix
    # # (common pattern: _w2.tif -> _Cells.tif)
    # if "_w2" in stem:
    #     # Handle both "_w2_" and "_w2." cases
    #     prefix = stem.split("_w2")[0]
    #     candidate = img_path.parent / (prefix + mask_suffix + ".tif")
    #     if candidate.exists():
    #         return candidate
    
    # # Strategy 3: Fallback - look for any "*Cells*.tif" in same folder 
    # # whose name shares prefix before first underscore
    # base_prefix = stem.split("_")[0]
    # for p in img_path.parent.glob(base_prefix + "*Cells*.tif"):
    #     return p
    
    # # Strategy 4: Global search in same directory for any Cells file
    # for p in img_path.parent.glob("*Cells*.tif"):
    #     return p
    
    return None

def find_activity_from_mcherry_path(img_path: Path, mcherry_suffix:str = "_mCherry", activity_suffix:str = "_activity", activity_dir:Path|None = None, must_exist: bool = False) -> Path | None:
    """Find the corresponding activity file for an image.
    
    Args:
        img_path: Path to the image file

    Returns:
        Path to the activity file, or None if not found
    """

    stem = img_path.name
    target_name = stem.replace(mcherry_suffix, activity_suffix)

    if activity_dir:
        candidate = activity_dir / target_name
    else:
        candidate = img_path.parent / target_name

    if candidate.exists() or not must_exist:
        return candidate
    
    return None    


def get_images_from_mcherry(img_path: Path, activity_dir: Path|None=None) -> Dict[str, Any]:
    # Load the image
    img = tiff.imread(str(img_path))

    lbl_path = find_label_from_mcherry_path(img_path)
    if lbl_path is not None and lbl_path.exists():
        lbl = tiff.imread(str(lbl_path))
        # print(f"Label found: {lbl_path.name}")
    else:
        lbl = None
        # print("No label image found for this image.")

    activity_path = find_activity_from_mcherry_path(img_path, activity_suffix="_activity_bin", activity_dir=activity_dir)
    # print(activity_path)
    if activity_path is not None and activity_path.exists():
        activity_labels = tiff.imread(str(activity_path))
        activity_is_bin = True
        # print(f"Activity classification found: {activity_path.name}")
    else:
        activity_path = find_activity_from_mcherry_path(img_path, activity_dir=activity_dir)
        # print(activity_path)
        activity_is_bin = False
        if activity_path is not None and activity_path.exists():
            activity_labels = tiff.imread(str(activity_path))
            # print(f"Activity classification found: {activity_path.name}")
        else:
            activity_labels = None
            # print("No activity classification found for this image.")

    bf_path = find_brightfield_from_mcherry_path(img_path)
    if bf_path is not None and bf_path.exists():
        brightfield = tiff.imread(str(bf_path))
        # print(f"Brightfield found: {bf_path.name}")
    else:
        brightfield = None
        # print("No brightfield image found for this image.")

    return {
        "img": img,
        "lbl": lbl,
        "activity_labels": activity_labels,
        "brightfield": brightfield,
        "activity_is_bin": activity_is_bin
    }



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

__all__ = ['list_tif_files', 'save_config', 'load_config', 'find_label_from_mcherry_path', 'ensure_2d']