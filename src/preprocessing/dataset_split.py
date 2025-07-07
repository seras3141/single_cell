"""
Dataset splitting utilities for single cell analysis.

This module provides functions to split image and mask datasets into training and test sets,
ensuring that images from the same group (well, position, timepoint) stay together.
"""

import os
import json
import re
from glob import glob
from pathlib import Path
import random
import logging
import shutil
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Union, NamedTuple, Set
from src.utils.file_utils import AbstractFileHandler, DefaultFileHandler, BF_IF_FileHandler

logger = logging.getLogger(__name__)


class DatasetSplit(NamedTuple):
    """Container for dataset split results."""
    train_images: List[str]
    train_masks: List[str]
    test_images: List[str]
    test_masks: List[str]


def get_groups_from_filenames(file_map, file_handler : DefaultFileHandler) -> Dict[str, List[str]]:
    """
    Group files based on a pattern in their filenames.
    
    Args:
        file_map: Dictionary mapping original file names to output names
        file_handler: File renamer/handler for extracting group info
        
    Returns:
        Dictionary mapping group identifiers to lists of file paths
    """
    groups = defaultdict(list)
    
    for filepath, out_path in file_map.items():
        group_id = file_handler.extract_group_id(out_path)

        # Use filename without extension as fallback
        groups[group_id].append(filepath)
            
    return dict(groups)

def copy_file(
    src_file: Union[str, Path],
    dest_file: Union[str, Path],
) -> None:
    """
    Copy a file from src_file to dest_file with metadata preservation.

    Args:
        src_file: Source file to copy
        dest_file: Destination file path
    """
    src = Path(src_file).resolve()
    dst = Path(dest_file).resolve()

    # Ensure the parent directory exists
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() or dst.is_symlink():
        dst.unlink()  # Remove existing file

    try:
        dst.symlink_to(src, target_is_directory=False)
    except (OSError, NotImplementedError):
        # Use shutil.copy2 which preserves metadata instead of symlinks
        # This works on all platforms without admin privileges
        shutil.copy2(src, dst)


def split_dataset(
    images: List[str],
    masks: List[str],
    test_size: float = 0.2,
    random_state: int = 42,
    file_handler: AbstractFileHandler = DefaultFileHandler,
    output_dir: Union[str, Path] = None,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Split a dataset of images and masks into train and test sets, keeping groups together.

    Args:
        images: List of image file paths
        masks: List of mask file paths
        test_size: Fraction of data to use for testing (0-1)
        random_state: Random seed for reproducibility
        file_handler: File renamer/handler for extracting group info
        copy_files: If True, copy files to output directory; if False, just return paths

    Returns:
        Tuple containing (train_images, train_masks, test_images, test_masks)
    """
    # Set random seed for reproducibility
    random.seed(random_state)

    # Map images and masks to desired output names using the file handler
    images_map = {image_name : file_handler.rename_image(image_name) for image_name in images}
    masks_map = {mask_name : file_handler.rename_mask(mask_name) for mask_name in masks}
    
    # Group images by position/group_id
    image_groups = get_groups_from_filenames(images_map, file_handler)
    mask_groups = get_groups_from_filenames(masks_map, file_handler)
    
    # Get common groups
    common_groups = set(image_groups.keys()) & set(mask_groups.keys())
    
    if not common_groups:
        raise ValueError("No matching image-mask groups found")
    
    # Split groups into train/test
    groups = list(common_groups)
    n_test = max(1, int(len(groups) * test_size))
    test_groups = set(random.sample(groups, n_test))
    
    # Separate images and masks
    train_images, train_masks = [], []
    test_images, test_masks = [], []
    
    for group in common_groups:
        if group in test_groups:
            test_images.extend(image_groups[group])
            test_masks.extend(mask_groups[group])
        else:
            train_images.extend(image_groups[group])
            train_masks.extend(mask_groups[group])
    
    logger.info(f"Split dataset: {len(train_images)} training images, {len(test_images)} test images")
    logger.info(f"Train groups: {sorted(common_groups - test_groups)}")
    logger.info(f"Test groups: {sorted(test_groups)}")

    if output_dir:
        # Create output directories
        train_dir = output_dir / 'train'
        test_dir = output_dir / 'test'
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy files to their respective directories
        for img, mask in zip(train_images, train_masks):
            copy_file(img, train_dir/images_map[img])
            copy_file(mask, train_dir/masks_map[mask])
            
        for img, mask in zip(test_images, test_masks):
            copy_file(img, test_dir/images_map[img])
            copy_file(mask, test_dir/masks_map[mask])

        # Log the copied files            
        logger.info(f"Files copied to {train_dir} and {test_dir}")

    return train_images, train_masks, test_images, test_masks


def train_test_split_directory(
    data_dir: Union[str, Path],
    output_dir: Union[str, Path],
    test_size: float = 0.2,
    random_state: int = 42,
    image_pattern: str = "*_w1_*.tif",
    mask_pattern: str = "Cells_*.tif",
    file_handler: Optional[AbstractFileHandler] = None,
) -> Dict[str, List[str]]:
    """
    Split data in a directory into train and test sets and organize into subdirectories.
    
    Args:
        data_dir: Directory containing images and masks
        output_dir: Directory to save the split data
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        image_pattern: Glob pattern for image files
        mask_pattern: Glob pattern for mask files
        
    Returns:
        Dictionary with keys 'train_images', 'train_masks', 'test_images', 'test_masks'
        containing the paths of the files in each split
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    # Find all images and masks
    images = sorted(glob(str(data_dir / "**" / image_pattern), recursive=True))
    
    # Check if masks are in a subdirectory
    masks = sorted(glob(str(data_dir / "**" / mask_pattern), recursive=True))
    
    logger.info(f"Found {len(images)} images and {len(masks)} masks in {data_dir}")

    # Use provided file handler or default to BF_IF_FileHandler
    if file_handler is None:
        file_handler = BF_IF_FileHandler()
    
    
    # Split the dataset
    train_images, train_masks, test_images, test_masks = split_dataset(
        images, masks, test_size, random_state, file_handler, output_dir
    )
    
    result = {
        'train_images': train_images,
        'train_masks': train_masks,
        'test_images': test_images,
        'test_masks': test_masks
    }    
    
    return result


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Split dataset into train and test sets")
    parser.add_argument("data_dir", help="Directory containing the dataset")
    parser.add_argument("output_dir", help="Directory to save the split dataset")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data to use for testing")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--image-pattern", default="*_w1_*.tif", help="Glob pattern for image files")
    parser.add_argument("--mask-pattern", default="Cells_*.tif", help="Glob pattern for mask files")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
    # Run the split
    result = train_test_split_directory(
        args.data_dir,
        args.output_dir,
        args.test_size,
        args.random_seed,
        args.image_pattern,
        args.mask_pattern,
        file_handler=BF_IF_FileHandler()  # Default for command-line usage
    )
    
    # Print summary
    print(f"Train set: {len(result['train_images'])} images")
    print(f"Test set: {len(result['test_images'])} images")
