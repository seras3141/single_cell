"""
Dataset splitting utilities for single cell analysis.

This module provides functions to split image and mask datasets into training and test sets,
ensuring that images from the same group (well, position, timepoint) stay together.
"""

import json
from pathlib import Path
import random
import logging
import shutil
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Union
from src.utils.file_utils import AbstractFileHandler, DefaultFileHandler, BF_IF_FileHandler
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def get_groups_from_filenames(renamed_tuples, file_handler: AbstractFileHandler) -> Dict[str, List[str]]:
    """
    Group files based on a pattern in their filenames.
    
    Args:
        renamed_tuples: List of tuples (original_file, renamed_file)
        file_handler: File renamer/handler for extracting group info
        
    Returns:
        Dictionary mapping group identifiers to lists of tuples (original_file, renamed_file)
    """
    groups = defaultdict(list)

    for src, dst in renamed_tuples:
        group_id = file_handler.extract_group_id(dst)

        groups[group_id].append((src, dst))
                
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
    dst = Path(dest_file)

    # Ensure the parent directory exists
    dst.parent.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        raise FileNotFoundError(f"Source file '{src}' does not exist")

    if dst.exists() or dst.is_symlink():
        dst.unlink()  # Remove existing file

    try:
        dst.symlink_to(src, target_is_directory=False)
    except (OSError, NotImplementedError):
        # Use shutil.copy2 which preserves metadata instead of symlinks
        # This works on all platforms without admin privileges
        shutil.copy2(src, dst)

def copy_without_split(
        image_tuple : List[Tuple[str, str]],
        mask_tuple : List[Tuple[str, str]],
        output_dir: Union[str, Path],
):
    """
    Copy image and mask files without splitting into train/test sets.

    Args:
        image_tuple: List of tuples (src_path, dest_path) for image files
        mask_tuple: List of tuples (src_path, dest_path) for mask files
        output_dir: Directory to copy the files to
    """
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for src, dst in image_tuple:
        copy_file(src, output_dir / dst)

    for src, dst in mask_tuple:
        copy_file(src, output_dir / dst)

def copy_with_split(
        train_image_tuple : List[Tuple[str, str]],
        train_mask_tuple : List[Tuple[str, str]],
        test_image_tuple : List[Tuple[str, str]],
        test_mask_tuple : List[Tuple[str, str]],
        output_dir: Union[str, Path],
):
    """
    Copy image and mask files into train and test subdirectories.

    Args:
        train_image_tuple: List of tuples (src_path, dest_path) for training image files
        train_mask_tuple: List of tuples (src_path, dest_path) for training mask files
        test_image_tuple: List of tuples (src_path, dest_path) for test image files
        test_mask_tuple: List of tuples (src_path, dest_path) for test mask files
        output_dir: Directory to copy the files to
    """
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dir = output_dir / 'train'
    test_dir = output_dir / 'test'
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # TODO : Handle masks separately into mask subdirs
    for src, dst in train_image_tuple:
        copy_file(src, train_dir / dst)

    for src, dst in train_mask_tuple:
        copy_file(src, train_dir / dst)

    for src, dst in test_image_tuple:
        copy_file(src, test_dir / dst)

    for src, dst in test_mask_tuple:
        copy_file(src, test_dir / dst)

def split_dataset(
    images: List[str],
    masks: Optional[List[str]] = [],
    test_size: float = 0.2,
    random_state: int = 42,
    file_handler: DefaultFileHandler = DefaultFileHandler(),
    output_dir: Optional[Union[str, Path]] = None,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Split a dataset of images and masks into train and test sets, keeping groups together.

    Args:
        images: List of image file paths
        masks: List of mask file paths
        test_size: Fraction of data to use for testing (0-1)
        random_state: Random seed for reproducibility
        file_handler: File renamer/handler for extracting group info
        output_dir: Directory to save the split files. If None, files are not copied.
    Returns:
        Tuple containing (train_images, train_masks, test_images, test_masks)
    """

    if file_handler is None:
        raise ValueError("No file handler provided, images may not be grouped correctly")

    if masks and len(images) != len(masks):
        raise ValueError("Number of images and masks must be the same", f"Images: {len(images)} != Masks: {len(masks)}")

    # Set random seed for reproducibility
    random.seed(random_state)

    if test_size <= 0 or test_size >= 1:
        logger.info("Test size is 0 or >=1, no splitting will be performed.")

        # Copy files to output directory without splitting
        image_tuples = [(src, file_handler.rename_image(src)) for src in images]
        mask_tuples = [(src, file_handler.rename_mask(src)) for src in masks] if masks else []
        if output_dir:
            copy_without_split(image_tuples, mask_tuples, output_dir)

        image_files = [dst for src, dst in image_tuples]
        mask_files = [dst for src, dst in mask_tuples]

        if test_size <= 0:
            logger.info("All data assigned to training set.")
            return image_files, mask_files, [], []
        else:
            logger.info("All data assigned to test set.")
            return [], [], image_files, mask_files

    else:
        logger.info(f"Splitting dataset with test size = {test_size}")

        # Rename files and create tuples : src, dst (base name only)
        image_tuples = [(src, file_handler.rename_image(src)) for src in images]
        mask_tuples = [(src, file_handler.rename_mask(src)) for src in masks] if masks else []

        # Group files based on extracted group IDs
        grouped_image_tuples = get_groups_from_filenames(image_tuples, file_handler)
        grouped_mask_tuples = get_groups_from_filenames(mask_tuples, file_handler) if masks else {}

        # Check if both images and masks have the same groups
        image_group_keys = set(grouped_image_tuples.keys())
        mask_group_keys = set(grouped_mask_tuples.keys())
        if masks and image_group_keys != mask_group_keys:
            raise ValueError(f"Image and mask groups do not match: {image_group_keys ^ mask_group_keys}")
        
        # split based on image_group_keys
        groups = list(image_group_keys)
        n_test = max(1, int(len(groups) * test_size))
        test_groups = set(random.sample(groups, n_test))

        # Separate images and masks
        train_images, train_masks = [], []
        test_images, test_masks = [], []        

        for group in image_group_keys:
            if group in test_groups:
                test_images.extend(grouped_image_tuples[group])
                if masks:
                    test_masks.extend(grouped_mask_tuples[group])
            else:
                train_images.extend(grouped_image_tuples[group])
                if masks:
                    train_masks.extend(grouped_mask_tuples[group])

        if output_dir:
            copy_with_split(
                train_images, train_masks,
                test_images, test_masks,
                output_dir
            )

        train_images = [dst for src, dst in train_images]
        test_images = [dst for src, dst in test_images]
        train_masks = [dst for src, dst in train_masks] if masks else []
        test_masks = [dst for src, dst in test_masks] if masks else []

        logger.info(f"Split dataset: {len(train_images)} training images, {len(test_images)} test images")
        logger.info(f"Train groups: {sorted(image_group_keys - test_groups)}")
        logger.info(f"Test groups: {sorted(test_groups)}")

        return train_images, train_masks, test_images, test_masks

def train_test_split_directory(
    data_dir: Union[str, Path],
    output_dir: Union[str, Path],
    test_size: float = 0.2,
    random_state: int = 42,
    image_pattern: Optional[str] = None,
    mask_pattern: Optional[str] = None,
    file_handler: DefaultFileHandler = BF_IF_FileHandler(),
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
    images = file_handler.get_files(str(data_dir), 'image')
    masks = file_handler.get_files(str(data_dir), 'mask')

    if not images:
        raise ValueError(f"No images found in {data_dir} with pattern {file_handler.image_pattern}")
    if not masks:
        raise Warning(f"No masks found in {data_dir} with pattern {file_handler.mask_pattern}")

    logger.info(f"Found {len(images)} images and {len(masks)} masks in {data_dir}")

    # TODO : Check if random seed works
    # Split the dataset
    train_images, train_masks, test_images, test_masks = split_dataset(
        images, masks, test_size, random_state, file_handler=file_handler, output_dir=output_dir, 
    )
    
    result = {
        'train_images': train_images,
        'train_masks': train_masks,
        'test_images': test_images,
        'test_masks': test_masks
    }

    # Save the split information to a JSON file
    split_info_path = output_dir / "dataset_split.json"
    with open(split_info_path, 'w') as f:
        json.dump(result, f, indent=4)
    
    return result


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Split dataset into train and test sets")
    parser.add_argument("data_dir", help="Directory containing the dataset")
    parser.add_argument("output_dir", help="Directory to save the split dataset")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data to use for testing")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducibility")
    # Image and mask patterns can be set via the file handler
    # parser.add_argument("--image-pattern", default="*_w1_*.tif", help="Glob pattern for image files")
    # parser.add_argument("--mask-pattern", default="Cells_*.tif", help="Glob pattern for mask files")
    
    args = parser.parse_args()
    
    # Configure logging
    setup_logging()
        
    # Run the split
    result = train_test_split_directory(
        args.data_dir,
        args.output_dir,
        args.test_size,
        args.random_seed,
        # args.image_pattern,
        # args.mask_pattern,
        file_handler=BF_IF_FileHandler()  # Default for command-line usage
    )
    
    # Print summary
    print(f"Train set: {len(result['train_images'])} images")
    print(f"Test set: {len(result['test_images'])} images")
