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
from typing import List, Tuple, Dict, Optional, Union, Any
from .file_utils import AbstractFileHandler, DefaultFileHandler, BF_IF_FileHandler, BF_FileHandler


import logging
import sys, os
from typing import Dict, Any
from pathlib import Path

def setup_logging(level: str = "INFO", log_file = None, verbose: bool = True, log_config: Dict[str, Any] = {}) -> None:
    """Setup logging configuration."""

    log_level = level or log_config.get('level', 'INFO')
    if isinstance(log_level, str):
        log_level = log_level.upper()

    log_file = log_file or log_config.get('log_file', None)
    
    if log_file:
        log_dir = Path(log_file).parent
        os.makedirs(log_dir, exist_ok=True)

    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Set up handlers
    handlers = []
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    if verbose:
        handlers.append(logging.StreamHandler(sys.stdout))

    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers
    )



logger = logging.getLogger(__name__)


def get_groups_from_filenames(
        renamed_tuples, file_handler: AbstractFileHandler
        ) -> Dict[str, List[Tuple[str, str]]]:
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

def copy_without_split_dict(
        file_tuple : Dict[str, List[Tuple[str, str]]],
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

    for k, v in file_tuple.items():
        for file_pair in v:
            src, dst = file_pair
            copy_file(src, output_dir / dst)

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

def copy_with_split_dict(
        train_file_tuple : Dict[str, List[Tuple[str, str]]],
        test_file_tuple : Dict[str, List[Tuple[str, str]]],
        output_dir: Union[str, Path],
        filter_file_keys: Optional[List[str]] = None,
    ):
    """
    Copy image and mask files into train and test subdirectories.

    Args:
        train_file_tuple: Dictionary containing train file tuples {"images": [...], "masks": [...]}
        test_file_tuple: Dictionary containing test file tuples {"images": [...], "masks": [...]}
        output_dir: Directory to copy the files to
    """
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dir = output_dir / 'train'
    test_dir = output_dir / 'test'
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    if filter_file_keys:
        train_file_tuple = {k: v for k, v in train_file_tuple.items() if k in filter_file_keys}
        test_file_tuple = {k: v for k, v in test_file_tuple.items() if k in filter_file_keys}

    # TODO : Handle masks separately into mask subdirs
    for k, v in test_file_tuple.items():
        for src, dst in v:
            copy_file(src, test_dir / dst)

    for k, v in train_file_tuple.items():
        for src, dst in v:
            copy_file(src, train_dir / dst)


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

def split_dataset_dict(
        file_list: Dict[str, List[str]],
        test_size: float = 0.2,
        random_state: int = 42,
        file_handler: DefaultFileHandler = BF_FileHandler(),
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, List[str]]:
    """
    Split a dataset of images and masks into train and test sets, keeping groups together.

    Args:
        file_list: List of all file paths
        test_size: Fraction of data to use for testing (0-1)
        random_state: Random seed for reproducibility
        file_handler: File renamer/handler for extracting group info
        output_dir: Directory to save the split files. If None, files are not copied.
    Returns:
        Tuple containing (train_images, train_masks, test_images, test_masks)
    """

    if file_handler is None:
        raise ValueError("No file handler provided, images may not be grouped correctly")
    
    # TODO : Temporarily disable check for masks
    # if masks and len(images) != len(masks):
    #     raise ValueError("Number of images and masks must be the same", f"Images: {len(images)} != Masks: {len(masks)}")

    # Set random seed for reproducibility
    random.seed(random_state)

    if test_size <= 0 or test_size >= 1:
        logger.info("Test size is 0 or >=1, no splitting will be performed.")

        file_tuples = {}
        for k, files in file_list.items():
            file_tuples[k] = [(src, file_handler.rename_file(src, k)) for src in files]

        if output_dir:
            copy_without_split_dict(file_tuples, output_dir)

        all_files = {k: [dst for src, dst in v] for k, v in file_tuples.items()}

        if test_size <= 0:
            logger.info("All data assigned to training set.")
            return all_files, {} # type: ignore
        else:
            logger.info("All data assigned to test set.")
            return {}, all_files # type: ignore

    else:
        logger.info(f"Splitting dataset with test size = {test_size}")

        # Rename files and create tuples : src, dst (base name only)
        file_tuples = {}
        for file_type, files in file_list.items():
            file_tuples[file_type] = [(src, file_handler.rename_file(src, file_type)) for src in files]

        grouped_file_tuples = {}
        group_keys = set()
        for file_type, files in file_tuples.items():
            grouped_file_tuples[file_type] = get_groups_from_filenames(files, file_handler)
            group_keys.update(grouped_file_tuples[file_type].keys())

        # Group files based on extracted group IDs
        # grouped_image_tuples = get_groups_from_filenames(image_tuples, file_handler)
        # grouped_mask_tuples = get_groups_from_filenames(mask_tuples, file_handler) if masks else {}

        # LETS SKIP THESE CHECKS FOR NOW : Check if both images and masks have the same groups
        # image_group_keys = set(grouped_image_tuples.keys())
        # mask_group_keys = set(grouped_mask_tuples.keys())
        # if masks and image_group_keys != mask_group_keys:
        #     raise ValueError(f"Image and mask groups do not match: {image_group_keys ^ mask_group_keys}")
        
        # split based on image_group_keys
        groups = list(group_keys)
        # This doesn't mean that the split is uniform (since groups can have different sizes)
        n_test = max(1, int(len(groups) * test_size))
        test_groups = set(random.sample(groups, n_test))

        train_files = {file_type: [] for file_type in file_tuples.keys()}
        test_files = {file_type: [] for file_type in file_tuples.keys()}

        for file_type, grouped_files in grouped_file_tuples.items():
            for group, files in grouped_files.items():
                if group in test_groups:
                    test_files[file_type].extend(files)
                else:
                    train_files[file_type].extend(files)

        if output_dir:
            copy_with_split_dict(train_files, test_files, output_dir)

        train_files = {file_type: list(map(lambda x: x[1], files)) for file_type, files in train_files.items()}
        test_files = {file_type: list(map(lambda x: x[1], files)) for file_type, files in test_files.items()}

        logger.info(f"Split dataset: {len(train_files.get('image', []))} training images, {len(test_files.get('image', []))} test images")
        logger.info(f"Train groups: {sorted(group_keys - test_groups)}")
        logger.info(f"Test groups: {sorted(test_groups)}")

        return train_files, test_files # type: ignore

def train_test_split_directory(
    data_dir: Union[str, Path],
    output_dir: Union[str, Path],
    test_size: float = 0.2,
    random_state: int = 42,
    file_handler: DefaultFileHandler = BF_FileHandler(),
) -> Dict[str, Any]:
    """
    Split data in a directory into train and test sets and organize into subdirectories.
    
    Args:
        data_dir: Directory containing images and masks
        output_dir: Directory to save the split data
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        file_handler: File renamer/handler for extracting group info
        
    Returns:
        Dictionary with keys 'train_images', 'train_masks', 'test_images', 'test_masks'
        containing the paths of the files in each split
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    # Find all images and masks
    file_list = {}
    for k, v in file_handler.patterns.items():
        file_list[k] = file_handler.get_files(str(data_dir), k)
        if len(file_list[k]) == 0:
            logger.warning(f"No files found for pattern '{v}' in {data_dir} for type '{k}'")

    # TODO : Check if random seed works
    # Split the dataset

    train_files, test_files = split_dataset_dict(
        file_list, test_size, random_state, file_handler=file_handler, output_dir=output_dir, 
    )
    
    result = {
        'train_files': train_files,
        'test_files': test_files
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
        file_handler=BF_IF_FileHandler()  # Default for command-line usage
    )
    
    # Print summary
    print(f"Train set: {len(result['train_files'])} images")
    print(f"Test set: {len(result['test_files'])} images")
