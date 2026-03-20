"""
Dataset splitting utilities for single cell analysis.

This module provides functions to split image and mask datasets into training and test sets,
ensuring that images from the same group (well, position, timepoint) stay together.
"""

import json
from pathlib import Path
import random
import logging
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Union, Any
from src.utils.file_utils import AbstractFileHandler, DefaultFileHandler, BF_IF_FileHandler, rename_all_files, copy_with_split_dict, copy_without_split_dict
from src.utils.logging_utils import setup_logging

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
        group_id = file_handler.extract_unique_id(dst)

        groups[group_id].append((src, dst))
                
    return dict(groups)

def split_dataset_dict(
        file_map: Dict[str, List[str]],
        test_size: float = 0.2,
        random_state: int = 42,
        file_handler: DefaultFileHandler = DefaultFileHandler(),
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, List[str]]:
    """
    Split a dataset of images and masks into train and test sets, keeping groups together.

    Args:
        file_map: Dictionary mapping file types to lists of file paths
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

    # {file_type : [(src_path, renamed_path)]}
    file_tuples = rename_all_files(file_map, file_handler)

    # file_tuples = {}
    # for file_type, files in file_map.items():
    #     file_tuples[file_type] = [(src, file_handler.rename_file(src, file_type)) for src in files]

    if test_size <= 0 or test_size >= 1:
        logger.info("Test size is 0 or >=1, no splitting will be performed.")

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

        # Set random seed for reproducibility
        random.seed(random_state)

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

def split_dataset_list(
        file_list: List[str],
        test_size: float = 0.2,
        random_state: int = 42,
        file_handler: DefaultFileHandler = DefaultFileHandler(),
        output_dir: Optional[Union[str, Path]] = None,
        ) -> Dict[str, List[str]]:
    """
    Same as split_dataset_dict but for a list of files instead of a dict. The file handler will be used to determine the file type and groupings. 
    """
    if file_handler is None:
        raise ValueError("No file handler provided, images may not be grouped correctly")
    
    file_map = defaultdict(list)
    for file in file_list:
        file_type = file_handler.get_file_type(file)
        file_map[file_type].append(file)

    return split_dataset_dict(file_map, test_size, random_state, file_handler, output_dir)

def train_test_split_directory(
    data_dir: Union[str, Path],
    output_dir: Union[str, Path],
    test_size: float = 0.2,
    random_state: int = 42,
    file_handler: DefaultFileHandler = DefaultFileHandler(),
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
        file_list[k] = file_handler.get_files_by_type(str(data_dir), k)
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
