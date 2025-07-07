"""
Utility functions for splitting datasets into training and test sets.

This module provides functions to split image datasets for machine learning tasks,
ensuring related images (like timepoints from the same series) stay together.
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

from .file_utils import StandardFileHandler

logger = logging.getLogger(__name__)


class DatasetSplit(NamedTuple):
    """Container for dataset split results."""
    train_images: List[str]
    train_masks: List[str]
    test_images: List[str]
    test_masks: List[str]


class DatasetSplitter:
    """Handles dataset splitting and organization."""

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        output_dir: str,
        file_handler: Optional[StandardFileHandler] = None,
        test_size: float = 0.2,
        random_state: int = 42
    ):
        """Initialize DatasetSplitter.
        
        Args:
            image_dir: Directory containing image files
            mask_dir: Directory containing mask files
            output_dir: Directory to save split datasets
            file_handler: File handler for renaming and grouping files
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.output_dir = output_dir
        self.test_size = test_size
        self.file_handler = file_handler or StandardFileHandler()
        self.random_state = random_state
        random.seed(random_state)
        
        # Find all image and mask files
        self.image_files = sorted(glob(os.path.join(image_dir, '**', '*.tif'), recursive=True))
        self.mask_files = sorted(glob(os.path.join(mask_dir, '**', '*.tif'), recursive=True))

    def find_matching_pairs(
        self, images: List[str], masks: List[str]
    ) -> Tuple[List[str], List[str]]:
        """Find matching image-mask pairs based on their position and z-stack."""
        img_keys = {img.replace('_BF', ''): img for img in images}
        mask_keys = {mask.replace('_Cells', ''): mask for mask in masks}
        
        common_keys = set(img_keys.keys()) & set(mask_keys.keys())
        
        return ([img_keys[key] for key in common_keys],
                [mask_keys[key] for key in common_keys])

    def group_by_position(
        self, images: List[str], masks: List[str]
    ) -> Dict[str, List[Tuple[str, str]]]:
        """Group images and masks by their row-column position (sample ID)."""
        position_groups = defaultdict(list)
        for img, mask in zip(images, masks):
            # Extract position from filename using pattern like "p2126_A01_z1_BF.tif"
            filename = Path(img).name
            match = re.search(r'p\d+_([A-Z]\d+)', filename)
            if match:
                position = match.group(1)
            else:
                position = Path(img).stem  # Fallback to filename without extension
                
            position_groups[position].append((img, mask))
        return position_groups

    def create_train_test_split(
        self, images: List[str] = None,
        masks: List[str] = None,
        split_by_group: bool = True,
    ) -> DatasetSplit:
        """Split data into train and test sets, keeping related images together."""
        # Use provided images/masks or instance variables
        images = images or self.image_files
        masks = masks or self.mask_files
        
        # Find matching image-mask pairs
        images, masks = self.find_matching_pairs(images, masks)
        
        if not images:
            logger.warning("No matching image-mask pairs found")
            return DatasetSplit([], [], [], [])
        
        if split_by_group:
            # Split by position/group
            position_groups = self.group_by_position(images, masks)
            positions = sorted(position_groups.keys())
            
            num_test = max(1, int(len(positions) * self.test_size))
            test_positions = set(random.sample(positions, num_test))
            
            train_images, train_masks = [], []
            test_images, test_masks = [], []
            
            for pos in positions:
                pairs = position_groups[pos]
                target_lists = (test_images, test_masks) if pos in test_positions else (train_images, train_masks)
                
                for img, mask in pairs:
                    target_lists[0].append(img)
                    target_lists[1].append(mask)
        else:
            # Simple random split
            indices = list(range(len(images)))
            random.shuffle(indices)
            split_idx = int(len(indices) * (1 - self.test_size))
            
            train_indices = indices[:split_idx]
            test_indices = indices[split_idx:]
            
            train_images = [images[i] for i in train_indices]
            train_masks = [masks[i] for i in train_indices]
            test_images = [images[i] for i in test_indices]
            test_masks = [masks[i] for i in test_indices]
        
        return DatasetSplit(train_images, train_masks, test_images, test_masks)

    def save_split(self, split: DatasetSplit, copy_files: bool = True) -> Dict[str, List[str]]:
        """Save the split to the output directory."""
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save split info to JSON
        split_info = {
            'train_images': [str(Path(p).relative_to(Path(self.image_dir))) for p in split.train_images],
            'train_masks': [str(Path(p).relative_to(Path(self.mask_dir))) for p in split.train_masks],
            'test_images': [str(Path(p).relative_to(Path(self.image_dir))) for p in split.test_images],
            'test_masks': [str(Path(p).relative_to(Path(self.mask_dir))) for p in split.test_masks]
        }
        
        with open(output_dir / 'split_info.json', 'w') as f:
            json.dump(split_info, f, indent=2)
        
        if copy_files:
            # Create train/test directories
            train_dir = output_dir / 'train'
            test_dir = output_dir / 'test'
            train_dir.mkdir(exist_ok=True)
            test_dir.mkdir(exist_ok=True)
            
            # Copy files
            for img, mask in zip(split.train_images, split.train_masks):
                shutil.copy2(img, train_dir / Path(img).name)
                shutil.copy2(mask, train_dir / Path(mask).name)
                
            for img, mask in zip(split.test_images, split.test_masks):
                shutil.copy2(img, test_dir / Path(img).name)
                shutil.copy2(mask, test_dir / Path(mask).name)
            
            logger.info(f"Split dataset saved to {output_dir}")
            
            # Return paths to copied files
            return {
                'train_images': [str(train_dir / Path(img).name) for img in split.train_images],
                'train_masks': [str(train_dir / Path(mask).name) for mask in split.train_masks],
                'test_images': [str(test_dir / Path(img).name) for img in split.test_images],
                'test_masks': [str(test_dir / Path(mask).name) for mask in split.test_masks]
            }
        
        return {
            'train_images': split.train_images,
            'train_masks': split.train_masks,
            'test_images': split.test_images,
            'test_masks': split.test_masks
        }


def split_dataset(
    images: List[str],
    masks: List[str],
    test_size: float = 0.2,
    random_state: int = 42,
    group_by_prefix: bool = True
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Split a dataset of images and masks into train and test sets.
    
    Args:
        images: List of image file paths
        masks: List of mask file paths
        test_size: Fraction of data to use for testing (0-1)
        random_state: Random seed for reproducibility
        group_by_prefix: If True, keeps images with the same prefix together
        
    Returns:
        Tuple containing (train_images, train_masks, test_images, test_masks)
    """
    # Set random seed for reproducibility
    random.seed(random_state)
    
    # Find matching image-mask pairs
    img_keys = {Path(img).stem.split('_BF')[0]: img for img in images}
    mask_keys = {Path(mask).stem.split('_Cells')[0]: mask for mask in masks}
    
    # Get common keys (images that have corresponding masks)
    common_keys = set(img_keys.keys()) & set(mask_keys.keys())
    
    if not common_keys:
        raise ValueError("No matching image-mask pairs found")
    
    # Extract pairs
    img_mask_pairs = [(img_keys[k], mask_keys[k]) for k in common_keys]
    
    # Group by prefix if requested
    if group_by_prefix:
        # Extract position/sample identifier from filename
        def get_prefix(filename):
            # Assumes filename format like "Pos01_...", "Sample5_...", etc.
            name = Path(filename).name
            # Try to find a prefix pattern (letters followed by numbers)
            import re
            match = re.search(r'([A-Za-z]+\d+)_', name)
            if match:
                return match.group(1)
            return name.split('_')[0]  # Fallback to first part before underscore
        
        # Group by prefix
        groups = {}
        for img, mask in img_mask_pairs:
            prefix = get_prefix(img)
            if prefix not in groups:
                groups[prefix] = []
            groups[prefix].append((img, mask))
        
        # Split groups into train/test
        prefixes = list(groups.keys())
        n_test = max(1, int(len(prefixes) * test_size))
        test_prefixes = set(random.sample(prefixes, n_test))
        
        # Separate images and masks
        train_images, train_masks = [], []
        test_images, test_masks = [], []
        
        for prefix, pairs in groups.items():
            if prefix in test_prefixes:
                for img, mask in pairs:
                    test_images.append(img)
                    test_masks.append(mask)
            else:
                for img, mask in pairs:
                    train_images.append(img)
                    train_masks.append(mask)
    else:
        # Simple random split without grouping
        n_test = max(1, int(len(img_mask_pairs) * test_size))
        random.shuffle(img_mask_pairs)
        test_pairs = img_mask_pairs[:n_test]
        train_pairs = img_mask_pairs[n_test:]
        
        train_images = [img for img, _ in train_pairs]
        train_masks = [mask for _, mask in train_pairs]
        test_images = [img for img, _ in test_pairs]
        test_masks = [mask for _, mask in test_pairs]
    
    logger.info(f"Split dataset: {len(train_images)} training images, {len(test_images)} test images")
    return train_images, train_masks, test_images, test_masks


def train_test_split_directory(
    data_dir: Union[str, Path],
    output_dir: Union[str, Path],
    test_size: float = 0.2,
    random_state: int = 42,
    image_pattern: str = "*_BF*.tif",
    mask_pattern: str = "*_Cells*.tif",
    copy_files: bool = True
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
        copy_files: If True, copy files to output directory; if False, just return paths
        
    Returns:
        Dictionary with keys 'train_images', 'train_masks', 'test_images', 'test_masks'
        containing the paths of the files in each split
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    
    # Find all images and masks
    images = sorted(glob(str(data_dir / image_pattern)))
    masks = sorted(glob(str(data_dir / mask_pattern)))
    
    logger.info(f"Found {len(images)} images and {len(masks)} masks in {data_dir}")
    
    # Split the dataset
    train_images, train_masks, test_images, test_masks = split_dataset(
        images, masks, test_size, random_state
    )
    
    result = {
        'train_images': train_images,
        'train_masks': train_masks,
        'test_images': test_images,
        'test_masks': test_masks
    }
    
    if copy_files:
        # Create output directories
        train_dir = output_dir / 'train'
        test_dir = output_dir / 'test'
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy files to their respective directories
        for img, mask in zip(train_images, train_masks):
            shutil.copy2(img, train_dir / Path(img).name)
            shutil.copy2(mask, train_dir / Path(mask).name)
            
        for img, mask in zip(test_images, test_masks):
            shutil.copy2(img, test_dir / Path(img).name)
            shutil.copy2(mask, test_dir / Path(mask).name)
            
        logger.info(f"Files copied to {train_dir} and {test_dir}")
        
        # Update result to use new paths
        result = {
            'train_images': [str(train_dir / Path(img).name) for img in train_images],
            'train_masks': [str(train_dir / Path(mask).name) for mask in train_masks],
            'test_images': [str(test_dir / Path(img).name) for img in test_images],
            'test_masks': [str(test_dir / Path(mask).name) for mask in test_masks]
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
    parser.add_argument("--image-pattern", default="*_BF*.tif", help="Glob pattern for image files")
    parser.add_argument("--mask-pattern", default="*_Cells*.tif", help="Glob pattern for mask files")
    
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
        args.mask_pattern
    )
    
    # Print summary
    print(f"Train set: {len(result['train_images'])} images")
    print(f"Test set: {len(result['test_images'])} images")

