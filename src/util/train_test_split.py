import json
import sys
proj_path = "/home/haicu/serena.sritharan/projects/single_cell"
sys.path.append(proj_path)

import os
from glob import glob
from pathlib import Path
from typing import List, Tuple, Dict, Set, NamedTuple
from collections import defaultdict
import random
import re
from dataclasses import dataclass
from logging import getLogger
from util.file_renamer import *

logger = getLogger(__name__)


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
        paths: DatasetPaths,
        file_handler: DefaultFileRenamer,
        test_size: float = 0.2,
        random_state: int = 42
    ):
        """Initialize DatasetSplitter.
        
        Args:
            paths: DatasetPaths object containing required paths
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
        """
        self.paths = paths
        self.test_size = test_size
        self.file_handler = file_handler
        self.random_state = random_state
        random.seed(random_state)

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
            position = self.file_handler.get_group(img)
            position_groups[position].append((img, mask))
        return position_groups

    def create_train_test_split(
        self, images: List[str],
        masks: List[str],
        split_by_group: bool = True,
    ) -> DatasetSplit:
        
        if split_by_group:
            """Split data into train and test sets, keeping z-stacks together."""
            position_groups = self.group_by_position(images, masks)
            positions = sorted(position_groups.keys())  # Sort positions to ensure deterministic order
        
            num_test = max(1, int(len(positions) * self.test_size))
            random.seed(self.random_state)  # Set the random seed for reproducibility
            test_positions = set(random.sample(positions, num_test))  # Randomly sample test positions
        
            train_images, train_masks = [], []
            test_images, test_masks = [], []
        
            for pos in positions:
                pairs = position_groups[pos]
                target_lists = (test_images, test_masks) if pos in test_positions else (train_images, train_masks)
            
                for img, mask in pairs:
                    target_lists[0].append(img)
                    target_lists[1].append(mask)
        
        else:
            """Split data into train and test sets without grouping."""
            combined = list(zip(images, masks))
            random.seed(self.random_state)  # Set the random seed for reproducibility
            random.shuffle(combined)  # Shuffle the data
            
            split_idx = int(len(combined) * (1 - self.test_size))
            train_pairs = combined[:split_idx]
            test_pairs = combined[split_idx:]
            
            train_images, train_masks = zip(*train_pairs) if train_pairs else ([], [])
            test_images, test_masks = zip(*test_pairs) if test_pairs else ([], [])
        
        return DatasetSplit(
            list(train_images), list(train_masks),
            list(test_images), list(test_masks)
        )
    def create_dataset_structure(
        self, 
        split: DatasetSplit, 
        original_files: Tuple[List[str], List[str]],
        make_json: bool = True,
    ) -> None:
        """Create directory structure with symlinks to original files."""
        def create_symlinks(orig_files: dict, target_names: List[str], target_dir: str) -> None:
            for target in target_names:
                src = os.path.abspath(orig_files[target])
                dst = os.path.join(target_dir, target)
                if os.path.exists(dst):
                    os.remove(dst)
                os.symlink(src, dst)

        # Create directory structure for train and test
        for split_type in ["train", "test"]:
            Path(f"{self.paths.output_dir}/{split_type}").mkdir(parents=True, exist_ok=True)

        original_images, original_masks = original_files
        img_name_map = {self.file_handler.rename_image(i): i for i in original_images}
        mask_name_map = {self.file_handler.rename_mask(i): i for i in original_masks}

        if make_json:
            # Save split information to JSON
            split_json = {
                "train_images": split.train_images,
                "train_masks": split.train_masks,
                "test_images": split.test_images,
                "test_masks": split.test_masks
            }
            with open(f"{self.paths.output_dir}/split.json", "w") as f:
                json.dump(split_json, f, indent=4)

        # Create symlinks for training and test sets
        for data_split, images, masks in [
            ("train", split.train_images, split.train_masks),
            ("test", split.test_images, split.test_masks)
        ]:
            split_dir = f"{self.paths.output_dir}/{data_split}"
            create_symlinks(img_name_map, images, split_dir)
            create_symlinks(mask_name_map, masks, split_dir)

    def verify_split(self, split: DatasetSplit) -> None:
        """Verify the integrity of the train-test split."""
        train_positions = {img.split('_')[0] for img in split.train_images}
        test_positions = {img.split('_')[0] for img in split.test_images}
        
        logger.info("\nDataset Summary:")
        logger.info(f"Training set: {len(split.train_images)} images")
        logger.info(f"Test set: {len(split.test_images)} images")
        logger.info(f"Unique positions in train: {len(train_positions)}")
        logger.info(f"Unique positions in test: {len(test_positions)}")
        logger.info(f"Clean train-test split: {len(train_positions & test_positions) == 0}")

    def process(self) -> None:
        """Execute the complete dataset preparation pipeline."""
        # Load and rename files
        
        renamed_images = [self.file_handler.rename_image(f) for f in self.paths.image_files]
        renamed_masks = [self.file_handler.rename_mask(f) for f in self.paths.mask_files]
        
        # Find matching pairs 
        matching_images, matching_masks = self.find_matching_pairs(renamed_images, renamed_masks)
        #  Create split
        split = self.create_train_test_split(matching_images, matching_masks)
        
        # Create dataset structure and verify
        self.create_dataset_structure(split, (self.paths.image_files, self.paths.mask_files))
        self.verify_split(split)

def main():
    # paths = DatasetPaths(
    #     image_path="data/EXP 1 - IF Markers 72hrs/Z-stacks/Plate_1844_Z-stacks/*_w5_*.tif",
    #     mask_path="data/EXP 1 IN Carta Segmentation/Masks/Cells_*.tif",
    #     output_dir="data/EXP 1 - train_test_dataset"
    # )

    
    paths = DatasetPaths(
        image_path="data/BF+IF Experiments Labeled/**/t1_*_w1_*.tif",
        mask_path="data/BF+IF Experiments Labeled/**/Cells_*.tif",
        output_dir="data/BF+IF Experiments_2D_train_test_dataset"
    )

    file_handler = BF_IF_FileRenamer()
    
    '''
    paths_3D = DatasetPaths(
        image_path="data/BF+IF Experiments Labeled_3D/**/*_BF_3D.tif",
        mask_path="data/BF+IF Experiments Labeled_3D/**/*_Cells_3D.tif",
        output_dir="data/BF+IF Experiments_3D_train_test_dataset"
    )

    file_handler_3D = BF_IF_FileRenamer_3D()
    '''
    
    splitter = DatasetSplitter(paths=paths, file_handler=file_handler)
    splitter.process()

if __name__ == "__main__":
    main()

