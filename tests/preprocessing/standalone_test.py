"""
Standalone test for dataset split functionality.
"""
import os
import re
import sys
import unittest
import tempfile
import shutil
from pathlib import Path

# Make sure we can import the modules we need
project_root = Path("e:/sera/Helmholtz/single_cell")
sys.path.insert(0, str(project_root))

# Direct imports to avoid dependency issues
# We're importing specifically what we need to test
from src.preprocessing.dataset_split import split_dataset, DatasetSplit
from src.utils.file_utils import BF_IF_FileHandler


def main():
    """Run a simple test to verify dataset split functionality."""
    print("Running standalone dataset split test...")
    
    # Setup paths
    data_dir = project_root / "data" / "Plate 2126 Compressed T 2hr"
    image_dir = data_dir
    mask_dir = data_dir / "Plate 2126 Masks"
    temp_dir = tempfile.mkdtemp()
    output_dir = Path(temp_dir) / "output"
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Get sample data - just a few files for quick testing
        image_files = list(str(p) for p in image_dir.glob("t1_*_w1_z1.tif"))[:3]  # First z-stack from each group
        mask_files = list(str(p) for p in mask_dir.glob("Cells_*-Z0-T0.tif"))[:3]  # First z-stack from each group
        
        print(f"Found {len(image_files)} image files and {len(mask_files)} mask files")
        
        # Create file handler
        file_handler = BF_IF_FileHandler()
        
        # Split with a fixed random seed for reproducibility
        train_images, train_masks, test_images, test_masks = split_dataset(
            image_files, mask_files, test_size=0.33, random_state=42,
            file_handler=file_handler
        )
        
        # Check that images were split
        print(f"Split result: {len(train_images)} training images, {len(test_images)} test images")
        
        # Extract groups from train and test images
        train_groups = set()
        test_groups = set()
        
        for img in train_images:
            match = re.search(r'_([A-Za-z]+\d+)_', Path(img).name)
            if match:
                train_groups.add(match.group(1))
                
        for img in test_images:
            match = re.search(r'_([A-Za-z]+\d+)_', Path(img).name)
            if match:
                test_groups.add(match.group(1))
        
        # Check that no group appears in both splits
        overlap = train_groups & test_groups
        if overlap:
            print(f"ERROR: Groups appear in both train and test splits: {overlap}")
        else:
            print("SUCCESS: Groups are correctly kept together in splits")
            print(f"Train groups: {train_groups}")
            print(f"Test groups: {test_groups}")
            
        return 0
    except Exception as e:
        print(f"ERROR: Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main())
