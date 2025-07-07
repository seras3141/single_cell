"""
Test dataset splitting functionality.

This module tests the dataset splitting functionality in the preprocessing module,
ensuring that images from the same key (e.g., J03) are kept together in either
the training or test split.
"""

import os
import re
import unittest
import tempfile
import shutil
from pathlib import Path

from src.preprocessing.dataset_split import (
    DatasetSplit,
    split_dataset,
    train_test_split_directory,
    get_groups_from_filenames
)
from src.utils.file_utils import BF_IF_FileHandler, DefaultFileHandler


class TestDatasetSplit(unittest.TestCase):
    """Test dataset splitting functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up mock test data."""
        # Create a temporary directory for mock data
        cls.mock_data_dir = tempfile.mkdtemp()
        # Create directory structure that includes "Plate" in the path for the file handler
        cls.data_dir = Path(cls.mock_data_dir) / "Plate 2126 Test Data"
        cls.image_dir = cls.data_dir
        cls.mask_dir = cls.data_dir / "masks"
        
        # Create directory structure
        cls.data_dir.mkdir(parents=True, exist_ok=True)
        cls.mask_dir.mkdir(parents=True, exist_ok=True)
        
        # Create mock image files for different groups
        cls.mock_image_files = [
            # Group J03 (position 3)
            "t1_J03_s1_w1_z1.tif",
            "t1_J03_s1_w1_z2.tif", 
            "t1_J03_s1_w1_z3.tif",
            "t1_J03_s1_w2_z1.tif",
            "t1_J03_s1_w2_z2.tif",
            # Group J04 (position 4)
            "t1_J04_s1_w1_z1.tif",
            "t1_J04_s1_w1_z2.tif",
            "t1_J04_s1_w1_z3.tif",
            "t1_J04_s1_w2_z1.tif", 
            "t1_J04_s1_w2_z2.tif",
            # Group L11 (position 11)
            "t1_L11_s1_w1_z1.tif",
            "t1_L11_s1_w1_z2.tif",
            "t1_L11_s1_w2_z1.tif",
        ]
        
        # Create mock mask files (corresponding to image groups)
        cls.mock_mask_files = [
            # Masks for J03 (R10-C3 maps to J03)
            "Cells_R10-C3-F0-Z0-T0.tif",
            "Cells_R10-C3-F0-Z1-T0.tif",
            "Cells_R10-C3-F0-Z2-T0.tif",
            # Masks for J04 (R10-C4 maps to J04)  
            "Cells_R10-C4-F0-Z0-T0.tif",
            "Cells_R10-C4-F0-Z1-T0.tif",
            "Cells_R10-C4-F0-Z2-T0.tif",
            # Masks for L11 (R12-C11 maps to L11)
            "Cells_R12-C11-F0-Z0-T0.tif",
            "Cells_R12-C11-F0-Z1-T0.tif",
        ]
        
        # Create the actual files (empty files for testing)
        for filename in cls.mock_image_files:
            file_path = cls.image_dir / filename
            file_path.touch()
            
        for filename in cls.mock_mask_files:
            file_path = cls.mask_dir / filename
            file_path.touch()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up mock data directory."""
        import shutil
        shutil.rmtree(cls.mock_data_dir, ignore_errors=True)
    
    def setUp(self):
        """Set up temporary directory for test outputs."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "output"
        self.output_dir.mkdir(exist_ok=True)
    
    def tearDown(self):
        """Clean up temporary directory after tests."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_file_handler(self):
        """Test BF_IF_FileHandler for correctly handling the mock data files."""
        file_handler = BF_IF_FileHandler()
        
        # Test image file renaming
        sample_image = str(self.image_dir / "t1_J03_s1_w1_z1.tif")
        renamed_image = file_handler.rename_image(sample_image)
        
        # The renamed file should contain J03
        self.assertIn("J03", renamed_image)
        
        # Test mask file renaming
        sample_mask = str(self.mask_dir / "Cells_R10-C3-F0-Z0-T0.tif")
        renamed_mask = file_handler.rename_mask(sample_mask)
        
        # The renamed file should contain J03 (R10-C3 gets converted to J03)
        self.assertIn("J03", renamed_mask)
        
        # Test group ID extraction
        group_id = file_handler.extract_group_id(renamed_image)
        self.assertEqual(group_id, "p2126_J03")
        
        # Verify that mask has correct group_id
        mask_group_id = file_handler.extract_group_id(renamed_mask)
        # This should map to p2126_J03 based on the BF_IF_FileHandler implementation
        self.assertEqual(mask_group_id, "p2126_J03")
    
    def test_get_groups_from_filenames(self):
        """Test extracting groups from filenames using file_handler."""
        # Setup file handler
        file_handler = BF_IF_FileHandler()
        
        # Get image files for testing (use mock files)
        image_files = [str(self.image_dir / f) for f in self.mock_image_files[:5]]  # Take just a few files
        
        # Generate output filenames
        output_files = [file_handler.rename_image(img) for img in image_files]
        
        # Group the files
        file_map = dict(zip(image_files, output_files))
        groups = get_groups_from_filenames(file_map, file_handler)
        
        # Should have one group (p2126_J03) since we're taking files from the same group
        self.assertEqual(len(groups), 1)
        self.assertIn("p2126_J03", groups)
        
        # The group should contain all 5 files
        self.assertEqual(len(groups["p2126_J03"]), 5)
    
    def test_split_dataset(self):
        """Test splitting dataset using the new split_dataset function."""
        # Get all mock image and mask files
        image_files = [str(self.image_dir / f) for f in self.mock_image_files]
        mask_files = [str(self.mask_dir / f) for f in self.mock_mask_files]
        
        # Split with a fixed random seed for reproducibility
        train_images, train_masks, test_images, test_masks = split_dataset(
            image_files, mask_files, test_size=0.33, random_state=42,
            file_handler=BF_IF_FileHandler()
        )
        
        # Check that images were split
        self.assertTrue(len(train_images) > 0)
        self.assertTrue(len(test_images) > 0)
        self.assertTrue(len(train_masks) > 0)
        self.assertTrue(len(test_masks) > 0)
        
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
        self.assertEqual(len(train_groups & test_groups), 0, 
                         f"Groups should not appear in both train and test splits. Found: {train_groups & test_groups}")
        

    def test_train_test_split_directory(self):
        """Test train_test_split_directory function with the updated interface."""
        # Split dataset using the mock data directory
        result = train_test_split_directory(
            str(self.data_dir),
            str(self.output_dir),
            test_size=0.33,
            random_state=42,
            image_pattern="t1_*_w1_*.tif",  # Only use w1 images for faster test
            mask_pattern="Cells_*.tif"
        )
        
        # Check that we got results
        self.assertTrue(len(result["train_images"]) > 0)
        self.assertTrue(len(result["test_images"]) > 0)
        
        # Get groups from train and test images
        train_groups = set()
        test_groups = set()
        
        for img in result['train_images']:
            filename = Path(img).name
            match = re.search(r'_([A-Za-z]+\d+)_', filename)
            if match:
                train_groups.add(match.group(1))
                
        for img in result['test_images']:
            filename = Path(img).name
            match = re.search(r'_([A-Za-z]+\d+)_', filename)
            if match:
                test_groups.add(match.group(1))
        
        # Check that no group appears in both splits
        self.assertEqual(len(train_groups & test_groups), 0, 
                         "Groups should not appear in both train and test splits")


if __name__ == '__main__':
    import re  # Ensure re is available for the tests
    unittest.main()
