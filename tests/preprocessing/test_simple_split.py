"""
Simple test for dataset split functionality.
"""
import os
import re
import unittest
import tempfile
import shutil
from pathlib import Path

# Import directly from the module to avoid import errors
import sys
sys.path.append("e:/sera/Helmholtz/single_cell")

from src.preprocessing.dataset_split import train_test_split_directory

class TestSimpleDatasetSplit(unittest.TestCase):
    """Simple test for dataset splitting functionality."""
    
    def setUp(self):
        """Set up mock test data paths and files."""
        # Create temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir) / "mock_data" / "Plate 2126"
        self.output_dir = Path(self.temp_dir) / "output"
        
        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create mock image and mask files following the expected pattern
        # Pattern: t1_A01_s1_w1_z1.tif for images
        # Pattern: Cells_R1-C1-F1-Z1-T1.tif for masks
        mock_files = [
            # Well A01 files
            "t1_A01_s1_w1_z1.tif",
            "t1_A01_s1_w1_z2.tif", 
            "Cells_R1-C1-F1-Z1-T1.tif",
            "Cells_R1-C1-F1-Z2-T1.tif",
            
            # Well B02 files  
            "t1_B02_s1_w1_z1.tif",
            "t1_B02_s1_w1_z2.tif",
            "Cells_R2-C2-F1-Z1-T1.tif",
            "Cells_R2-C2-F1-Z2-T1.tif",
            
            # Well C03 files
            "t1_C03_s1_w1_z1.tif", 
            "t1_C03_s1_w1_z2.tif",
            "Cells_R3-C3-F1-Z1-T1.tif",
            "Cells_R3-C3-F1-Z2-T1.tif"
        ]
        
        # Create mock files
        for filename in mock_files:
            mock_file = self.data_dir / filename
            mock_file.write_text("mock image data")
        
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_split_keeps_groups_together(self):
        """Test that splitting keeps files from the same group together."""
        # Run the split function with mock data using correct patterns
        result = train_test_split_directory(
            str(self.data_dir),
            str(self.output_dir),
            test_size=0.33,
            random_state=42,
            image_pattern="t1_*_s1_w1_z1.tif",  # BF+IF pattern
            mask_pattern="Cells_*-Z1-T1.tif"   # BF+IF mask pattern
        )
        
        # Verify that we have results
        self.assertGreater(len(result['train_images']), 0, "Should have training images")
        self.assertGreater(len(result['test_images']), 0, "Should have test images")
        
        # Extract well identifiers from filenames (A01, B02, C03)
        train_wells = set()
        test_wells = set()
        
        for img in result['train_images']:
            filename = Path(img).name
            match = re.search(r't1_([A-Z]\d+)_s1_w1_z1', filename)
            if match:
                train_wells.add(match.group(1))
                
        for img in result['test_images']:
            filename = Path(img).name
            match = re.search(r't1_([A-Z]\d+)_s1_w1_z1', filename)
            if match:
                test_wells.add(match.group(1))
        
        # Print the wells for debugging
        print(f"Train wells: {train_wells}")
        print(f"Test wells: {test_wells}")
        
        # Verify wells are kept together (no overlap between train and test)
        overlap = train_wells & test_wells
        self.assertEqual(len(overlap), 0, 
                        f"Wells should not appear in both train and test splits. Found in both: {overlap}")
        
        # Verify we have at least one well in each split
        self.assertGreater(len(train_wells), 0, "Should have at least one training well")
        self.assertGreater(len(test_wells), 0, "Should have at least one test well")
        
    def test_split_with_no_matching_files(self):
        """Test behavior when no files match the pattern."""
        # The function should raise a ValueError when no matching files are found
        with self.assertRaises(ValueError) as context:
            train_test_split_directory(
                str(self.data_dir),
                str(self.output_dir),
                test_size=0.33,
                random_state=42,
                image_pattern="nonexistent_*.tif",
                mask_pattern="nonexistent_*.tif"
            )
        
        # Verify the error message
        self.assertIn("No matching image-mask groups found", str(context.exception))
        
    def test_split_proportions(self):
        """Test that the split proportions are approximately correct."""
        result = train_test_split_directory(
            str(self.data_dir),
            str(self.output_dir),
            test_size=0.33,
            random_state=42,
            image_pattern="t1_*_s1_w1_z1.tif",
            mask_pattern="Cells_*-Z1-T1.tif"
        )
        
        total_images = len(result['train_images']) + len(result['test_images'])
        test_proportion = len(result['test_images']) / total_images
        
        # Allow for some rounding - with 3 wells and 33% test size, we expect 1/3 ≈ 0.33
        self.assertAlmostEqual(test_proportion, 0.33, delta=0.1, 
                             msg=f"Test proportion {test_proportion:.2f} should be close to 0.33")
        
        # Verify all splits have corresponding masks
        self.assertEqual(len(result['train_images']), len(result['train_masks']),
                        "Number of training images should match training masks")
        self.assertEqual(len(result['test_images']), len(result['test_masks']),
                        "Number of test images should match test masks")


if __name__ == '__main__':
    unittest.main()
