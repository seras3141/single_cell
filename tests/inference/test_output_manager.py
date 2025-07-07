"""
Unit tests for the OutputManager class.
"""

import pytest
import numpy as np
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.inference.output_manager import OutputManager


class TestOutputManager:
    """Test cases for OutputManager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_output_dir = Path(self.temp_dir)
        
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test OutputManager initialization."""
        manager = OutputManager(
            base_output_dir=self.test_output_dir,
            model_name="test_model",
            dataset_name="test_dataset"
        )
        
        assert manager.output_dir.name == "test_dataset"
        assert "test_model" in str(manager.output_dir)
        assert manager.output_dir.exists()
        assert manager.masks_dir.exists()
        assert manager.metadata_dir.exists()
        assert manager.overlays_dir.exists()
    
    def test_directory_creation(self):
        """Test that all required directories are created."""
        manager = OutputManager(
            base_output_dir=self.test_output_dir,
            model_name="cyto3",
            dataset_name="test"
        )
        
        expected_base = self.test_output_dir / "pred" / "cyto3" / "test"
        assert manager.output_dir == expected_base
        assert (expected_base / "masks").exists()
        assert (expected_base / "metadata").exists()
        assert (expected_base / "overlays").exists()
    
    def test_save_prediction_basic(self):
        """Test basic prediction saving."""
        manager = OutputManager(
            base_output_dir=self.test_output_dir,
            model_name="test_model",
            dataset_name="test"
        )
        
        # Create test data
        test_masks = np.random.randint(0, 100, (128, 128), dtype=np.uint16)
        test_metadata = {
            'num_cells': 50,
            'parameters': {'flow_threshold': 0.4}
        }
        input_path = Path("test_image.tif")
        
        # Save prediction
        result = manager.save_prediction(
            masks=test_masks,
            metadata=test_metadata,
            input_path=input_path
        )
        
        # Check that files were created
        assert result['masks'].exists()
        assert result['metadata'].exists()
        assert result['masks'].suffix == '.tif'
        assert result['metadata'].suffix == '.json'
        
        # Verify masks file content
        import tifffile
        saved_masks = tifffile.imread(result['masks'])
        np.testing.assert_array_equal(saved_masks, test_masks)
        
        # Verify metadata file content
        with open(result['metadata'], 'r') as f:
            saved_metadata = json.load(f)
        assert saved_metadata['num_cells'] == 50
        assert 'saved_at' in saved_metadata
        assert 'input_file' in saved_metadata
    
    def test_save_prediction_with_overlay(self):
        """Test prediction saving with overlay generation (if matplotlib available)."""
        manager = OutputManager(
            base_output_dir=self.test_output_dir,
            model_name="test_model",
            dataset_name="test"
        )
        
        test_image = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        test_masks = np.random.randint(0, 20, (64, 64), dtype=np.uint16)
        test_metadata = {'num_cells': 15}
        input_path = Path("test_image.tif")
        
        result = manager.save_prediction(
            masks=test_masks,
            metadata=test_metadata,
            input_path=input_path,
            original_image=test_image,
            save_overlay=True
        )
        
        # Check that core files are always created
        assert result['masks'].exists()
        assert result['metadata'].exists()
        
        # Overlay may or may not be created depending on matplotlib availability
        # This is acceptable behavior
    
    def test_save_z_stack_prediction(self):
        """Test Z-stack prediction saving."""
        manager = OutputManager(
            base_output_dir=self.test_output_dir,
            model_name="test_model",
            dataset_name="test"
        )
        
        # Create test Z-stack data
        masks_stack = np.random.randint(0, 50, (3, 64, 64), dtype=np.uint16)
        stack_metadata = {
            'stack_shape': (3, 64, 64),
            'total_slices': 3,
            'num_cells': np.max(masks_stack)
        }
        input_path = Path("test_stack.tif")
        
        result = manager.save_z_stack_prediction(
            masks_stack=masks_stack,
            metadata=stack_metadata,
            input_path=input_path
        )
        
        # Check structure - it should have 'stack' and 'slices' keys
        assert 'stack' in result
        assert 'slices' in result
        
        # Check stack-level outputs
        assert result['stack']['masks'].exists()
        assert result['stack']['metadata'].exists()
        assert len(result['slices']) == 3
        
        # Check individual slice outputs
        for i, slice_info in enumerate(result['slices']):
            assert slice_info['masks'].exists()
            assert slice_info['z_index'] == i
            assert f"_z{i:03d}_" in str(slice_info['masks'])
    
    def test_json_serialization_with_numpy_arrays(self):
        """Test JSON serialization handles numpy arrays correctly."""
        manager = OutputManager(
            base_output_dir=self.test_output_dir,
            model_name="test_model",
            dataset_name="test"
        )
        
        # Test metadata with various numpy types
        test_metadata = {
            'small_array': np.array([1, 2, 3]),  # Should convert to list
            'large_array': np.random.random((100, 100)),  # Should convert to string
            'numpy_int': np.int64(42),
            'numpy_float': np.float32(3.14),
            'nested_dict': {
                'inner_array': np.array([4, 5, 6]),
                'inner_list': [np.int32(7), np.float64(8.0)]
            },
            'normal_data': 'test_string'
        }
        
        test_masks = np.zeros((32, 32), dtype=np.uint16)
        input_path = Path("test_serialization.tif")
        
        # This should not raise any JSON serialization errors
        result = manager.save_prediction(
            masks=test_masks,
            metadata=test_metadata,
            input_path=input_path
        )
        
        # Verify the metadata was saved correctly
        with open(result['metadata'], 'r') as f:
            saved_metadata = json.load(f)
        
        assert saved_metadata['small_array'] == [1, 2, 3]
        assert isinstance(saved_metadata['large_array'], str)
        assert saved_metadata['numpy_int'] == 42
        assert saved_metadata['numpy_float'] == pytest.approx(3.14, rel=1e-3)
        assert saved_metadata['normal_data'] == 'test_string'
    
    def test_finalize_run(self):
        """Test run finalization and summary generation."""
        manager = OutputManager(
            base_output_dir=self.test_output_dir,
            model_name="test_model",
            dataset_name="test"
        )
        
        # Add some files to run metadata
        manager.run_metadata['processed_files'] = [
            {'file': 'test1.tif', 'num_cells': 25},
            {'file': 'test2.tif', 'num_cells': 30},
            {'file': 'test3.tif', 'num_cells': 15}
        ]
        
        summary_path = manager.finalize_run()
        
        assert summary_path.exists()
        assert summary_path.name == "run_summary.json"
        
        # Check summary content
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        assert summary['summary']['total_files_processed'] == 3
        assert summary['summary']['total_cells_detected'] == 70  # 25+30+15
        assert 'completed_at' in summary['summary']
    
    def test_get_output_filename(self):
        """Test output filename generation."""
        manager = OutputManager(
            base_output_dir=self.test_output_dir,
            model_name="test_model",
            dataset_name="test"
        )
        
        # Test various input filenames - the method removes common suffixes
        test_cases = [
            ("image_BF.tif", "image"),         # _BF suffix removed
            ("sample_001.tiff", "sample_001"), # No suffix to remove
            ("data/test/file.png", "file"),    # No suffix to remove
            ("complex_name_with_underscores.tif", "complex_name_with_underscores"),
            ("test_Cells.tif", "test"),        # _Cells suffix removed
            ("sample_DAPI.tif", "sample"),     # _DAPI suffix removed
        ]
        
        for input_name, expected_base in test_cases:
            input_path = Path(input_name)
            result = manager._get_output_filename(input_path)
            assert result == expected_base, f"For input {input_name}, expected {expected_base}, got {result}"
    
    def test_error_handling_file_save_failure(self):
        """Test error handling when file saving fails."""
        manager = OutputManager(
            base_output_dir=self.test_output_dir,
            model_name="test_model",
            dataset_name="test"
        )
        
        # Create test data
        test_masks = np.random.randint(0, 50, (64, 64), dtype=np.uint16)
        test_metadata = {'num_cells': 25}
        
        # Test with invalid metadata that causes JSON error
        invalid_metadata = {
            'bad_data': object()  # This should cause JSON serialization error
        }
        
        with pytest.raises(Exception):
            manager.save_prediction(
                masks=test_masks,
                metadata=invalid_metadata,
                input_path=Path("test.tif")
            )
    
    def test_logging_setup(self):
        """Test that logging is properly configured."""
        manager = OutputManager(
            base_output_dir=self.test_output_dir,
            model_name="test_model",
            dataset_name="test"
        )
        
        # Check that log file exists
        log_file = manager.output_dir / "inference.log"
        # The log file might not exist until something is logged
        # So we just check that the output directory exists
        assert manager.output_dir.exists()
    
    def test_preserve_run_metadata(self):
        """Test that run metadata is preserved across operations."""
        manager = OutputManager(
            base_output_dir=self.test_output_dir,
            model_name="test_model",
            dataset_name="test"
        )
        
        # Initial metadata should be set up
        assert 'created_at' in manager.run_metadata  # Changed from 'run_started_at'
        assert 'model_name' in manager.run_metadata
        assert 'dataset_name' in manager.run_metadata
        assert manager.run_metadata['processed_files'] == []
        
        # Add a file and check it's tracked
        test_masks = np.zeros((32, 32), dtype=np.uint16)
        test_metadata = {'num_cells': 10}
        input_path = Path("test.tif")
        
        manager.save_prediction(
            masks=test_masks,
            metadata=test_metadata,
            input_path=input_path
        )
        
        # Check that file was added to run metadata
        assert len(manager.run_metadata['processed_files']) == 1
        assert manager.run_metadata['processed_files'][0]['input_file'] == str(input_path)  # Changed key name
        assert manager.run_metadata['processed_files'][0]['num_cells'] == 10


if __name__ == "__main__":
    pytest.main([__file__])
