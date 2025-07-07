"""
Tests for the 3D cell tracking processor.

This module tests the complete 3D cell tracking pipeline with blur filtering,
ensuring robust processing of segmentation masks and proper integration of
all components.
"""

import os
import tempfile
import shutil
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd
import tifffile

from src.postprocessing.tracking_processor import (
    TrackingProcessor,
    TrackingProcessorConfig,
    run_tracking_pipeline,
    main_compatible
)
from src.postprocessing import TrackingConfig, FilterConfig


class TestTrackingProcessorConfig:
    """Test the configuration class for tracking processor."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TrackingProcessorConfig()
        
        assert config.mask_pattern == "*_3d.tif"
        assert config.blur_threshold == 0.5
        assert config.invert_blur_threshold is False
        assert config.create_output_dirs is True
        assert config.overwrite_existing is False
        
    def test_custom_config(self):
        """Test custom configuration values."""
        tracking_config = TrackingConfig(search_range=10.0, memory=2)
        filter_config = FilterConfig(patch_size=64, blur_threshold=0.3)
        
        config = TrackingProcessorConfig(
            blur_threshold=0.3,
            invert_blur_threshold=True,
            tracking_config=tracking_config,
            filter_config=filter_config
        )
        
        assert config.blur_threshold == 0.3
        assert config.invert_blur_threshold is True
        assert config.tracking_config.search_range == 10.0
        assert config.filter_config.patch_size == 64


class TestTrackingProcessor:
    """Test the main tracking processor class."""
    
    @pytest.fixture
    def processor(self):
        """Create a processor for testing."""
        config = TrackingProcessorConfig(
            blur_threshold=0.4,
            create_output_dirs=True,
            overwrite_existing=True
        )
        return TrackingProcessor(config)
    
    @pytest.fixture
    def mock_segmentation_stack(self):
        """Create a mock 3D segmentation stack."""
        # Create a simple 3D stack with labeled regions
        stack = np.zeros((3, 50, 50), dtype=int)
        
        # Add some labeled regions in each slice
        stack[0, 10:20, 10:20] = 1
        stack[0, 30:40, 30:40] = 2
        stack[1, 12:22, 12:22] = 1  # Moved region 1
        stack[1, 32:42, 32:42] = 2  # Moved region 2
        stack[2, 14:24, 14:24] = 1  # Moved region 1 again
        stack[2, 34:44, 34:44] = 2  # Moved region 2 again
        
        return stack
    
    @pytest.fixture
    def mock_sharpness_image(self):
        """Create a mock sharpness image."""
        # Create sharpness values (higher = sharper)
        sharpness = np.random.rand(3, 50, 50) * 0.8 + 0.1  # Values between 0.1-0.9
        return sharpness
    
    def test_get_label_centers_with_blur_filtering(self, processor):
        """Test extraction of label centers with blur filtering."""
        # Create simple 2D mask
        mask = np.zeros((50, 50), dtype=int)
        mask[10:20, 10:20] = 1
        mask[30:40, 30:40] = 2
        
        # Create sharpness image
        sharpness = np.ones((50, 50)) * 0.8  # Above threshold (blurry)  
        sharpness[10:20, 10:20] = 0.3  # Below threshold (sharp)
        
        # Test without blur filtering
        centers = processor.get_label_centers_with_blur_filtering(mask)
        assert len(centers) == 2
        assert 'label' in centers.columns
        assert 'x' in centers.columns
        assert 'y' in centers.columns
        
        # Test with blur filtering (should keep region 1, filter out region 2)
        centers_filtered = processor.get_label_centers_with_blur_filtering(
            mask, sharpness, blur_thresh=0.5, inv=False
        )
        assert len(centers_filtered) == 1
        # Check that we kept the sharp cell (label 1)
        kept_labels = set(centers_filtered['label'])
        assert 1 in kept_labels
    
    def test_extract_3d_centers_with_blur(self, processor, mock_segmentation_stack, mock_sharpness_image):
        """Test extraction of 3D centers with blur filtering."""
        centers_with_z = processor.extract_3d_centers_with_blur(
            mock_segmentation_stack, mock_sharpness_image
        )
        
        assert len(centers_with_z) == 3  # Three z-slices
        
        # Check each slice
        for z, centers in centers_with_z:
            assert isinstance(z, int)
            assert isinstance(centers, pd.DataFrame)
            assert 0 <= z < 3
    
    def test_track_3d_centers(self, processor, mock_segmentation_stack):
        """Test 3D cell tracking."""
        tracked_stack = processor.track_3d_centers(mock_segmentation_stack)
        
        assert tracked_stack.shape == mock_segmentation_stack.shape
        assert tracked_stack.dtype == np.int32
        
        # Check that we have some tracked cells
        unique_ids = np.unique(tracked_stack)
        assert 0 in unique_ids  # Background
        assert len(unique_ids) > 1  # At least one tracked cell
    
    @patch('src.postprocessing.tracking_processor.measure_blur_heatmap')
    @patch('tifffile.imwrite')
    @patch('tifffile.imread')
    def test_get_or_create_blur_heatmap(self, mock_imread, mock_imwrite, mock_measure_blur, processor):
        """Test blur heatmap creation and caching."""
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "test_image.tif"
            blur_cache_dir = Path(temp_dir) / "blur_cache"
            
            # Mock blur measurement
            mock_blur_map = np.random.rand(50, 50).astype(np.float32)
            mock_measure_blur.return_value = mock_blur_map
            
            # Test creation (file doesn't exist)
            result = processor.get_or_create_blur_heatmap(image_path, blur_cache_dir)
            
            assert mock_measure_blur.called
            assert mock_imwrite.called
            np.testing.assert_array_equal(result, mock_blur_map)
            
            # Test caching (file exists)
            mock_measure_blur.reset_mock()
            mock_imwrite.reset_mock()
            mock_imread.return_value = mock_blur_map
            
            # Mock that file exists
            with patch('pathlib.Path.exists', return_value=True):
                result = processor.get_or_create_blur_heatmap(image_path, blur_cache_dir)
            
            assert not mock_measure_blur.called  # Should use cache
            assert mock_imread.called
    
    @patch('tifffile.imread')
    @patch('tifffile.imwrite')
    @patch.object(TrackingProcessor, 'get_or_create_blur_heatmap')
    def test_process_single_file(self, mock_blur_heatmap, mock_imwrite, mock_imread, processor):
        """Test processing of a single file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set up directories
            mask_dir = Path(temp_dir) / "masks"
            image_dir = Path(temp_dir) / "images" 
            output_dir = Path(temp_dir) / "output"
            blur_dir = Path(temp_dir) / "blur"
            
            mask_dir.mkdir()
            image_dir.mkdir()
            
            # Create test files
            mask_path = mask_dir / "test_3d.tif"
            
            # Mock file reading
            mock_segmentation = np.zeros((3, 50, 50), dtype=int)
            mock_segmentation[0, 10:20, 10:20] = 1
            mock_imread.return_value = mock_segmentation
            
            mock_blur_heatmap.return_value = np.random.rand(3, 50, 50)
            
            # Process file
            result = processor.process_single_file(
                mask_path, image_dir, output_dir, blur_dir
            )
            
            assert result['status'] == 'success'
            assert 'output_path' in result
            assert result['input_shape'] == (3, 50, 50)
            assert mock_imwrite.called
    
    @patch('pathlib.Path.glob')
    @patch.object(TrackingProcessor, 'process_single_file')
    def test_process_batch(self, mock_process_single, mock_glob, processor):
        """Test batch processing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock file discovery
            mock_files = [
                Path(temp_dir) / "file1_3d.tif",
                Path(temp_dir) / "file2_3d.tif"
            ]
            mock_glob.return_value = mock_files
            
            # Mock single file processing
            mock_process_single.side_effect = [
                {'status': 'success', 'input_path': str(mock_files[0])},
                {'status': 'success', 'input_path': str(mock_files[1])}
            ]
            
            # Run batch processing
            result = processor.process_batch(
                temp_dir, temp_dir, temp_dir, temp_dir
            )
            
            assert result['total_files'] == 2
            assert result['successful'] == 2
            assert result['failed'] == 0
            assert result['success_rate'] == 1.0
            assert mock_process_single.call_count == 2


class TestHighLevelFunctions:
    """Test high-level convenience functions."""
    
    @patch.object(TrackingProcessor, 'process_batch')
    def test_run_tracking_pipeline(self, mock_process_batch):
        """Test the high-level pipeline runner."""
        mock_process_batch.return_value = {
            'total_files': 2,
            'successful': 2,
            'failed': 0,
            'success_rate': 1.0
        }
        
        result = run_tracking_pipeline(
            mask_directory="masks",
            image_directory="images", 
            output_directory="output",
            blur_directory="blur"
        )
        
        assert result['successful'] == 2
        assert mock_process_batch.called
    
    @patch('src.postprocessing.tracking_processor.run_tracking_pipeline')
    def test_main_compatible(self, mock_run_pipeline):
        """Test the compatibility function."""
        mock_run_pipeline.return_value = {'successful': 1, 'total_files': 1}
        
        result = main_compatible(
            blur_thresh=0.3,
            inv=True
        )
        
        assert mock_run_pipeline.called
        call_args = mock_run_pipeline.call_args
        config = call_args.kwargs['config']
        assert config.blur_threshold == 0.3
        assert config.invert_blur_threshold is True


class TestIntegration:
    """Integration tests with real data."""
    
    def create_test_data(self, temp_dir):
        """Create realistic test data."""
        # Create 3D segmentation stack
        segmentation = np.zeros((5, 100, 100), dtype=int)
        
        # Add some cells that move across frames more consistently
        for z in range(5):
            # Cell 1: moves from (20, 20) to (30, 30)
            x1, y1 = 20 + z*2, 20 + z*2
            segmentation[z, y1:y1+10, x1:x1+10] = 1
            
            # Cell 2: moves from (60, 60) to (50, 50) 
            x2, y2 = 60 - z*2, 60 - z*2
            segmentation[z, y2:y2+15, x2:x2+15] = 2
            
            # Cell 3: stationary in center
            segmentation[z, 45:55, 45:55] = 3
        
        # Create corresponding image and blur data
        image_data = np.random.randint(0, 255, (5, 100, 100), dtype=np.uint8)
        blur_data = np.random.rand(5, 100, 100).astype(np.float32)
        
        return segmentation, image_data, blur_data
    
    @patch('src.postprocessing.tracking_processor.measure_blur_heatmap')
    def test_end_to_end_processing(self, mock_measure_blur):
        """Test end-to-end processing with realistic data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set up directory structure
            dirs = {
                'masks': Path(temp_dir) / "masks",
                'images': Path(temp_dir) / "images", 
                'output': Path(temp_dir) / "output",
                'blur': Path(temp_dir) / "blur"
            }
            
            for d in dirs.values():
                d.mkdir(parents=True, exist_ok=True)
            
            # Create test data
            segmentation, image_data, blur_data = self.create_test_data(temp_dir)
            
            # Save test files
            mask_path = dirs['masks'] / "test_3d.tif"
            image_path = dirs['images'] / "test_BF_3d.tif"
            
            tifffile.imwrite(str(mask_path), segmentation)
            tifffile.imwrite(str(image_path), image_data)
            
            # Mock blur measurement
            mock_measure_blur.return_value = blur_data
            
            # Create processor and run
            config = TrackingProcessorConfig(
                blur_threshold=1.0,  # Accept all cells (no blur filtering)
                create_output_dirs=True,
                overwrite_existing=True,
                tracking_config=TrackingConfig(
                    min_track_length=1,  # Accept single-frame tracks
                    search_range=10.0,   # Larger search range
                    memory=2             # More memory for linking
                )
            )
            processor = TrackingProcessor(config)
            
            result = processor.process_batch(
                dirs['masks'], dirs['images'], dirs['output'], dirs['blur']
            )
            
            # Verify results
            assert result['successful'] == 1
            assert result['failed'] == 0
            
            # Check output file exists
            output_files = list(dirs['output'].glob("*.tif"))
            assert len(output_files) == 1
            
            # Load and verify output
            tracked_result = tifffile.imread(str(output_files[0]))
            assert tracked_result.shape == segmentation.shape
            assert tracked_result.dtype == np.int32
            
            # Should have tracked some cells
            unique_ids = np.unique(tracked_result)
            assert len(unique_ids) > 1  # Background + at least one cell


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
