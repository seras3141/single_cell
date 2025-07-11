"""
Tests for the 3D cell tracking pipeline (unified postprocessing).

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
    CellTrackingPipeline,
    PostprocessingConfig
)
from src.postprocessing import TrackingConfig, FilterConfig


class TestPipelineConfig:
    """Test the configuration class for the unified pipeline."""
    
    def test_default_config(self):
        config = PostprocessingConfig()
        
        assert config.enable_blur_filtering is True
        assert config.filter_before_tracking is True
        assert config.save_intermediate_results is False
        
    def test_custom_config(self):
        tracking_config = TrackingConfig(search_range=10.0, memory=2)
        filter_config = FilterConfig(patch_size=64, blur_threshold=0.3)
        
        config = PostprocessingConfig(
            enable_blur_filtering=False,
            filter_before_tracking=False,
            save_intermediate_results=True,
            tracking_config=tracking_config,
            filter_config=filter_config
        )
        
        assert config.enable_blur_filtering is False
        assert config.filter_before_tracking is False
        assert config.save_intermediate_results is True
        assert config.tracking_config.search_range == 10.0
        assert config.filter_config.patch_size == 64


class TestCellTrackingPipeline:
    """Test the main cell tracking pipeline class."""
    
    @pytest.fixture
    def pipeline(self):
        config = PostprocessingConfig(
            enable_blur_filtering=True,
            filter_before_tracking=True,
            save_intermediate_results=False
        )
        return CellTrackingPipeline(config)
    
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
    def mock_image_stack(self):
        """Create a mock 3D image stack."""
        return np.random.randint(0, 255, (3, 50, 50), dtype=np.uint8)
    
    def test_process_single_file(self, pipeline, tmp_path, mock_segmentation_stack, mock_image_stack):
        """Test processing of a single file."""
        seg_path = tmp_path / "test_3d.tif"
        img_path = tmp_path / "test_BF_3d.tif"
        tifffile.imwrite(str(seg_path), mock_segmentation_stack)
        tifffile.imwrite(str(img_path), mock_image_stack)
        output_dir = tmp_path / "output"
        
        # Process file
        result = pipeline.process_single_file(seg_path, img_path, output_dir)
        
        assert 'final_output' in result
        assert Path(result['final_output']).exists()
    
    def test_process_batch(self, pipeline, tmp_path, mock_segmentation_stack, mock_image_stack):
        """Test batch processing."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        for i in range(2):
            seg_path = input_dir / f"sample_{i}_masks_3d.tif"
            img_path = input_dir / f"sample_{i}_BF_3d.tif"
            tifffile.imwrite(str(seg_path), mock_segmentation_stack)
            tifffile.imwrite(str(img_path), mock_image_stack)
        output_dir = tmp_path / "output"
        
        # Run batch processing
        results = pipeline.process_batch(input_dir, input_dir, output_dir)
        
        assert len(results) == 2
        for r in results:
            assert 'final_output' in r
            assert Path(r['final_output']).exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
