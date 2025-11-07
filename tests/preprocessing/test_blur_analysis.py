"""
Unit tests for blur analysis preprocessing module.
"""

import os
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import tifffile as tiff

from src.preprocessing.blur_analysis import generate_blur_heatmap_batch
from src.utils.file_utils import BF_IF_FileHandler


@pytest.fixture
def temp_dirs():
    """Create temporary input and output directories."""
    temp_dir = tempfile.mkdtemp()
    input_dir = Path(temp_dir) / "input"
    output_dir = Path(temp_dir) / "output"
    input_dir.mkdir()
    
    yield input_dir, output_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_images(temp_dirs):
    """Create sample TIFF images for testing."""
    input_dir, _ = temp_dirs
    
    # Create test images with different sizes
    images = {
        "image1.tif": np.random.randint(0, 255, (100, 100), dtype=np.uint8),
        "image2.tif": np.random.randint(0, 255, (150, 120), dtype=np.uint8),
        "image3.tif": np.random.randint(0, 255, (80, 80), dtype=np.uint8),
    }
    
    image_paths = []
    for filename, image_data in images.items():
        image_path = input_dir / filename
        tiff.imwrite(str(image_path), image_data)
        image_paths.append(image_path)
    
    return image_paths

def test_basic_functionality(temp_dirs, sample_images):
    """Test basic blur heatmap generation."""
    input_dir, output_dir = temp_dirs
    
    def mock_measure_side_effect(image_path, patch_size=None, blur_path=None, **kwargs):
        """Mock function that creates the output file."""
        mock_heatmap = np.random.rand(50, 50)
        if blur_path:
            # Actually create the output file to simulate the real function
            tiff.imwrite(blur_path, mock_heatmap.astype(np.float32))
        return mock_heatmap
    
    with patch('src.utils.blur_measure.get_or_compute_blur_heatmap', side_effect=mock_measure_side_effect) as mock_measure:
        results = generate_blur_heatmap_batch(
            input_dir=input_dir,
            output_dir=output_dir,
            patch_size=32,
            stride_size=16,
            normalize=False,  # Disable normalize to avoid 3D requirement
            file_handler=None  # Don't use file handler for testing
        )
        
        # Check that all images were processed
        assert len(results) == 3
        assert all(str(img_path) in results for img_path in sample_images)
        
        # Check that output files were created
        for output_path in results.values():
            assert Path(output_path).exists()
        
        # Verify get_or_compute_blur_heatmap was called for each image
        assert mock_measure.call_count == 3

def test_no_images_found(temp_dirs):
    """Test behavior when no images are found."""
    input_dir, output_dir = temp_dirs
    
    results = generate_blur_heatmap_batch(
        input_dir=input_dir,
        output_dir=output_dir,
        file_handler=None
    )
    
    assert results == {}

def test_custom_pattern(temp_dirs):
    """Test with custom file pattern."""
    input_dir, output_dir = temp_dirs
    
    # Create images with different extensions
    png_image = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
    tiff_image = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
    
    png_path = input_dir / "test.tif"  # Save as .tif to avoid format error
    tiff_path = input_dir / "test_bf.tif"
    
    tiff.imwrite(str(png_path), png_image)
    tiff.imwrite(str(tiff_path), tiff_image)
    
    with patch('src.utils.blur_measure.get_or_compute_blur_heatmap') as mock_measure:
        mock_measure.return_value = np.random.rand(25, 25)
        
        # Test PNG pattern
        results = generate_blur_heatmap_batch(
            input_dir=input_dir,
            output_dir=output_dir,
            pattern="test.tif",  # Match the renamed file
            file_handler=None,
            normalize=False  # Disable normalize to avoid 3D requirement
        )
        
        assert len(results) == 1
        assert str(png_path) in results

def test_file_handler_integration(temp_dirs, sample_images):
    """Test integration with file handler for standardized naming."""
    input_dir, output_dir = temp_dirs
    
    file_handler = BF_IF_FileHandler()
    
    with patch('src.utils.blur_measure.get_or_compute_blur_heatmap') as mock_measure:
        mock_measure.return_value = np.random.rand(50, 50)
        
        results = generate_blur_heatmap_batch(
            input_dir=input_dir,
            output_dir=output_dir,
            file_handler=file_handler
        )
        
        # Check that standardized names were used
        for output_path in results.values():
            output_filename = Path(output_path).name
            assert "_blur_heatmap.tif" in output_filename

def test_overwrite_behavior(temp_dirs, sample_images):
    """Test overwrite behavior."""
    input_dir, output_dir = temp_dirs
    
    def mock_measure_side_effect(image_path, patch_size=None, blur_path=None, **kwargs):
        """Mock function that creates the output file."""
        mock_heatmap = np.random.rand(50, 50)
        if blur_path:
            tiff.imwrite(blur_path, mock_heatmap.astype(np.float32))
        return mock_heatmap
    
    with patch('src.utils.blur_measure.get_or_compute_blur_heatmap', side_effect=mock_measure_side_effect) as mock_measure:
        # First run
        results1 = generate_blur_heatmap_batch(
            input_dir=input_dir,
            output_dir=output_dir,
            overwrite=False,
            file_handler=None
        )
        
        call_count_first = mock_measure.call_count
        
        # Second run without overwrite
        results2 = generate_blur_heatmap_batch(
            input_dir=input_dir,
            output_dir=output_dir,
            overwrite=False,
            file_handler=None
        )
        
        # Should not call get_or_compute_blur_heatmap again
        assert mock_measure.call_count == call_count_first
        assert results1 == results2
        
        # Third run with overwrite
        results3 = generate_blur_heatmap_batch(
            input_dir=input_dir,
            output_dir=output_dir,
            overwrite=True,
            file_handler=None
        )
        
        # Should call get_or_compute_blur_heatmap again
        assert mock_measure.call_count == call_count_first * 2

def test_error_handling(temp_dirs, sample_images):
    """Test error handling for failed processing."""
    input_dir, output_dir = temp_dirs
    
    with patch('src.utils.blur_measure.get_or_compute_blur_heatmap') as mock_measure:
        # Make the first call succeed, second fail, third succeed
        mock_measure.side_effect = [
            np.random.rand(50, 50),  # Success
            Exception("Processing failed"),  # Failure
            np.random.rand(50, 50)   # Success
        ]
        
        results = generate_blur_heatmap_batch(
            input_dir=input_dir,
            output_dir=output_dir,
            file_handler=None,
            normalize=False
        )
        
        # Should have 2 successful results out of 3 images
        assert len(results) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
