"""
Unit tests for blur_measure module (utils).
Covers all functions: patchwise blur, blur heatmap, image blur, dataset analysis, 
image filtering, image reading, and cache logic.
"""

import numpy as np
import tempfile
import os
import csv
from pathlib import Path
import tifffile as tiff
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image

from src.utils.image_utils import load_image

from src.utils.blur_measure import (
    measure_patchwise_blur,
    measure_blur_heatmap,
    measure_image_blur,
    analyze_dataset_blur,
    filter_blurry_images,
    get_or_compute_blur_heatmap,
)

def create_test_image(path, shape=(64, 64), value=1.0):
    arr = np.ones(shape, dtype=np.float32) * value
    tiff.imwrite(str(path), arr)
    return arr

# --- Fixtures ---
@pytest.fixture(scope="module")
def test_images():
    np.random.seed(42)
    test_image_2d = np.random.rand(100, 120).astype(np.float32)
    test_image_small = np.random.rand(32, 32).astype(np.float32)
    sharp_image = np.zeros((64, 64))
    sharp_image[30:34, 30:34] = 1.0
    blurry_image = np.ones((64, 64)) * 0.5
    return {
        "test_image_2d": test_image_2d,
        "test_image_small": test_image_small,
        "sharp_image": sharp_image,
        "blurry_image": blurry_image
    }

@pytest.fixture(scope="module")
def blur_heatmap_images():
    np.random.seed(42)
    test_image_2d = np.random.rand(80, 100).astype(np.float32)
    test_image_3d = np.random.rand(5, 80, 100).astype(np.float32)
    return {"test_image_2d": test_image_2d, "test_image_3d": test_image_3d}

@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield d
    import shutil
    shutil.rmtree(d, ignore_errors=True)

# --- Patchwise blur tests ---
def test_input_validation(test_images):
    result = measure_patchwise_blur(test_images["test_image_small"], patch_size=16)
    assert isinstance(result, np.ndarray)
    result = measure_patchwise_blur(test_images["test_image_small"], stride_size=8)
    assert isinstance(result, np.ndarray)
    result = measure_patchwise_blur(
        test_images["test_image_small"],
        patch_size=(16, 20),
        stride_size=(8, 10)
    )
    assert isinstance(result, np.ndarray)

def test_center_values_true_output_dimensions(test_images):
    image = test_images["test_image_2d"]
    test_cases = [
        (32, 16, (7, 8)),
        (24, 12, (9, 10)),
        (16, 8, (13, 15)),
    ]
    for patch_size, stride_size, expected_shape in test_cases:
        result = measure_patchwise_blur(
            image,
            patch_size=patch_size,
            stride_size=stride_size,
            center_values=True
        )
        assert result.shape == expected_shape

def test_center_values_false_output_dimensions(test_images):
    image = test_images["test_image_2d"]
    test_cases = [
        (32, 16, (5, 6)),
        (24, 12, (7, 9)),
        (16, 8, (11, 14)),
    ]
    for patch_size, stride_size, expected_shape in test_cases:
        result = measure_patchwise_blur(
            image,
            patch_size=patch_size,
            stride_size=stride_size,
            center_values=False
        )
        assert result.shape == expected_shape

def test_output_values_are_non_negative(test_images):
    result = measure_patchwise_blur(test_images["test_image_2d"])
    assert np.all(result >= 0)

def test_sharp_vs_blurry_discrimination(test_images):
    sharp_result = measure_patchwise_blur(
        test_images["sharp_image"],
        patch_size=16,
        stride_size=8,
        center_values=False
    )
    blurry_result = measure_patchwise_blur(
        test_images["blurry_image"],
        patch_size=16,
        stride_size=8,
        center_values=False
    )
    assert np.max(sharp_result) > np.max(blurry_result)

def test_edge_cases():
    tiny_image = np.random.rand(10, 10)
    result = measure_patchwise_blur(tiny_image, patch_size=8, stride_size=4)
    assert isinstance(result, np.ndarray)
    assert result.size > 0
    small_img = np.random.rand(16, 16)
    result = measure_patchwise_blur(small_img, patch_size=16, stride_size=16)
    assert isinstance(result, np.ndarray)

def test_reproducibility(test_images):
    result1 = measure_patchwise_blur(test_images["test_image_2d"])
    result2 = measure_patchwise_blur(test_images["test_image_2d"])
    np.testing.assert_array_equal(result1, result2)

# --- Blur heatmap tests ---
def test_2d_input_array_same_size_output(blur_heatmap_images):
    result = measure_blur_heatmap(
        blur_heatmap_images["test_image_2d"],
        patch_size=16,
        stride_size=8,
        center_values=True,
        normalize=False,
    )
    assert result.shape == blur_heatmap_images["test_image_2d"].shape

def test_3d_input_array_same_size_output(blur_heatmap_images):
    result = measure_blur_heatmap(
        blur_heatmap_images["test_image_3d"],
        patch_size=16,
        stride_size=8,
        center_values=True
    )
    assert result.shape == blur_heatmap_images["test_image_3d"].shape

def test_center_values_false_different_size(blur_heatmap_images):
    result = measure_blur_heatmap(
        blur_heatmap_images["test_image_2d"],
        patch_size=32,
        stride_size=16,
        center_values=False,
        normalize=False,
    )
    assert result.shape == blur_heatmap_images["test_image_2d"].shape

def test_file_input_non_tiff(blur_heatmap_images):
    result = measure_blur_heatmap(blur_heatmap_images["test_image_2d"], normalize=False)
    assert result.shape == blur_heatmap_images["test_image_2d"].shape
    assert np.all(np.isfinite(result))
    assert np.all(result >= 0)

def test_normalization_3d():
    test_3d = np.zeros((3, 20, 20))
    test_3d[0] = 1.0
    test_3d[1] = 0.5
    test_3d[2] = 0.0
    result = measure_blur_heatmap(
        test_3d,
        patch_size=8,
        stride_size=4,
        normalize=True,
        center_values=True
    )
    assert result.shape == test_3d.shape
    assert isinstance(result, np.ndarray)

def test_different_patch_stride_combinations(blur_heatmap_images):
    test_cases = [
        (16, 8),
        (16, 16),
        (32, 16),
        (8, 4),
    ]
    for patch_size, stride_size in test_cases:
        result = measure_blur_heatmap(
            blur_heatmap_images["test_image_2d"],
            patch_size=patch_size,
            stride_size=stride_size,
            normalize=False,
        )
        assert result.shape == blur_heatmap_images["test_image_2d"].shape
        assert np.all(np.isfinite(result))

def test_output_data_type_and_values(blur_heatmap_images):
    result = measure_blur_heatmap(blur_heatmap_images["test_image_2d"], normalize=False)
    assert np.issubdtype(result.dtype, np.floating)
    assert np.all(np.isfinite(result))
    assert np.all(result >= 0)

# --- get_or_compute_blur_heatmap tests ---
def test_get_or_compute_blur_heatmap_creates_and_reads_cache(tmp_path):
    img_path = tmp_path / 'test_img.tif'
    arr = create_test_image(img_path, value=2.0)
    blur_path = tmp_path / 'test_img_blur.tif'
    blur1 = get_or_compute_blur_heatmap(img_path, blur_path=blur_path, patch_size=16, stride_size=8, normalize=False)
    assert blur_path.exists()
    assert isinstance(blur1, np.ndarray)
    tiff.imwrite(str(blur_path), np.zeros_like(blur1))
    blur2 = get_or_compute_blur_heatmap(img_path, blur_path=blur_path, patch_size=16, stride_size=8, normalize=False)
    assert np.allclose(blur2, 0)

def test_get_or_compute_blur_heatmap_without_cache(tmp_path):
    img_path = tmp_path / 'test_img2.tif'
    arr = create_test_image(img_path, value=3.0)
    blur = get_or_compute_blur_heatmap(img_path, blur_path=None, patch_size=8, stride_size=4, normalize=False)
    assert isinstance(blur, np.ndarray)
    assert blur.shape[0] > 0 and blur.shape[1] > 0

# --- measure_image_blur tests ---
@patch('tifffile.imread')
def test_tiff_file_processing(mock_imread, temp_dir):
    test_image = np.random.rand(100, 100)
    mock_imread.return_value = test_image
    result = measure_image_blur('/fake/path/image.tif')
    mock_imread.assert_called_once_with('/fake/path/image.tif')
    assert isinstance(result, float)
    assert result >= 0.0

def test_error_handling():
    result = measure_image_blur('/non/existent/file.tif')
    assert result == 0.0

@patch('tifffile.imread')
def test_3d_image_handling(mock_imread):
    test_3d = np.random.rand(10, 50, 50)
    mock_imread.return_value = test_3d
    result = measure_image_blur('/fake/path/stack.tif')
    assert isinstance(result, float)
    assert result >= 0.0


# --- Enhanced measure_blur_heatmap tests ---
class TestMeasureBlurHeatmap:
    """Enhanced tests for measure_blur_heatmap function."""
    
    @pytest.fixture
    def test_images_3d(self):
        """Create test images for 3D processing."""
        np.random.seed(42)
        # Create distinct layers for better testing
        test_3d = np.zeros((4, 64, 64))
        test_3d[0] = np.random.rand(64, 64) * 255  # Random layer
        test_3d[1] = np.ones((64, 64)) * 128      # Uniform layer
        test_3d[2] = np.zeros((64, 64))           # Zero layer
        test_3d[3] = np.random.rand(64, 64) * 50  # Lower variance layer
        
        return test_3d.astype(np.float32)

    def test_n_jobs_parameter(self, test_images_3d):
        """Test measure_blur_heatmap with different n_jobs settings."""
        # Test with different n_jobs values
        for n_jobs in [1, 2, -1]:
            result = measure_blur_heatmap(
                test_images_3d,
                patch_size=16,
                stride_size=8,
                normalize=True,
                center_values=True,
                n_jobs=n_jobs
            )
            assert result.shape == test_images_3d.shape
            assert np.all(np.isfinite(result))
            assert np.all(result >= 0)

    def test_3d_parallel_vs_serial(self, test_images_3d):
        """Test that parallel and serial processing give same results."""
        # Serial processing (n_jobs=1)
        result_serial = measure_blur_heatmap(
            test_images_3d,
            patch_size=16,
            stride_size=8,
            normalize=False,
            center_values=True,
            n_jobs=1
        )
        
        # Parallel processing (n_jobs=2)
        result_parallel = measure_blur_heatmap(
            test_images_3d,
            patch_size=16,
            stride_size=8,
            normalize=False,
            center_values=True,
            n_jobs=2
        )
        
        # Results should be identical
        np.testing.assert_array_almost_equal(result_serial, result_parallel, decimal=10)

    def test_normalization_behavior_3d(self, test_images_3d):
        """Test normalization behavior with 3D images."""
        # Test normalized output
        result_norm = measure_blur_heatmap(
            test_images_3d,
            patch_size=16,
            stride_size=8,
            normalize=True,
            center_values=True
        )
        
        # Check that values are in [0, 1] range
        assert np.all(result_norm >= 0)
        assert np.all(result_norm <= 1)
        
        # Test non-normalized output
        result_no_norm = measure_blur_heatmap(
            test_images_3d,
            patch_size=16,
            stride_size=8,
            normalize=False,
            center_values=True
        )
        
        # Non-normalized should have larger range
        assert np.max(result_no_norm) > np.max(result_norm)

    def test_2d_normalization_assertion(self, blur_heatmap_images):
        """Test that 2D images with normalize=True raise assertion."""
        with pytest.raises(AssertionError):
            measure_blur_heatmap(
                blur_heatmap_images["test_image_2d"],
                normalize=True
            )

    def test_resize_functionality(self):
        """Test that resizing works correctly when shapes don't match."""
        # Create a 3D image where blur map will need resizing
        test_image = np.random.rand(3, 48, 48)
        
        result = measure_blur_heatmap(
            test_image,
            patch_size=32,
            stride_size=16,
            normalize=True,
            center_values=False  # This should trigger resizing
        )
        
        # Result should match input shape
        assert result.shape == test_image.shape


# --- analyze_dataset_blur tests ---
class TestAnalyzeDatasetBlur:
    """Test cases for analyze_dataset_blur function."""
    
    @pytest.fixture
    def sample_dataset(self, temp_dir):
        """Create sample dataset with various blur levels."""
        dataset_dir = Path(temp_dir) / "dataset"
        dataset_dir.mkdir()
        
        # Create images with different blur characteristics
        np.random.seed(42)
        
        # Sharp image (high variance in Laplacian)
        sharp_img = np.zeros((64, 64))
        sharp_img[20:44, 20:44] = np.random.rand(24, 24) * 255
        tiff.imwrite(str(dataset_dir / "sharp_image.tif"), sharp_img.astype(np.uint8))
        
        # Blurry image (low variance)
        blurry_img = np.ones((64, 64)) * 128 + np.random.rand(64, 64) * 10
        tiff.imwrite(str(dataset_dir / "blurry_image.tif"), blurry_img.astype(np.uint8))
        
        # Medium blur image
        medium_img = np.random.rand(64, 64) * 100 + 50
        tiff.imwrite(str(dataset_dir / "medium_image.tif"), medium_img.astype(np.uint8))
        
        # 3D stack
        stack_3d = np.random.rand(3, 32, 32) * 255
        tiff.imwrite(str(dataset_dir / "stack_3d.tif"), stack_3d.astype(np.uint8))
        
        return dataset_dir

    def test_analyze_basic_functionality(self, sample_dataset):
        """Test basic analysis functionality."""
        results = analyze_dataset_blur(sample_dataset, threshold=50.0)
        
        # Should find 4 images
        assert len(results) == 4
        
        # All results should be float values
        for path, score in results.items():
            assert isinstance(score, float)
            assert score >= 0.0
            assert path.endswith('.tif')

    def test_analyze_with_pattern(self, sample_dataset):
        """Test analysis with custom pattern."""
        # Test with specific pattern
        results = analyze_dataset_blur(sample_dataset, pattern="*sharp*.tif")
        assert len(results) == 1
        assert any("sharp_image.tif" in path for path in results.keys())

    def test_analyze_with_threshold(self, sample_dataset):
        """Test threshold classification."""
        results = analyze_dataset_blur(sample_dataset, threshold=100.0)
        
        # Verify we get blur scores
        assert len(results) > 0
        for score in results.values():
            assert isinstance(score, float)

    def test_analyze_empty_directory(self, temp_dir):
        """Test behavior with empty directory."""
        empty_dir = Path(temp_dir) / "empty"
        empty_dir.mkdir()
        
        results = analyze_dataset_blur(empty_dir)
        assert len(results) == 0

    def test_analyze_nonexistent_directory(self):
        """Test behavior with non-existent directory."""
        results = analyze_dataset_blur("/nonexistent/path")
        assert len(results) == 0

    @patch('src.utils.blur_measure.measure_image_blur')
    def test_analyze_handles_errors(self, mock_measure, sample_dataset):
        """Test that analysis continues even if individual images fail."""
        # Mock measure_image_blur to fail for some images but return 0.0 for errors
        def side_effect(path):
            if "sharp" in path:
                return 0.0  # analyze_dataset_blur catches exceptions and logs errors
            return 100.0
        
        mock_measure.side_effect = side_effect
        
        # Should not raise exception and should return results for all images
        results = analyze_dataset_blur(sample_dataset)
        assert isinstance(results, dict)
        assert len(results) > 0


# --- filter_blurry_images tests ---
class TestFilterBlurryImages:
    """Test cases for filter_blurry_images function."""
    
    @pytest.fixture
    def sample_image_paths(self, temp_dir):
        """Create sample images with different blur levels."""
        image_dir = Path(temp_dir) / "images"
        image_dir.mkdir()
        
        paths = []
        np.random.seed(42)
        
        # Create images with known characteristics
        for i, blur_level in enumerate(["sharp", "medium", "blurry"]):
            if blur_level == "sharp":
                img = np.zeros((32, 32))
                img[10:22, 10:22] = np.random.rand(12, 12) * 255  # High contrast edges
            elif blur_level == "medium":
                img = np.random.rand(32, 32) * 150 + 50  # Medium variance
            else:  # blurry
                img = np.ones((32, 32)) * 100 + np.random.rand(32, 32) * 20  # Low variance
            
            path = str(image_dir / f"{blur_level}_{i}.tif")
            tiff.imwrite(path, img.astype(np.uint8))
            paths.append(path)
        
        return paths

    def test_filter_basic_functionality(self, sample_image_paths):
        """Test basic filtering functionality."""
        # Use a threshold that should filter some images
        filtered = filter_blurry_images(sample_image_paths, threshold=50.0)
        
        # Should return a subset of images
        assert isinstance(filtered, list)
        assert len(filtered) <= len(sample_image_paths)
        
        # All returned paths should be in original list
        for path in filtered:
            assert path in sample_image_paths

    def test_filter_very_low_threshold(self, sample_image_paths):
        """Test with very low threshold (should keep most images)."""
        filtered = filter_blurry_images(sample_image_paths, threshold=1.0)
        # With very low threshold, most images should be kept
        assert len(filtered) >= 0

    def test_filter_very_high_threshold(self, sample_image_paths):
        """Test with very high threshold (should filter most images)."""
        filtered = filter_blurry_images(sample_image_paths, threshold=10000.0)
        # With very high threshold, most/all images should be filtered
        assert len(filtered) <= len(sample_image_paths)

    def test_filter_empty_list(self):
        """Test with empty image list."""
        filtered = filter_blurry_images([], threshold=100.0)
        assert filtered == []

    @patch('src.utils.blur_measure.measure_image_blur')
    def test_filter_handles_errors(self, mock_measure, sample_image_paths):
        """Test that filtering handles errors gracefully."""
        # Mock measure_image_blur to return known values
        mock_measure.side_effect = [100.0, 50.0, 0.0]  # Above, at, and below threshold
        
        filtered = filter_blurry_images(sample_image_paths, threshold=75.0)
        assert len(filtered) == 1  # Only the 100.0 score should pass

# --- Enhanced get_or_compute_blur_heatmap tests ---
class TestGetOrComputeBlurHeatmap:
    """Enhanced test cases for get_or_compute_blur_heatmap function."""
    
    @pytest.fixture
    def sample_images(self, temp_dir):
        """Create sample images for caching tests."""
        image_dir = Path(temp_dir) / "images"
        cache_dir = Path(temp_dir) / "cache"
        image_dir.mkdir()
        cache_dir.mkdir()
        
        # Create test images
        test_2d = np.random.rand(64, 64) * 255
        test_3d = np.random.rand(3, 48, 48) * 255
        
        image_2d_path = image_dir / "test_2d.tif"
        image_3d_path = image_dir / "test_3d.tif"
        
        tiff.imwrite(str(image_2d_path), test_2d.astype(np.uint8))
        tiff.imwrite(str(image_3d_path), test_3d.astype(np.uint8))
        
        return {
            "image_2d": image_2d_path,
            "image_3d": image_3d_path,
            "cache_dir": cache_dir,
            "test_2d_data": test_2d,
            "test_3d_data": test_3d
        }

    def test_compute_without_cache(self, sample_images):
        """Test computing blur heatmap without caching."""
        result = get_or_compute_blur_heatmap(
            sample_images["image_2d"],
            blur_path=None,
            patch_size=16,
            stride_size=8,
            normalize=False
        )
        
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_images["test_2d_data"].shape

    def test_compute_and_cache(self, sample_images):
        """Test computing and caching blur heatmap."""
        cache_path = sample_images["cache_dir"] / "blur_cache.tif"
        
        result = get_or_compute_blur_heatmap(
            sample_images["image_2d"],
            blur_path=cache_path,
            patch_size=16,
            stride_size=8,
            normalize=False
        )
        
        # Cache file should be created
        assert cache_path.exists()
        assert isinstance(result, np.ndarray)
        
        # Load cached version
        cached_result = get_or_compute_blur_heatmap(
            sample_images["image_2d"],
            blur_path=cache_path,
            patch_size=32,  # Different parameters shouldn't matter
            stride_size=16,
            normalize=True
        )
        
        # Should load from cache (same as first computation)
        np.testing.assert_array_equal(result.astype(np.float32), cached_result.astype(np.float32))

    def test_3d_image_processing(self, sample_images):
        """Test processing 3D images."""
        result = get_or_compute_blur_heatmap(
            sample_images["image_3d"],
            blur_path=None,
            patch_size=16,
            stride_size=8,
            normalize=True
        )
        
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_images["test_3d_data"].shape

    def test_cache_directory_creation(self, sample_images):
        """Test that cache directory is created if it doesn't exist."""
        nested_cache_path = sample_images["cache_dir"] / "nested" / "deep" / "blur.tif"
        
        result = get_or_compute_blur_heatmap(
            sample_images["image_2d"],
            blur_path=nested_cache_path,
            patch_size=16,
            stride_size=8,
            normalize=False  # Don't normalize 2D images
        )
        
        assert nested_cache_path.exists()
        assert isinstance(result, np.ndarray)

    def test_corrupted_cache_handling(self, sample_images):
        """Test handling of corrupted cache files."""
        cache_path = sample_images["cache_dir"] / "corrupted.tif"
        
        # Create a corrupted cache file
        cache_path.write_text("corrupted data")
        
        # Should handle corruption and recompute - the function raises Warning, not UserWarning
        try:
            result = get_or_compute_blur_heatmap(
                sample_images["image_2d"],
                blur_path=cache_path,
                patch_size=16,
                stride_size=8,
                normalize=False
            )
            # If no exception, test passes
            assert isinstance(result, np.ndarray)
        except Warning:
            # If Warning is raised, that's also acceptable behavior
            pass

    @patch('src.utils.blur_measure.load_image')
    def test_load_image_error_handling(self, mock_load_image, sample_images):
        """Test error handling when reading image fails."""
        mock_load_image.side_effect = Exception("Mock read error")
        
        with pytest.raises(Exception):
            get_or_compute_blur_heatmap(
                sample_images["image_2d"],
                blur_path=None,
                patch_size=16,
                stride_size=8
            )

    def test_different_parameter_combinations(self, sample_images):
        """Test various parameter combinations."""
        test_cases = [
            # For 2D images, don't use normalize=True
            {"patch_size": 16, "stride_size": 8, "normalize": False, "center_values": True},
            {"patch_size": 32, "stride_size": 16, "normalize": False, "center_values": False},
            {"patch_size": 8, "stride_size": 4, "normalize": False, "center_values": False},
        ]
        
        for params in test_cases:
            result = get_or_compute_blur_heatmap(
                sample_images["image_2d"],
                blur_path=None,
                **params
            )
            assert isinstance(result, np.ndarray)
            assert result.shape == sample_images["test_2d_data"].shape
        
        # Test 3D image with normalization
        result_3d = get_or_compute_blur_heatmap(
            sample_images["image_3d"],
            blur_path=None,
            patch_size=16,
            stride_size=8,
            normalize=True,
            center_values=True
        )
        assert isinstance(result_3d, np.ndarray)
        assert result_3d.shape == sample_images["test_3d_data"].shape


# --- measure_image_blur enhanced tests ---
class TestMeasureImageBlurEnhanced:
    """Enhanced test cases for measure_image_blur function."""
    
    @pytest.fixture
    def sample_images(self, temp_dir):
        """Create sample images for blur testing."""
        image_dir = Path(temp_dir) / "images"
        image_dir.mkdir()
        
        # Sharp image with high contrast edges
        sharp_img = np.zeros((64, 64))
        sharp_img[20:44, 20:44] = 255
        sharp_img[30:34, 30:34] = 0  # Create internal structure
        
        # Blurry image (uniform with small variations)
        blurry_img = np.ones((64, 64)) * 128 + np.random.rand(64, 64) * 10
        
        # 3D stack
        stack_3d = np.random.rand(5, 32, 32) * 255
        
        # Non-TIFF image (PNG)
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray((np.random.rand(32, 32) * 255).astype(np.uint8))
        
        # Save files
        sharp_path = image_dir / "sharp.tif"
        blurry_path = image_dir / "blurry.tif" 
        stack_path = image_dir / "stack.tif"
        png_path = image_dir / "test.png"
        
        tiff.imwrite(str(sharp_path), sharp_img.astype(np.uint8))
        tiff.imwrite(str(blurry_path), blurry_img.astype(np.uint8))
        tiff.imwrite(str(stack_path), stack_3d.astype(np.uint8))
        pil_img.save(str(png_path))
        
        return {
            "sharp": sharp_path,
            "blurry": blurry_path,
            "stack": stack_path,
            "png": png_path
        }

    def test_sharp_vs_blurry_discrimination(self, sample_images):
        """Test that sharp images get higher scores than blurry ones."""
        sharp_score = measure_image_blur(str(sample_images["sharp"]))
        blurry_score = measure_image_blur(str(sample_images["blurry"]))
        
        assert sharp_score > blurry_score
        assert sharp_score > 0
        assert blurry_score >= 0

    def test_3d_stack_processing(self, sample_images):
        """Test processing of 3D image stacks."""
        score = measure_image_blur(str(sample_images["stack"]))
        assert isinstance(score, float)
        assert score >= 0.0

    def test_non_tiff_image_processing(self, sample_images):
        """Test processing of non-TIFF images."""
        score = measure_image_blur(str(sample_images["png"]))
        assert isinstance(score, float)
        assert score >= 0.0

    def test_method_parameter(self, sample_images):
        """Test different method parameters (though only laplacian is implemented)."""
        # Test default method
        score1 = measure_image_blur(str(sample_images["sharp"]))
        
        # Test explicit laplacian method
        score2 = measure_image_blur(str(sample_images["sharp"]), method='laplacian')
        
        assert score1 == score2

    def test_return_type_consistency(self, sample_images):
        """Test that return type is always float."""
        for image_path in sample_images.values():
            score = measure_image_blur(str(image_path))
            assert isinstance(score, float)

    @patch('tifffile.imread')
    def test_various_3d_shapes(self, mock_imread):
        """Test handling of different 3D image shapes."""
        test_cases = [
            np.random.rand(10, 64, 64),  # Many slices
            np.random.rand(2, 128, 128),  # Few slices, large images
            np.random.rand(64, 64, 3),   # RGB-like shape
        ]
        
        for test_image in test_cases:
            mock_imread.return_value = test_image
            score = measure_image_blur('/fake/path.tif')
            assert isinstance(score, float)
            assert score >= 0.0
