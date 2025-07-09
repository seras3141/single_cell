"""
Unit tests for blur_measure module (utils).
Covers patchwise blur, blur heatmap, image blur, and cache logic.
"""

import numpy as np
import tempfile
import os
from pathlib import Path
import tifffile as tiff
import pytest
from unittest.mock import patch

from src.utils.blur_measure import (
    measure_patchwise_blur,
    measure_blur_heatmap,
    measure_image_blur,
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

@patch('tifffile.imread')
def test_file_input_tiff(mock_imread, blur_heatmap_images):
    mock_imread.return_value = blur_heatmap_images["test_image_2d"]
    result = measure_blur_heatmap('/fake/path/image.tif', normalize=False)
    mock_imread.assert_called_once_with('/fake/path/image.tif')
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

@patch('tifffile.imwrite')
def test_output_file_saving(mock_imwrite, blur_heatmap_images, temp_dir):
    output_path = os.path.join(temp_dir, 'test_output.tif')
    result = measure_blur_heatmap(
        blur_heatmap_images["test_image_2d"],
        output_path=output_path,
        normalize=False
    )
    mock_imwrite.assert_called_once()
    args, kwargs = mock_imwrite.call_args
    assert args[0] == output_path
    assert isinstance(args[1], np.ndarray)

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
