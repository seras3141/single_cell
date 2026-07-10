"""Tests for src/utils/image_utils.py."""

import numpy as np
import pytest
import tifffile
from pathlib import Path
from PIL import Image as PILImage

from src.utils.image_utils import load_image


class TestLoadImage:
    def test_load_2d_tiff(self, tmp_path):
        arr = np.random.randint(0, 255, (32, 64), dtype=np.uint8)
        p = tmp_path / "img.tif"
        tifffile.imwrite(str(p), arr)

        result = load_image(p)
        np.testing.assert_array_equal(result, arr)

    def test_load_3d_tiff(self, tmp_path):
        arr = np.random.randint(0, 255, (4, 32, 32), dtype=np.uint16)
        p = tmp_path / "stack.tif"
        tifffile.imwrite(str(p), arr)

        result = load_image(p)
        assert result.shape == (4, 32, 32)
        assert result.dtype == np.uint16

    def test_load_tiff_string_path(self, tmp_path):
        arr = np.ones((8, 8), dtype=np.uint8)
        p = tmp_path / "img.tif"
        tifffile.imwrite(str(p), arr)

        result = load_image(str(p))
        np.testing.assert_array_equal(result, arr)

    def test_load_tiff_extension_case_insensitive(self, tmp_path):
        arr = np.ones((8, 8), dtype=np.uint8)
        p = tmp_path / "img.TIFF"
        tifffile.imwrite(str(p), arr)

        result = load_image(p)
        np.testing.assert_array_equal(result, arr)

    def test_load_png(self, tmp_path):
        arr = np.random.randint(0, 255, (16, 16), dtype=np.uint8)
        p = tmp_path / "img.png"
        PILImage.fromarray(arr).save(str(p))

        result = load_image(p)
        assert result.shape == (16, 16)
        np.testing.assert_array_equal(result, arr)

    def test_load_returns_ndarray(self, tmp_path):
        arr = np.zeros((4, 4), dtype=np.uint8)
        p = tmp_path / "img.tif"
        tifffile.imwrite(str(p), arr)

        result = load_image(p)
        assert isinstance(result, np.ndarray)

    def test_nonexistent_file_raises(self, tmp_path):
        with pytest.raises(Exception):
            load_image(tmp_path / "missing.tif")
