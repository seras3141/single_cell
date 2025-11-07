import numpy as np
import pandas as pd
import tempfile
import tifffile
import pytest
from pathlib import Path
from src.postprocessing.blur_filtering import BlurFilter, FilterConfig, blur_intensity_metric

def test_blur_intensity_metric():
    regionmask = np.zeros((10, 10), dtype=bool)
    regionmask[2:5, 2:5] = True
    intensity_image = np.ones((10, 10)) * 0.5
    intensity_image[2:5, 2:5] = 0.8
    val = blur_intensity_metric(regionmask, intensity_image)
    assert np.isclose(val, 0.8)

def test_filter_cells_by_blur():
    mask = np.zeros((20, 20), dtype=int)
    mask[2:8, 2:8] = 1
    mask[10:18, 10:18] = 2
    blur = np.ones((20, 20)) * 0.3
    blur[2:8, 2:8] = 0.7  # region 1 is blurry, region 2 is sharp
    config = FilterConfig(blur_threshold=0.5)
    blur_filter = BlurFilter(config)
    filtered, stats = blur_filter.filter_cells_by_blur(mask, blur)
    # Only region 2 should remain
    assert np.max(filtered) == 2
    assert 1 not in filtered
    assert 2 in filtered
    assert isinstance(stats, pd.DataFrame)

def test_get_or_compute_blur_heatmap(tmp_path):
    # Create a fake 3D image file
    img = np.random.randint(0, 255, (3, 10, 10), dtype=np.uint8)
    img_path = tmp_path / "img_BF_3d.tif"
    tifffile.imwrite(str(img_path), img)
    blur_filter = BlurFilter()
    # Should compute and cache
    blur_map = blur_filter.get_or_compute_blur_heatmap(img_path, tmp_path)
    assert blur_map.shape == img.shape
    # Should load from cache
    blur_map2 = blur_filter.get_or_compute_blur_heatmap(img_path, tmp_path)
    assert np.allclose(blur_map, blur_map2)
