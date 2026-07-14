"""
Tests for blur_filtering.py and its interaction with the blur_measure pipeline.

BlurFileHandler.rename_image requires filenames that match the plate naming
convention: p<plate>_<row><col>_t<time>[_z<z>]_<type>.tif
(e.g. 'p1_A01_t1_BF_3d.tif').  Tests that exercise disk caching must
use such a filename.
"""

import numpy as np
import pandas as pd
import pytest
import tifffile
from pathlib import Path

from src.postprocessing.blur_filtering import BlurFilter, blur_intensity_metric
from src.utils.config_schemas import FilterConfig


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_3d_tif(path: Path, shape=(3, 20, 20), dtype=np.uint8) -> np.ndarray:
    """Write a random 3D TIFF and return the array."""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, shape, dtype=dtype)
    tifffile.imwrite(str(path), img, photometric="minisblack")
    return img


# ─── blur_intensity_metric ────────────────────────────────────────────────────

def test_blur_intensity_metric_basic():
    regionmask = np.zeros((10, 10), dtype=bool)
    regionmask[2:5, 2:5] = True
    intensity_image = np.ones((10, 10)) * 0.5
    intensity_image[2:5, 2:5] = 0.8
    val = blur_intensity_metric(regionmask, intensity_image)
    assert np.isclose(val, 0.8)


def test_blur_intensity_metric_empty_region():
    """An all-False mask returns NaN."""
    regionmask = np.zeros((10, 10), dtype=bool)
    intensity_image = np.ones((10, 10)) * 0.5
    assert np.isnan(blur_intensity_metric(regionmask, intensity_image))


# ─── filter_cells_by_blur ─────────────────────────────────────────────────────

def test_filter_cells_by_blur():
    mask = np.zeros((20, 20), dtype=int)
    mask[2:8, 2:8] = 1
    mask[10:18, 10:18] = 2
    blur = np.ones((20, 20)) * 0.3
    blur[2:8, 2:8] = 0.7  # region 1 blurry (high value), region 2 sharp (low value)

    config = FilterConfig(blur_threshold=0.5, invert_threshold=False)
    blur_filter = BlurFilter(config)
    filtered, stats = blur_filter.filter_cells_by_blur(mask, blur)

    # invert_threshold=False: keep cells where blur_value < threshold
    assert 1 not in filtered          # region 1 (0.7 > 0.5) removed
    assert 2 in filtered              # region 2 (0.3 < 0.5) kept
    assert np.max(filtered) == 2
    assert isinstance(stats, pd.DataFrame)
    assert {"label", "area", "blur_intensity", "passes_threshold"}.issubset(stats.columns)


def test_filter_cells_by_blur_invert_threshold():
    """invert_threshold=True keeps cells where blur > threshold."""
    mask = np.zeros((20, 20), dtype=int)
    mask[2:8, 2:8] = 1
    mask[10:18, 10:18] = 2
    blur = np.ones((20, 20)) * 0.3
    blur[2:8, 2:8] = 0.7

    config = FilterConfig(blur_threshold=0.5, invert_threshold=True)
    filtered, _ = BlurFilter(config).filter_cells_by_blur(mask, blur)

    assert 1 in filtered              # region 1 (0.7 > 0.5) kept
    assert 2 not in filtered          # region 2 (0.3 < 0.5) removed


def test_filter_cells_by_blur_no_regions():
    """Empty segmentation mask returns zero mask and empty DataFrame."""
    mask = np.zeros((20, 20), dtype=int)
    blur = np.ones((20, 20)) * 0.3
    filtered, stats = BlurFilter().filter_cells_by_blur(mask, blur)
    assert np.all(filtered == 0)
    assert isinstance(stats, pd.DataFrame)
    assert len(stats) == 0


# ─── filter_cells_by_blur_fast ────────────────────────────────────────────────

def test_filter_cells_by_blur_fast_matches_slow():
    """Fast and slow methods agree on which labels survive."""
    mask = np.zeros((20, 20), dtype=int)
    mask[2:8, 2:8] = 1
    mask[10:18, 10:18] = 2
    blur = np.ones((20, 20)) * 0.3
    blur[2:8, 2:8] = 0.7

    bf = BlurFilter(FilterConfig(blur_threshold=0.5))
    filtered_slow, _ = bf.filter_cells_by_blur(mask, blur)
    filtered_fast, stats_fast = bf.filter_cells_by_blur_fast(mask, blur)

    assert 1 not in filtered_fast
    assert 2 in filtered_fast
    assert isinstance(stats_fast, pd.DataFrame)
    assert set(np.unique(filtered_slow)) == set(np.unique(filtered_fast))


def test_filter_cells_by_blur_fast_no_regions():
    mask = np.zeros((20, 20), dtype=int)
    blur = np.ones((20, 20)) * 0.3
    filtered, stats = BlurFilter().filter_cells_by_blur_fast(mask, blur)
    assert np.all(filtered == 0)
    assert len(stats) == 0


# ─── filter_3d_stack ─────────────────────────────────────────────────────────

def test_filter_3d_stack():
    """Per-slice filtering is applied and z column is added to each DataFrame."""
    stack = np.zeros((3, 20, 20), dtype=int)
    stack[0, 2:8, 2:8] = 1       # slice 0: blurry cell
    stack[1, 2:8, 2:8] = 1       # slice 1: blurry cell
    stack[2, 10:18, 10:18] = 2   # slice 2: sharp cell

    blur_maps = np.ones((3, 20, 20)) * 0.3
    blur_maps[0, 2:8, 2:8] = 0.7
    blur_maps[1, 2:8, 2:8] = 0.7

    bf = BlurFilter(FilterConfig(blur_threshold=0.5))
    filtered_stack, all_stats = bf.filter_3d_stack(stack, blur_maps)

    assert filtered_stack.shape == stack.shape
    assert 1 not in filtered_stack[0]
    assert 1 not in filtered_stack[1]
    assert 2 in filtered_stack[2]
    assert len(all_stats) == 3
    for z_idx, df in enumerate(all_stats):
        assert isinstance(df, pd.DataFrame)
        assert df["z"].iloc[0] == z_idx


def test_filter_3d_stack_mismatched_blur_raises():
    stack = np.zeros((3, 10, 10), dtype=int)
    blur_maps = np.ones((2, 10, 10)) * 0.3  # wrong z count
    with pytest.raises(ValueError, match="must match"):
        BlurFilter().filter_3d_stack(stack, blur_maps)


# ─── generate_blur_heatmap ────────────────────────────────────────────────────

def test_generate_blur_heatmap_shape(tmp_path):
    """generate_blur_heatmap returns an array with the same spatial shape as the input."""
    from src.utils.blur_measure import generate_blur_heatmap

    img_path = tmp_path / "p1_A01_t1_BF_3d.tif"
    img = _make_3d_tif(img_path)

    blur_map = generate_blur_heatmap(img_path)
    assert blur_map.shape == img.shape


def test_generate_blur_heatmap_saves_to_disk(tmp_path):
    """When blur_path is provided the heatmap file is created on disk."""
    from src.utils.blur_measure import generate_blur_heatmap

    img_path = tmp_path / "p1_A01_t1_BF_3d.tif"
    _make_3d_tif(img_path)
    blur_path = tmp_path / "p1_A01_t1_BF_3d_blur_heatmap.tif"

    generate_blur_heatmap(img_path, blur_path=blur_path)
    assert blur_path.exists()


def test_get_or_compute_blur_heatmap_loads_from_disk(tmp_path):
    """get_or_compute_blur_heatmap reads a pre-existing cached file."""
    from src.utils.blur_measure import generate_blur_heatmap, get_or_compute_blur_heatmap

    img_path = tmp_path / "p1_A01_t1_BF_3d.tif"
    img = _make_3d_tif(img_path)
    blur_path = tmp_path / "p1_A01_t1_BF_3d_blur_heatmap.tif"

    generate_blur_heatmap(img_path, blur_path=blur_path)
    loaded = get_or_compute_blur_heatmap(img_path, blur_path=blur_path)
    assert loaded.shape == img.shape


# ─── BlurFilter.get_or_compute_blur_heatmap ──────────────────────────────────

def test_get_or_compute_blur_heatmap_no_cache_dir(tmp_path):
    """Without a cache dir the heatmap is computed and matches input shape."""
    img_path = tmp_path / "p1_A01_t1_BF_3d.tif"
    img = _make_3d_tif(img_path)

    blur_map = BlurFilter().get_blur_heatmap(img_path)
    assert blur_map.shape == img.shape


def test_get_or_compute_blur_heatmap_memory_cache(tmp_path):
    """Second call returns the cached in-memory result."""
    img_path = tmp_path / "p1_A01_t1_BF_3d.tif"
    _make_3d_tif(img_path)

    bf = BlurFilter()
    blur_map1 = bf.get_blur_heatmap(img_path)
    blur_map2 = bf.get_blur_heatmap(img_path)

    assert np.allclose(blur_map1, blur_map2)


def test_get_or_compute_blur_heatmap_disk_cache(tmp_path):
    """With a valid plate-convention filename and cache dir, heatmap is persisted to disk."""
    # BlurFileHandler.rename_image requires: p<plate>_<row><col>_t<time>_<type>_3d.tif
    img_path = tmp_path / "p1_A01_t1_BF_3d.tif"
    img = _make_3d_tif(img_path)

    blur_map = BlurFilter().get_blur_heatmap(img_path, tmp_path)
    assert blur_map.shape == img.shape

    blur_files = [f for f in tmp_path.glob("*.tif") if "blur_heatmap" in f.name]
    assert len(blur_files) == 1
