"""
Tests for the postprocessing module.

This module contains tests for all postprocessing functionality including
cell tracking, and blur filtering.
"""

import numpy as np
import pandas as pd
import tifffile
import pytest
from pathlib import Path
import glob
from unittest.mock import patch

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.postprocessing.cell_tracking import CellTracker3D, TrackingConfig
from src.postprocessing.blur_filtering import BlurFilter, FilterConfig
from src.postprocessing.tracking_processor import CellTrackingPipeline, PostprocessingConfig


@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path


@pytest.fixture
def tracker():
    config = TrackingConfig(
        search_range=5.0,
        memory=1,
        min_track_length=2,
        min_area=10,
        max_area=1000
    )
    return CellTracker3D(config)


def create_mock_segmentation_stack(shape=(5, 100, 100)):
    z, h, w = shape
    stack = np.zeros(shape, dtype=np.int32)
    for z_idx in range(z):
        y1 = 20 + z_idx * 2
        x1 = 30 + z_idx * 1
        stack[z_idx, y1:y1+10, x1:x1+10] = 1
        y2 = 50 + z_idx * 3
        x2 = 60
        stack[z_idx, y2:y2+8, x2:x2+8] = 2
        y3 = 70
        x3 = 20
        stack[z_idx, y3:y3+12, x3:x3+12] = 3
    return stack


def test_extract_cell_properties(tracker):
    mask = np.zeros((100, 100), dtype=np.int32)
    mask[20:30, 30:40] = 1
    mask[50:58, 60:68] = 2
    properties = tracker.extract_cell_properties(mask)
    assert len(properties) == 2
    assert set(['x', 'y', 'label', 'area']).issubset(properties.columns)
    assert all(properties['area'] >= tracker.config.min_area)
    assert all(properties['area'] <= tracker.config.max_area)


def test_track_cells_3d(tracker):
    segmentation_stack = create_mock_segmentation_stack()
    tracked_stack = tracker.track_cells(segmentation_stack)
    assert tracked_stack.shape == segmentation_stack.shape
    assert tracker.last_tracking_data is not None
    assert len(tracker.last_tracking_data) > 0
    stats = tracker.get_tracking_summary()
    assert 'n_particles' in stats
    assert 'n_detections' in stats
    assert stats['n_particles'] > 0


def test_empty_segmentation(tracker):
    empty_stack = np.zeros((3, 50, 50), dtype=np.int32)
    tracked_stack = tracker.track_cells(empty_stack)
    assert np.all(tracked_stack == 0)
    stats = tracker.get_tracking_summary()
    assert stats.get('n_particles', 0) == 0


@pytest.fixture
def blur_filter():
    config = FilterConfig(
        patch_size=16,
        stride_size=8,
        blur_threshold=0.5,
        cache_blur_maps=False
    )
    return BlurFilter(config)


def create_mock_blur_heatmap(shape=(100, 100)):
    h, w = shape
    heatmap = np.random.rand(h, w).astype(np.float32)
    heatmap[20:40, 20:40] = 0.8
    heatmap[60:80, 60:80] = 0.2
    return heatmap


def create_mock_segmentation(shape=(100, 100)):
    mask = np.zeros(shape, dtype=np.int32)
    mask[25:35, 25:35] = 1
    mask[65:75, 65:75] = 2
    mask[45:55, 45:55] = 3
    return mask


def test_filter_cells_by_blur(blur_filter):
    segmentation = create_mock_segmentation()
    blur_heatmap = create_mock_blur_heatmap()
    filtered_mask, quality_stats = blur_filter.filter_cells_by_blur(segmentation, blur_heatmap)
    assert filtered_mask.shape == segmentation.shape
    assert isinstance(quality_stats, pd.DataFrame)
    assert 'passes_threshold' in quality_stats.columns
    assert 'blur_intensity' in quality_stats.columns
    n_original = len(np.unique(segmentation)) - 1
    n_filtered = len(np.unique(filtered_mask)) - 1
    assert n_filtered <= n_original


def test_filter_3d_stack(blur_filter):
    segmentation_stack = np.stack([create_mock_segmentation() for _ in range(3)])
    blur_heatmaps = [create_mock_blur_heatmap() for _ in range(3)]
    filtered_stack, quality_stats_list = blur_filter.filter_3d_stack(segmentation_stack, blur_heatmaps)
    assert filtered_stack.shape == segmentation_stack.shape
    assert len(quality_stats_list) == 3
    for stats in quality_stats_list:
        assert isinstance(stats, pd.DataFrame)
        assert 'z' in stats.columns


@pytest.fixture
def pipeline():
    config = PostprocessingConfig(
        enable_blur_filtering=True,
        filter_before_tracking=True,
        save_intermediate_results=False
    )
    return CellTrackingPipeline(config)


def create_test_files(tmp_path):
    segmentation = np.zeros((5, 60, 60), dtype=np.int32)
    for z in range(5):
        y, x = 20 + z * 2, 30 + z * 1
        segmentation[z, y:y+8, x:x+8] = 1
        segmentation[z, 40:48, 40:48] = 2
    image = np.random.randint(0, 255, (5, 60, 60), dtype=np.uint8)
    seg_path = tmp_path / "test_seg_3d.tif"
    img_path = tmp_path / "test_img_BF_3d.tif"
    tifffile.imwrite(str(seg_path), segmentation)
    tifffile.imwrite(str(img_path), image)
    return seg_path, img_path


def test_process_single_file(pipeline, tmp_path):
    seg_path, img_path = create_test_files(tmp_path)
    output_dir = tmp_path / "output"
    result = pipeline.process_single_file(seg_path, img_path, output_dir)
    assert 'input_segmentation' in result
    assert 'final_output' in result
    final_output = Path(result['final_output'])
    assert final_output.exists()


def test_batch_processing(pipeline, tmp_path):
    # Clean up any existing files first
    for f in glob.glob(str(tmp_path / "test_*_3d.tif")):
        os.remove(f)
    test_files = []
    for i in range(3):
        seg, img = create_test_files(tmp_path)
        new_seg = seg.parent / f"test_{i}_masks_3d.tif"
        new_img = img.parent / f"test_{i}_BF_3d.tif"
        seg.rename(new_seg)
        img.rename(new_img)
        test_files.append((new_seg, new_img))
    output_dir = tmp_path / "batch_output"
    results = pipeline.process_batch(tmp_path, tmp_path, output_dir)
    assert len(results) == 3
    successful = [r for r in results if 'error' not in r]
    assert len(successful) == 3


def test_track_3d_centers():
    processor = CellTrackingPipeline(PostprocessingConfig(
        enable_blur_filtering=True,
        filter_before_tracking=True,
        save_intermediate_results=False
    ))
    segmentation = np.zeros((5, 100, 100), dtype=int)
    for z in range(5):
        x1, y1 = 20 + z*3, 20 + z*2
        if x1 < 85 and y1 < 85:
            segmentation[z, y1:y1+10, x1:x1+10] = 1
        x2, y2 = 70 - z*2, 70 - z*3
        if x2 > 15 and y2 > 15:
            segmentation[z, y2:y2+12, x2:x2+12] = 2
    tracked_stack = processor.tracker.track_cells(segmentation)
    assert tracked_stack.shape == segmentation.shape
    assert tracked_stack.dtype == np.int32
    unique_ids = np.unique(tracked_stack)
    assert 0 in unique_ids
    assert len(unique_ids) > 1
