"""
Tests for the 3D cell tracking pipeline (unified postprocessing).
"""

import importlib.util as _ilu
import json
import tempfile
import shutil
from pathlib import Path

import numpy as np
import pytest
import tifffile

from src.postprocessing.tracking_processor import (
    CellTrackingPipeline,
    TrackingOutputManager,
)
from src.postprocessing import TrackingConfig, FilterConfig
from src.utils.config_schemas import PostprocessingConfig
from src.inference.output_manager import LABEL_FORMATS, load_labels

# ─── Skip markers for optional deps ──────────────────────────────────────────

_ZARR_AVAILABLE = _ilu.find_spec("zarr") is not None
_H5PY_AVAILABLE = _ilu.find_spec("h5py") is not None

_skip_no_zarr = pytest.mark.skipif(not _ZARR_AVAILABLE, reason="zarr not installed")
_skip_no_h5py = pytest.mark.skipif(not _H5PY_AVAILABLE, reason="h5py not installed")

FORMAT_CASES = [
    pytest.param("tif", ".tif", id="tif"),
    pytest.param("zarr", ".zarr", id="zarr", marks=_skip_no_zarr),
    pytest.param("hdf5", ".h5", id="hdf5", marks=_skip_no_h5py),
]

# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_segmentation_stack():
    """3D segmentation stack with two labelled regions per slice."""
    stack = np.zeros((3, 50, 50), dtype=np.uint16)
    stack[0, 10:20, 10:20] = 1
    stack[0, 30:40, 30:40] = 2
    stack[1, 12:22, 12:22] = 1
    stack[1, 32:42, 32:42] = 2
    stack[2, 14:24, 14:24] = 1
    stack[2, 34:44, 34:44] = 2
    return stack


@pytest.fixture
def mock_image_stack():
    return np.random.randint(0, 255, (3, 50, 50), dtype=np.uint8)


@pytest.fixture
def pipeline():
    config = PostprocessingConfig(
        enable_blur_filtering=True,
        filter_before_tracking=True,
        save_intermediate_results=False,
    )
    return CellTrackingPipeline(config)


# ─── TestPipelineConfig ───────────────────────────────────────────────────────

class TestPipelineConfig:
    """Tests for the PostprocessingConfig class."""

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
            tracking=tracking_config,
            filtering=filter_config,
        )

        assert config.enable_blur_filtering is False
        assert config.filter_before_tracking is False
        assert config.save_intermediate_results is True
        assert config.tracking.search_range == 10.0
        assert config.filtering.patch_size == 64


# ─── TestTrackingOutputManager ────────────────────────────────────────────────

class TestTrackingOutputManager:
    """Tests for TrackingOutputManager — save methods and label format support."""

    def test_default_label_format_is_zarr(self, tmp_path):
        om = TrackingOutputManager(tmp_path)
        assert om.label_format == "zarr"
        assert om._label_ext == ".zarr"

    def test_invalid_label_format(self, tmp_path):
        with pytest.raises(ValueError, match="label_format"):
            TrackingOutputManager(tmp_path, label_format="png")

    def test_subdirectory_creation(self, tmp_path):
        om = TrackingOutputManager(tmp_path)
        for key in ("blur_filtered", "tracked", "tracked_blur_filtered", "final", "final_2d"):
            assert om.subdirs[key].exists()

    @pytest.mark.parametrize("fmt,ext", FORMAT_CASES)
    def test_save_final_format(self, tmp_path, mock_segmentation_stack, fmt, ext):
        om = TrackingOutputManager(tmp_path, label_format=fmt)
        out = om.save_final(mock_segmentation_stack, "sample")
        out_path = Path(out)
        assert out_path.suffix == ext
        assert out_path.exists()
        recovered = load_labels(out_path)
        np.testing.assert_array_equal(mock_segmentation_stack, recovered)

    @pytest.mark.parametrize("fmt,ext", FORMAT_CASES)
    def test_save_blur_filtered_format(self, tmp_path, mock_segmentation_stack, fmt, ext):
        om = TrackingOutputManager(tmp_path, label_format=fmt)
        out = om.save_blur_filtered(mock_segmentation_stack, "sample")
        out_path = Path(out)
        assert out_path.suffix == ext
        assert out_path.exists()

    @pytest.mark.parametrize("fmt,ext", FORMAT_CASES)
    def test_save_tracked_format(self, tmp_path, mock_segmentation_stack, fmt, ext):
        om = TrackingOutputManager(tmp_path, label_format=fmt)
        out = om.save_tracked(mock_segmentation_stack, "sample")
        out_path = Path(out)
        assert out_path.suffix == ext
        assert out_path.exists()

    @pytest.mark.parametrize("fmt,ext", FORMAT_CASES)
    def test_save_tracked_blur_filtered_format(self, tmp_path, mock_segmentation_stack, fmt, ext):
        om = TrackingOutputManager(tmp_path, label_format=fmt)
        out = om.save_tracked_blur_filtered(mock_segmentation_stack, "sample")
        out_path = Path(out)
        assert out_path.suffix == ext
        assert out_path.exists()

    @pytest.mark.parametrize("fmt,ext", FORMAT_CASES)
    def test_roundtrip_data_integrity(self, tmp_path, mock_segmentation_stack, fmt, ext):
        """Data written and read back must be identical for all formats."""
        om = TrackingOutputManager(tmp_path, label_format=fmt)
        out = om.save_final(mock_segmentation_stack, "roundtrip")
        recovered = load_labels(Path(out))
        np.testing.assert_array_equal(mock_segmentation_stack, recovered)

    def test_batch_summary_saved(self, tmp_path):
        om = TrackingOutputManager(tmp_path)
        results = [{"file": "a.tif", "status": "ok"}, {"file": "b.tif", "status": "ok"}]
        summary_path = om.save_batch_summary(results)
        assert Path(summary_path).exists()
        with open(summary_path) as f:
            loaded = json.load(f)
        assert len(loaded) == 2


# ─── TestCellTrackingPipeline ────────────────────────────────────────────────

class TestCellTrackingPipeline:
    """Integration tests for the CellTrackingPipeline."""

    def test_process_single_file(self, pipeline, tmp_path, mock_segmentation_stack, mock_image_stack):
        """Pipeline completes and the final output file exists."""
        seg_path = tmp_path / "test_3d.tif"
        img_path = tmp_path / "test_BF_3d.tif"
        tifffile.imwrite(str(seg_path), mock_segmentation_stack)
        tifffile.imwrite(str(img_path), mock_image_stack)
        output_dir = tmp_path / "output"

        result = pipeline.process_single_file(seg_path, img_path, output_dir)

        assert "final_output" in result
        assert Path(result["final_output"]).exists()

    def test_process_single_file_default_format_is_zarr(
        self, pipeline, tmp_path, mock_segmentation_stack, mock_image_stack
    ):
        """Default TrackingOutputManager writes zarr output."""
        seg_path = tmp_path / "test_3d.tif"
        img_path = tmp_path / "test_BF_3d.tif"
        tifffile.imwrite(str(seg_path), mock_segmentation_stack)
        tifffile.imwrite(str(img_path), mock_image_stack)

        result = pipeline.process_single_file(seg_path, img_path, tmp_path / "out")

        assert Path(result["final_output"]).suffix == ".zarr"

    def test_process_batch(self, pipeline, tmp_path, mock_segmentation_stack, mock_image_stack):
        """Batch processing returns one result per input file."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        for i in range(2):
            tifffile.imwrite(str(input_dir / f"sample_{i}_masks_3d.tif"), mock_segmentation_stack)
            tifffile.imwrite(str(input_dir / f"sample_{i}_BF_3d.tif"), mock_image_stack)
        output_dir = tmp_path / "output"

        results = pipeline.process_batch(input_dir, input_dir, output_dir)

        assert len(results) == 2
        for r in results:
            assert "final_output" in r
            assert Path(r["final_output"]).exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
