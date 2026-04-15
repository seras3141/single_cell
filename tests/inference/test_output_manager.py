"""
Unit tests for the OutputManager class and module-level label I/O helpers.
"""

import importlib.util as _ilu
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import tifffile

from src.inference.output_manager import (
    LABEL_FORMATS,
    OutputManager,
    _optimal_label_dtype,
    load_labels,
    save_labels,
)

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
def temp_output_dir():
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def masks_2d():
    rng = np.random.default_rng(0)
    return rng.integers(0, 10, (32, 32), dtype=np.uint16)


@pytest.fixture
def masks_3d():
    rng = np.random.default_rng(0)
    return rng.integers(0, 50, (4, 32, 32), dtype=np.uint16)


# ─── save_labels / load_labels ────────────────────────────────────────────────

class TestSaveLoadLabels:
    """Round-trip tests for the module-level save_labels / load_labels helpers."""

    @pytest.mark.parametrize("fmt,ext", FORMAT_CASES)
    def test_creates_output(self, tmp_path, masks_3d, fmt, ext):
        out = tmp_path / f"labels{ext}"
        save_labels(masks_3d, out)
        assert out.exists()

    @pytest.mark.parametrize("fmt,ext", FORMAT_CASES)
    def test_roundtrip_3d(self, tmp_path, masks_3d, fmt, ext):
        out = tmp_path / f"labels{ext}"
        save_labels(masks_3d, out)
        recovered = load_labels(out)
        np.testing.assert_array_equal(masks_3d, recovered)

    @pytest.mark.parametrize("fmt,ext", FORMAT_CASES)
    def test_roundtrip_2d(self, tmp_path, masks_2d, fmt, ext):
        out = tmp_path / f"labels{ext}"
        save_labels(masks_2d, out)
        recovered = load_labels(out)
        np.testing.assert_array_equal(masks_2d, recovered)

    @pytest.mark.parametrize("fmt,ext", FORMAT_CASES)
    def test_dtype_normalisation(self, tmp_path, fmt, ext):
        """save_labels always persists data as an unsigned integer dtype."""
        masks = np.array([[[0, 1, 2], [3, 4, 5]]], dtype=np.int32)
        out = tmp_path / f"labels{ext}"
        save_labels(masks, out)
        recovered = load_labels(out)
        assert recovered.dtype in (np.dtype("uint8"), np.dtype("uint16"), np.dtype("uint32"))
        np.testing.assert_array_equal(masks, recovered)

    @pytest.mark.parametrize("fmt,ext", FORMAT_CASES)
    def test_large_instance_ids_use_uint32(self, tmp_path, fmt, ext):
        masks = np.array([[[0, 70000]]], dtype=np.uint32)
        out = tmp_path / f"labels{ext}"
        save_labels(masks, out)
        recovered = load_labels(out)
        assert recovered.dtype == np.dtype("uint32")
        np.testing.assert_array_equal(masks, recovered)

    def test_unsupported_extension_save(self, tmp_path, masks_3d):
        with pytest.raises(ValueError, match="Unsupported"):
            save_labels(masks_3d, tmp_path / "labels.npy")

    def test_unsupported_extension_load(self, tmp_path):
        p = tmp_path / "labels.npy"
        p.write_bytes(b"dummy")
        with pytest.raises(ValueError, match="Unsupported"):
            load_labels(p)

    def test_zarr_chunk_layout(self, tmp_path, masks_3d):
        """Zarr arrays should be chunked one slice at a time along the z-axis."""
        zarr = pytest.importorskip("zarr")
        out = tmp_path / "labels.zarr"
        save_labels(masks_3d, out)
        zarr_major = int(zarr.__version__.split(".")[0])
        z = zarr.open_array(str(out), mode="r") if zarr_major >= 3 else zarr.open(str(out), mode="r")
        chunks = z.chunks
        assert chunks[0] == 1                          # one z-slice per chunk
        assert chunks[1] == masks_3d.shape[1]
        assert chunks[2] == masks_3d.shape[2]

    def test_hdf5_dataset_key(self, tmp_path, masks_3d):
        """HDF5 files must store data under the 'labels' key."""
        h5py = pytest.importorskip("h5py")
        out = tmp_path / "labels.h5"
        save_labels(masks_3d, out)
        with h5py.File(out, "r") as f:
            assert "labels" in f
            assert f["labels"].shape == masks_3d.shape


# ─── _optimal_label_dtype ─────────────────────────────────────────────────────

class TestOptimalLabelDtype:
    """Unit tests for the dtype-selection helper."""

    def test_uint8_for_values_up_to_255(self):
        arr = np.array([0, 128, 255], dtype=np.int32)
        assert _optimal_label_dtype(arr) == np.uint8

    def test_uint16_for_values_256_to_65535(self):
        arr = np.array([0, 256, 65535], dtype=np.int32)
        assert _optimal_label_dtype(arr) == np.uint16

    def test_uint32_for_values_above_65535(self):
        arr = np.array([0, 65536], dtype=np.int64)
        assert _optimal_label_dtype(arr) == np.uint32

    def test_boundary_255_is_uint8(self):
        arr = np.array([255], dtype=np.uint16)
        assert _optimal_label_dtype(arr) == np.uint8

    def test_boundary_256_is_uint16(self):
        arr = np.array([256], dtype=np.uint32)
        assert _optimal_label_dtype(arr) == np.uint16

    def test_boundary_65535_is_uint16(self):
        arr = np.array([65535], dtype=np.uint32)
        assert _optimal_label_dtype(arr) == np.uint16

    def test_boundary_65536_is_uint32(self):
        arr = np.array([65536], dtype=np.uint32)
        assert _optimal_label_dtype(arr) == np.uint32

    def test_all_zeros_is_uint8(self):
        arr = np.zeros((4, 4), dtype=np.uint16)
        assert _optimal_label_dtype(arr) == np.uint8


# ─── OutputManager ────────────────────────────────────────────────────────────

class TestOutputManagerInit:
    def test_initialization(self, temp_output_dir):
        manager = OutputManager(
            base_output_dir=temp_output_dir,
            model_name="test_model",
            dataset_name="test_dataset",
        )
        assert manager.output_dir.name == "test_dataset"
        assert "test_model" in str(manager.output_dir)
        assert manager.output_dir.exists()
        assert manager.masks_dir.exists()
        assert manager.metadata_dir.exists()
        assert manager.overlays_dir.exists()

    def test_directory_creation(self, temp_output_dir):
        manager = OutputManager(
            base_output_dir=temp_output_dir,
            model_name="cyto3",
            dataset_name="test",
        )
        expected_base = temp_output_dir / "cyto3" / "test"
        assert manager.output_dir == expected_base
        assert (expected_base / "masks").exists()
        assert (expected_base / "metadata").exists()
        assert (expected_base / "overlays").exists()

    def test_invalid_label_format(self, temp_output_dir):
        with pytest.raises(ValueError, match="label_format"):
            OutputManager(temp_output_dir, label_format="png")

    @pytest.mark.parametrize("fmt,ext", FORMAT_CASES)
    def test_label_format_stored(self, temp_output_dir, fmt, ext):
        manager = OutputManager(temp_output_dir, label_format=fmt)
        assert manager.label_format == fmt
        assert manager._label_ext == ext


class TestSavePrediction:
    @pytest.mark.parametrize("fmt,ext", FORMAT_CASES)
    def test_label_formats(self, temp_output_dir, masks_2d, fmt, ext):
        manager = OutputManager(
            base_output_dir=temp_output_dir,
            model_name="model",
            dataset_name=f"ds_{fmt}",
            label_format=fmt,
        )
        metadata = {"num_cells": 10, "parameters": {}}
        result = manager.save_prediction(
            masks=masks_2d,
            metadata=metadata,
            input_path=Path("input.tif"),
            save_overlay=False,
        )
        assert result["masks"].suffix == ext
        assert result["masks"].exists()
        recovered = load_labels(result["masks"])
        np.testing.assert_array_equal(masks_2d, recovered)

    def test_default_format_is_zarr(self, temp_output_dir, masks_2d):
        """Default label_format must be zarr."""
        manager = OutputManager(
            base_output_dir=temp_output_dir,
            model_name="test_model",
            dataset_name="test",
        )
        result = manager.save_prediction(
            masks=masks_2d,
            metadata={"num_cells": 5},
            input_path=Path("test_image.tif"),
            save_overlay=False,
        )
        assert result["masks"].suffix == ".zarr"
        assert result["masks"].exists()
        assert result["metadata"].suffix == ".json"

    def test_roundtrip_default(self, temp_output_dir):
        manager = OutputManager(
            base_output_dir=temp_output_dir,
            model_name="test_model",
            dataset_name="test",
        )
        test_masks = np.random.randint(0, 100, (128, 128), dtype=np.uint16)
        result = manager.save_prediction(
            masks=test_masks,
            metadata={"num_cells": 50, "parameters": {"flow_threshold": 0.4}},
            input_path=Path("test_image.tif"),
            save_overlay=False,
        )
        assert result["masks"].exists()
        assert result["metadata"].exists()
        saved_masks = OutputManager.load_masks(result["masks"])
        np.testing.assert_array_equal(saved_masks, test_masks)

    def test_metadata_keys(self, temp_output_dir, masks_2d):
        manager = OutputManager(
            base_output_dir=temp_output_dir,
            model_name="test_model",
            dataset_name="test",
        )
        result = manager.save_prediction(
            masks=masks_2d,
            metadata={"num_cells": 50},
            input_path=Path("test_image.tif"),
            save_overlay=False,
        )
        with open(result["metadata"]) as f:
            saved = json.load(f)
        assert saved["num_cells"] == 50
        assert "saved_at" in saved
        assert "input_file" in saved

    def test_save_prediction_with_overlay(self, temp_output_dir):
        manager = OutputManager(
            base_output_dir=temp_output_dir,
            model_name="test_model",
            dataset_name="test",
        )
        test_image = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        test_masks = np.random.randint(0, 20, (64, 64), dtype=np.uint16)
        result = manager.save_prediction(
            masks=test_masks,
            metadata={"num_cells": 15},
            input_path=Path("test_image.tif"),
            original_image=test_image,
            save_overlay=True,
        )
        assert result["masks"].exists()
        assert result["metadata"].exists()


class TestSaveZStackPrediction:
    @pytest.mark.parametrize("fmt,ext", FORMAT_CASES)
    def test_label_formats(self, temp_output_dir, masks_3d, fmt, ext):
        manager = OutputManager(
            base_output_dir=temp_output_dir,
            model_name="model",
            dataset_name=f"ds_{fmt}",
            label_format=fmt,
        )
        result = manager.save_z_stack_prediction(
            masks_stack=masks_3d,
            metadata={"num_cells": 5},
            input_path=Path("stack.tif"),
        )
        assert result["stack"]["masks"].suffix == ext
        assert result["stack"]["masks"].exists()
        recovered = load_labels(result["stack"]["masks"])
        np.testing.assert_array_equal(masks_3d, recovered)

        assert len(result["slices"]) == masks_3d.shape[0]
        for s in result["slices"]:
            assert s["masks"].suffix == ext
            assert s["masks"].exists()

    def test_default_format_zarr(self, temp_output_dir, masks_3d):
        manager = OutputManager(
            base_output_dir=temp_output_dir,
            model_name="test_model",
            dataset_name="test",
        )
        result = manager.save_z_stack_prediction(
            masks_stack=masks_3d,
            metadata={"stack_shape": masks_3d.shape, "num_cells": 10},
            input_path=Path("test_stack.tif"),
        )
        assert "stack" in result
        assert "slices" in result
        assert result["stack"]["masks"].exists()
        assert result["stack"]["metadata"].exists()
        assert len(result["slices"]) == masks_3d.shape[0]
        for i, slice_info in enumerate(result["slices"]):
            assert slice_info["masks"].exists()
            assert slice_info["z_index"] == i
            assert f"_z{i:03d}_" in str(slice_info["masks"])


# ─── Other OutputManager behaviour ───────────────────────────────────────────

def test_json_serialization_with_numpy_arrays(temp_output_dir):
    manager = OutputManager(
        base_output_dir=temp_output_dir,
        model_name="test_model",
        dataset_name="test",
    )
    test_metadata = {
        "small_array": np.array([1, 2, 3]),
        "large_array": np.random.random((100, 100)),
        "numpy_int": np.int64(42),
        "numpy_float": np.float32(3.14),
        "nested_dict": {
            "inner_array": np.array([4, 5, 6]),
            "inner_list": [np.int32(7), np.float64(8.0)],
        },
        "normal_data": "test_string",
    }
    test_masks = np.zeros((32, 32), dtype=np.uint16)
    result = manager.save_prediction(
        masks=test_masks,
        metadata=test_metadata,
        input_path=Path("test_serialization.tif"),
        save_overlay=False,
    )
    with open(result["metadata"]) as f:
        saved_metadata = json.load(f)
    assert saved_metadata["small_array"] == [1, 2, 3]
    assert isinstance(saved_metadata["large_array"], str)
    assert saved_metadata["numpy_int"] == 42
    assert saved_metadata["numpy_float"] == pytest.approx(3.14, rel=1e-3)
    assert saved_metadata["normal_data"] == "test_string"


def test_finalize_run(temp_output_dir):
    manager = OutputManager(
        base_output_dir=temp_output_dir,
        model_name="test_model",
        dataset_name="test",
    )
    manager.run_metadata["processed_files"] = [
        {"file": "test1.tif", "num_cells": 25},
        {"file": "test2.tif", "num_cells": 30},
        {"file": "test3.tif", "num_cells": 15},
    ]
    summary_path = manager.finalize_run()
    assert summary_path.exists()
    assert summary_path.name == "run_summary.json"
    with open(summary_path) as f:
        summary = json.load(f)
    assert summary["summary"]["total_files_processed"] == 3
    assert summary["summary"]["total_cells_detected"] == 70
    assert "completed_at" in summary["summary"]


def test_get_output_filename(temp_output_dir):
    manager = OutputManager(
        base_output_dir=temp_output_dir,
        model_name="test_model",
        dataset_name="test",
    )
    test_cases = [
        ("image_BF.tif", "image"),
        ("sample_001.tiff", "sample_001"),
        ("data/test/file.png", "file"),
        ("complex_name_with_underscores.tif", "complex_name_with_underscores"),
        ("test_Cells.tif", "test"),
        ("sample_DAPI.tif", "sample"),
    ]
    for input_name, expected_base in test_cases:
        result = manager._get_output_filename(Path(input_name))
        assert result == expected_base, f"For {input_name!r}: expected {expected_base!r}, got {result!r}"


def test_error_handling_file_save_failure(temp_output_dir):
    manager = OutputManager(
        base_output_dir=temp_output_dir,
        model_name="test_model",
        dataset_name="test",
    )
    test_masks = np.random.randint(0, 50, (64, 64), dtype=np.uint16)
    invalid_metadata = {"bad_data": object()}
    with pytest.raises(Exception):
        manager.save_prediction(
            masks=test_masks,
            metadata=invalid_metadata,
            input_path=Path("test.tif"),
            save_overlay=False,
        )


def test_preserve_run_metadata(temp_output_dir):
    manager = OutputManager(
        base_output_dir=temp_output_dir,
        model_name="test_model",
        dataset_name="test",
    )
    assert "created_at" in manager.run_metadata
    assert "model_name" in manager.run_metadata
    assert "dataset_name" in manager.run_metadata
    assert manager.run_metadata["processed_files"] == []

    test_masks = np.zeros((32, 32), dtype=np.uint16)
    manager.save_prediction(
        masks=test_masks,
        metadata={"num_cells": 10},
        input_path=Path("test.tif"),
        save_overlay=False,
    )
    assert len(manager.run_metadata["processed_files"]) == 1
    assert manager.run_metadata["processed_files"][0]["num_cells"] == 10
