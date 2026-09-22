"""Tests for per-(well, timepoint) metrics measured on the z0 projection masks."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest
import tifffile

from src.dataset_analysis.z0_population import (
    compute_z0_population,
    drop_unmeasurable,
    measure_z0_mask,
    saturation_report,
)
from src.utils.image_utils import save_labels

# The real plate layout, as tests/dataset_analysis/test_collapse_summary.py does: a
# hand-rolled stand-in silently diverges from the schema get_well_annotation expects.
LAYOUT_PATH = Path(__file__).resolve().parents[2] / "config" / "MF5v1_plate_layout.json"
LAYOUT = json.loads(LAYOUT_PATH.read_text())


def _write_mask(path, array, dtype=None):
    """Write a label array as a .tif mask (load_labels reads tif and zarr alike)."""
    arr = np.asarray(array)
    tifffile.imwrite(str(path), arr if dtype is None else arr.astype(dtype))
    return path


def _mask_name(well="C09", timepoint=1, z=0):
    return f"pMF5V1_{well}_t{timepoint}_z{z}_pred_mask.tif"


class TestMeasureZ0Mask:
    def test_counts_objects_and_coverage(self, tmp_path):
        arr = np.zeros((10, 10), dtype=np.uint16)
        arr[0, 0] = 1
        arr[0, 1] = 1
        arr[5, 5] = 2
        path = _write_mask(tmp_path / _mask_name(), arr)

        result = measure_z0_mask(path)

        assert result["n_objects"] == 2
        assert result["coverage_fraction"] == pytest.approx(3 / 100)
        assert result["mean_object_area_fraction"] == pytest.approx(3 / 100 / 2)
        assert result["fov_pixels"] == 100

    def test_excludes_background_label_zero(self, tmp_path):
        arr = np.array([[0, 1], [2, 3]], dtype=np.uint16)
        result = measure_z0_mask(_write_mask(tmp_path / _mask_name(), arr))
        assert result["n_objects"] == 3

    def test_handles_non_contiguous_labels(self, tmp_path):
        """Relabeling leaves gaps, so max() would badly overcount."""
        arr = np.array([[0, 1], [5, 99]], dtype=np.uint16)
        result = measure_z0_mask(_write_mask(tmp_path / _mask_name(), arr))
        assert result["n_objects"] == 3
        assert result["max_label"] == 99

    def test_frame_with_no_background_is_not_off_by_one(self, tmp_path):
        arr = np.array([[1, 2], [3, 4]], dtype=np.uint16)
        result = measure_z0_mask(_write_mask(tmp_path / _mask_name(), arr))
        assert result["n_objects"] == 4
        assert result["coverage_fraction"] == pytest.approx(1.0)

    def test_all_background_frame_does_not_divide_by_zero(self, tmp_path):
        arr = np.zeros((4, 4), dtype=np.uint16)
        result = measure_z0_mask(_write_mask(tmp_path / _mask_name(), arr))
        assert result["n_objects"] == 0
        assert result["coverage_fraction"] == 0.0
        assert np.isnan(result["mean_object_area_fraction"])

    def test_flags_uint8_width_boundary(self, tmp_path):
        arr = np.zeros((4, 4), dtype=np.uint8)
        arr[0, 0] = 255
        result = measure_z0_mask(_write_mask(tmp_path / _mask_name(), arr, np.uint8))
        assert result["dtype"] == "uint8"
        assert result["dtype_saturated"] is True

    def test_uint16_below_boundary_is_not_flagged(self, tmp_path):
        arr = np.zeros((4, 4), dtype=np.uint16)
        arr[0, 0] = 300
        result = measure_z0_mask(_write_mask(tmp_path / _mask_name(), arr, np.uint16))
        assert result["dtype"] == "uint16"
        assert result["dtype_saturated"] is False

    def test_uint8_below_boundary_is_not_flagged(self, tmp_path):
        arr = np.zeros((4, 4), dtype=np.uint8)
        arr[0, 0] = 12
        result = measure_z0_mask(_write_mask(tmp_path / _mask_name(), arr, np.uint8))
        assert result["dtype_saturated"] is False

    def test_rejects_a_non_2d_mask(self, tmp_path):
        """Measuring a volume would divide coverage by the plane count, silently."""
        arr = np.zeros((2, 4, 4), dtype=np.uint16)
        arr[0, 0, 0] = 1
        path = _write_mask(tmp_path / _mask_name(), arr)
        with pytest.raises(ValueError, match="2-D projection mask"):
            measure_z0_mask(path)

    def test_reads_a_zarr_store(self, tmp_path):
        """Production masks are .zarr only; the tif path is a test convenience."""
        arr = np.zeros((10, 10), dtype=np.uint16)
        arr[0, 0] = 1
        arr[1, 1] = 2
        arr[2, 2] = 3
        path = tmp_path / "pMF5V1_C09_t1_z0_pred_mask.zarr"
        save_labels(arr, path)

        result = measure_z0_mask(path)

        assert result["n_objects"] == 3
        assert result["coverage_fraction"] == pytest.approx(3 / 100)

    def test_flags_uint32_width_boundary(self, tmp_path):
        arr = np.zeros((4, 4), dtype=np.uint32)
        arr[0, 0] = 4294967295
        result = measure_z0_mask(_write_mask(tmp_path / _mask_name(), arr, np.uint32))
        assert result["dtype_saturated"] is True

    def test_flags_a_count_sitting_on_a_wrap_boundary(self, tmp_path):
        """A wrap whose top label misses the boundary still lands the count on it."""
        arr = np.zeros((16, 16), dtype=np.uint16)
        arr.flat[:255] = np.arange(1, 256)
        result = measure_z0_mask(_write_mask(tmp_path / _mask_name(), arr, np.uint16))
        assert result["n_objects"] == 255
        assert result["dtype_saturated"] is True


class TestComputeZ0Population:
    def _populate(self, directory, wells=("C09", "N11"), timepoints=(1, 11)):
        directory.mkdir(parents=True, exist_ok=True)
        for well in wells:
            for tp in timepoints:
                arr = np.zeros((10, 10), dtype=np.uint16)
                arr[0, : len(well)] = np.arange(1, len(well) + 1)
                _write_mask(directory / _mask_name(well, tp, 0), arr)
                # an optical slice that must be ignored
                _write_mask(directory / _mask_name(well, tp, 7), arr)
        return directory

    def test_one_row_per_well_timepoint(self, tmp_path):
        masks = self._populate(tmp_path / "masks")
        df = compute_z0_population(masks)
        assert len(df) == 4
        assert set(df["sample_id"]) == {"C09", "N11"}
        assert set(df["timepoint"]) == {1, 11}

    def test_ignores_other_z_planes(self, tmp_path):
        masks = self._populate(tmp_path / "masks")
        df = compute_z0_population(masks)
        assert len(df) == 4, "only z0 must be measured"

    def test_sorted_by_well_then_time(self, tmp_path):
        masks = self._populate(tmp_path / "masks", timepoints=(11, 1))
        df = compute_z0_population(masks)
        assert list(df["sample_id"]) == ["C09", "C09", "N11", "N11"]
        assert list(df["ti"]) == [1, 11, 1, 11]

    def test_ti_is_integer_not_negative_one(self, tmp_path):
        """Guards the float-timepoint / str.isdigit() trap that produced ti = -1."""
        masks = self._populate(tmp_path / "masks")
        df = compute_z0_population(masks)
        assert df["ti"].dtype.kind == "i"
        assert (df["ti"] > 0).all()

    def test_annotates_drug_from_layout(self, tmp_path):
        """Matches the shipped cell_population semantics exactly.

        A control well's ``drug`` is the literal ``"control"`` (verified against
        results/dataset_analysis/HD1883/cell_population/cell_population.csv, which
        records ``drug=control, is_dmso=True`` for N11) — the vehicle identity is
        carried by ``is_dmso``, not by ``drug``.
        """
        masks = self._populate(tmp_path / "masks")
        df = compute_z0_population(masks, layout=LAYOUT)
        by_well = dict(zip(df["sample_id"], df["drug"]))
        assert by_well["C09"] == "Doxorubicin"
        assert by_well["N11"] == "control"

    def test_flags_the_dmso_well(self, tmp_path):
        masks = self._populate(tmp_path / "masks")
        df = compute_z0_population(masks, dmso_well="n11")
        assert df.loc[df["sample_id"] == "N11", "is_dmso"].all()
        assert not df.loc[df["sample_id"] == "C09", "is_dmso"].any()

    def test_unannotatable_well_is_left_none_not_fatal(self, tmp_path):
        masks = self._populate(tmp_path / "masks", wells=("Z99",))
        df = compute_z0_population(masks, layout=LAYOUT)
        assert df["drug"].isna().all()

    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="masks directory"):
            compute_z0_population(tmp_path / "absent")

    def test_directory_without_z0_masks_raises(self, tmp_path):
        masks = tmp_path / "masks"
        masks.mkdir()
        _write_mask(masks / _mask_name(z=7), np.zeros((4, 4), dtype=np.uint16))
        with pytest.raises(FileNotFoundError, match="pred_mask"):
            compute_z0_population(masks)

    def test_warns_when_frames_sit_on_a_width_boundary(self, tmp_path, caplog):
        masks = tmp_path / "masks"
        masks.mkdir()
        arr = np.zeros((4, 4), dtype=np.uint8)
        arr[0, 0] = 255
        _write_mask(masks / _mask_name(), arr, np.uint8)
        with caplog.at_level(logging.WARNING):
            compute_z0_population(masks)
        assert "integer-width boundary" in caplog.text


class TestSaturationReport:
    def test_lists_only_flagged_frames(self, tmp_path):
        masks = tmp_path / "masks"
        masks.mkdir()
        clean = np.zeros((4, 4), dtype=np.uint16)
        clean[0, 0] = 7
        _write_mask(masks / _mask_name("C09", 1), clean, np.uint16)
        wrapped = np.zeros((4, 4), dtype=np.uint8)
        wrapped[0, 0] = 255
        _write_mask(masks / _mask_name("C09", 11), wrapped, np.uint8)

        report = saturation_report(compute_z0_population(masks))

        assert len(report) == 1
        assert report.loc[0, "timepoint"] == 11
        assert report.loc[0, "max_label"] == 255

    def test_empty_when_nothing_is_flagged(self, tmp_path):
        masks = tmp_path / "masks"
        masks.mkdir()
        arr = np.zeros((4, 4), dtype=np.uint16)
        arr[0, 0] = 7
        _write_mask(masks / _mask_name(), arr, np.uint16)
        assert saturation_report(compute_z0_population(masks)).empty

    def test_rejects_a_frame_without_the_audit_column(self):
        import pandas as pd

        with pytest.raises(ValueError, match="dtype_saturated"):
            saturation_report(pd.DataFrame({"sample_id": ["C09"]}))


class TestMaskSelection:
    def test_ignores_intensity_images_beside_the_masks(self, tmp_path):
        """split_data sits next to masks in the staged tree and parses identically."""
        masks = tmp_path / "masks"
        masks.mkdir()
        arr = np.zeros((10, 10), dtype=np.uint16)
        arr[0, 0] = 1
        _write_mask(masks / "pMF5V1_C09_t1_z0_pred_mask.tif", arr)
        for channel in ("BF", "mCherry", "FlipGFP"):
            _write_mask(masks / f"pMF5V1_C09_t1_z0_{channel}.tif", arr)

        df = compute_z0_population(masks)

        assert len(df) == 1, "only the mask may be measured"

    def test_directory_of_only_intensity_images_raises(self, tmp_path):
        split = tmp_path / "split_data"
        split.mkdir()
        arr = np.zeros((4, 4), dtype=np.uint16)
        _write_mask(split / "pMF5V1_C09_t1_z0_BF.tif", arr)
        with pytest.raises(FileNotFoundError, match="pred_mask"):
            compute_z0_population(split)

    def test_duplicate_well_timepoint_key_raises(self, tmp_path):
        masks = tmp_path / "masks"
        masks.mkdir()
        arr = np.zeros((4, 4), dtype=np.uint16)
        arr[0, 0] = 1
        _write_mask(masks / "pMF5V1_C09_t1_z0_pred_mask.tif", arr)
        save_labels(arr, masks / "pMF5V1_C09_t1_z0_pred_mask.zarr")
        with pytest.raises(ValueError, match="share a .well, timepoint. key"):
            compute_z0_population(masks)


class TestUnreadableMasks:
    def test_bad_store_yields_a_nan_row_not_a_dead_sweep(self, tmp_path):
        masks = tmp_path / "masks"
        masks.mkdir()
        good = np.zeros((4, 4), dtype=np.uint16)
        good[0, 0] = 1
        _write_mask(masks / "pMF5V1_C09_t1_z0_pred_mask.tif", good)
        (masks / "pMF5V1_C09_t11_z0_pred_mask.zarr").mkdir()  # empty, unreadable

        df = compute_z0_population(masks)

        assert len(df) == 2
        bad = df[df["ti"] == 11].iloc[0]
        assert np.isnan(bad["n_objects"])
        assert bad["error"] is not None
        assert df[df["ti"] == 1].iloc[0]["error"] is None

    def test_drop_unmeasurable_removes_failed_and_flagged_rows(self, tmp_path):
        masks = tmp_path / "masks"
        masks.mkdir()
        good = np.zeros((4, 4), dtype=np.uint16)
        good[0, 0] = 7
        _write_mask(masks / "pMF5V1_C09_t1_z0_pred_mask.tif", good, np.uint16)
        wrapped = np.zeros((4, 4), dtype=np.uint8)
        wrapped[0, 0] = 255
        _write_mask(masks / "pMF5V1_C09_t11_z0_pred_mask.tif", wrapped, np.uint8)
        (masks / "pMF5V1_C09_t21_z0_pred_mask.zarr").mkdir()

        clean = drop_unmeasurable(compute_z0_population(masks))

        assert list(clean["ti"]) == [1]

    def test_drop_unmeasurable_is_a_no_op_on_a_clean_table(self, tmp_path):
        masks = tmp_path / "masks"
        masks.mkdir()
        arr = np.zeros((4, 4), dtype=np.uint16)
        arr[0, 0] = 7
        _write_mask(masks / "pMF5V1_C09_t1_z0_pred_mask.tif", arr, np.uint16)
        df = compute_z0_population(masks)
        assert len(drop_unmeasurable(df)) == len(df)

    def test_drop_unmeasurable_rejects_a_frame_without_the_columns(self):
        import pandas as pd

        with pytest.raises(ValueError, match="dtype_saturated"):
            drop_unmeasurable(pd.DataFrame({"sample_id": ["C09"]}))
