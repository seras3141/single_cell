"""Tests for per-file error surfacing, loading and unpaired-mask handling.

Covers the feature-extraction defect fixes:

- a partial per-file failure is recorded, logged in the parent (also for
  parallel workers), written to the summary regardless of
  ``save_combined_file``, and fails the CLI entry point;
- code defects (e.g. ``TypeError``) propagate instead of being recorded;
- masks load through ``image_utils.load_labels`` (uint32, zarr) and BF through
  ``load_image``; non-2D or shape-mismatched inputs are per-file errors;
- an unpaired mask is an error unless registered as known-missing;
- ``_setup_output`` accepts an ``output_dir`` coming only from config.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

import src.feature_extraction.feature_extraction_pipeline as fep
from src.feature_extraction.feature_extraction_pipeline import (
    FeatureExtractionError,
    FeatureExtractionPipeline,
)
from src.utils.data_exclusions import DataExclusions, KnownMissing
from src.utils.image_utils import save_labels

EXP = "HD1883 MF5V1 0-72h 20-03-26"


def _mask() -> np.ndarray:
    mask = np.zeros((32, 32), dtype=np.uint16)
    mask[2:10, 2:10] = 1
    mask[15:25, 15:28] = 2
    return mask


def _image() -> np.ndarray:
    return (np.random.RandomState(0).rand(32, 32) * 1000).astype(np.uint16)


def _dataset(root: Path, names, mask_suffix: str = "_pred_mask.tif"):
    img_dir, msk_dir = root / "imgs", root / "msks"
    img_dir.mkdir(parents=True)
    msk_dir.mkdir(parents=True)
    for name in names:
        tifffile.imwrite(img_dir / f"{name}_BF.tif", _image())
        save_labels(_mask(), msk_dir / f"{name}{mask_suffix}")
    return img_dir, msk_dir


def _pipeline(tmp_path, n_jobs=1, exclusions=None, **output):
    return FeatureExtractionPipeline(
        config={
            "method": "regionprops",
            "n_jobs": n_jobs,
            "output": {
                "save_individual_files": False,
                "create_subdirs": False,
                **output,
            },
        },
        output_dir=str(tmp_path / "out"),
        exclusions=exclusions if exclusions is not None else DataExclusions.empty(),
    )


def _run(pipeline, img_dir, msk_dir, mask_pattern="*_pred_mask.tif"):
    return pipeline.process_batch(
        img_dir, msk_dir, image_patterns=["*_BF.tif"], mask_patterns=[mask_pattern]
    )


class TestPartialFailure:
    @pytest.mark.parametrize("n_jobs", [1, 2])
    def test_bad_file_recorded_and_others_extracted(self, tmp_path, n_jobs, caplog):
        img_dir, msk_dir = _dataset(tmp_path, ["a", "b", "c"])
        # A 3-page stack where a 2D mask is expected.
        tifffile.imwrite(msk_dir / "b_pred_mask.tif", np.stack([_mask()] * 3))
        pipeline = _pipeline(tmp_path, n_jobs=n_jobs)

        with caplog.at_level(logging.ERROR, logger=fep.__name__):
            df = _run(pipeline, img_dir, msk_dir)

        assert len(df) == 4  # a + c, two cells each
        assert len(pipeline.error_files) == 1
        path, message = pipeline.error_files[0]
        assert path.endswith("b_BF.tif") and "2D" in message
        # Logged in the parent process (the run log), also for loky workers.
        assert any("b_BF.tif" in r.getMessage() for r in caplog.records)

    def test_summary_written_without_combined_file(self, tmp_path):
        img_dir, msk_dir = _dataset(tmp_path, ["a", "b"])
        tifffile.imwrite(msk_dir / "b_pred_mask.tif", np.stack([_mask()] * 3))
        pipeline = _pipeline(tmp_path, save_combined_file=False)
        df = _run(pipeline, img_dir, msk_dir)
        pipeline.save_combined_features(df)
        summary = pipeline.save_summary(df)

        text = summary.read_text()
        assert not (pipeline.output_dir / "all_features.csv").exists()
        assert "Files processed: 1" in text
        assert "Files with errors: 1" in text
        assert "b_BF.tif" in text
        with pytest.raises(FeatureExtractionError, match="1 input file"):
            pipeline.raise_if_errors(summary)

    def test_no_errors_does_not_raise(self, tmp_path):
        img_dir, msk_dir = _dataset(tmp_path, ["a"])
        pipeline = _pipeline(tmp_path)
        _run(pipeline, img_dir, msk_dir)
        pipeline.raise_if_errors()


class TestCodeDefectsPropagate:
    def test_name_error_is_raised_not_recorded(self, tmp_path, monkeypatch):
        img_dir, msk_dir = _dataset(tmp_path, ["a", "b"])

        def broken(mask, intensity_image=None):
            raise NameError("backend bug")

        monkeypatch.setattr(fep, "get_region_properties", broken)
        pipeline = _pipeline(tmp_path)
        with pytest.raises(NameError, match="backend bug"):
            _run(pipeline, img_dir, msk_dir)

    def test_type_error_from_data_is_recorded(self, tmp_path, monkeypatch):
        # Numeric libraries raise TypeError for bad data; one file must not
        # abort the batch.
        img_dir, msk_dir = _dataset(tmp_path, ["a", "b"])
        real = fep.get_region_properties
        calls = []

        def flaky(mask, intensity_image=None):
            calls.append(1)
            if len(calls) == 1:
                raise TypeError("bad data")
            return real(mask, intensity_image=intensity_image)

        monkeypatch.setattr(fep, "get_region_properties", flaky)
        pipeline = _pipeline(tmp_path)
        assert len(_run(pipeline, img_dir, msk_dir)) == 2
        assert [m for _, m in pipeline.error_files] == ["bad data"]

    def test_run_raises_on_file_errors(self, tmp_path):
        img_dir, msk_dir = _dataset(tmp_path, ["a", "b"])
        tifffile.imwrite(msk_dir / "b_pred_mask.tif", np.stack([_mask()] * 3))
        pipeline = _pipeline(tmp_path)
        with pytest.raises(FeatureExtractionError):
            pipeline.run([img_dir], [msk_dir])
        assert (pipeline.output_dir / "feature_extraction_summary.txt").exists()


class TestLoading:
    def test_uint32_mask(self, tmp_path):
        img_dir, msk_dir = _dataset(tmp_path, ["a"])
        mask = _mask().astype(np.uint32)
        mask[mask == 2] = 70_000  # > uint16, so save_labels keeps uint32
        save_labels(mask, msk_dir / "a_pred_mask.tif")
        assert tifffile.imread(msk_dir / "a_pred_mask.tif").dtype == np.uint32

        pipeline = _pipeline(tmp_path)
        df = _run(pipeline, img_dir, msk_dir)
        assert sorted(df["cell_id"]) == [1, 70_000]
        assert not pipeline.error_files

    def test_zarr_mask(self, tmp_path):
        pytest.importorskip("zarr")
        img_dir, msk_dir = _dataset(tmp_path, ["a"], mask_suffix="_pred_mask.zarr")
        pipeline = _pipeline(tmp_path)
        df = _run(pipeline, img_dir, msk_dir, mask_pattern="*_pred_mask.zarr")
        assert len(df) == 2
        assert not pipeline.error_files

    def test_float_mask_is_error(self, tmp_path):
        img_dir, msk_dir = _dataset(tmp_path, ["a"])
        tifffile.imwrite(msk_dir / "a_pred_mask.tif", _mask().astype(np.float32))
        pipeline = _pipeline(tmp_path)
        assert _run(pipeline, img_dir, msk_dir).empty
        assert "integer label dtype" in pipeline.error_files[0][1]

    def test_shape_mismatch_is_error(self, tmp_path):
        img_dir, msk_dir = _dataset(tmp_path, ["a"])
        tifffile.imwrite(img_dir / "a_BF.tif", np.zeros((16, 16), dtype=np.uint16))
        pipeline = _pipeline(tmp_path)
        assert _run(pipeline, img_dir, msk_dir).empty
        assert "shapes differ" in pipeline.error_files[0][1]

    def test_new_loader_matches_raw_arrays(self, tmp_path):
        img_dir, msk_dir = _dataset(tmp_path, ["a"])
        image, mask = FeatureExtractionPipeline._load_pair(
            img_dir / "a_BF.tif", msk_dir / "a_pred_mask.tif"
        )
        np.testing.assert_array_equal(image, _image())
        np.testing.assert_array_equal(mask, _mask())  # save_labels may narrow dtype
        assert image.dtype == np.uint16


class TestUnpairedMasks:
    def _layout(self, tmp_path):
        root = tmp_path / EXP
        img_dir, msk_dir = _dataset(root, ["pMF5V1_H09_t201_z3", "pMF5V1_H09_t201_z5"])
        save_labels(_mask(), msk_dir / "pMF5V1_H09_t201_z4_pred_mask.tif")
        return img_dir, msk_dir

    def test_unpaired_mask_is_error(self, tmp_path):
        img_dir, msk_dir = self._layout(tmp_path)
        pipeline = _pipeline(tmp_path)
        df = _run(pipeline, img_dir, msk_dir)
        assert len(df) == 4
        assert [msg for _, msg in pipeline.error_files] == ["No matching image"]
        assert pipeline.error_files[0][0].endswith("z4_pred_mask.tif")

    def test_known_missing_is_not_error(self, tmp_path):
        img_dir, msk_dir = self._layout(tmp_path)
        registry = DataExclusions(
            known_missing=(KnownMissing(EXP, "H09", 201, 4, "BF", "absent"),)
        )
        pipeline = _pipeline(tmp_path, exclusions=registry)
        _run(pipeline, img_dir, msk_dir)
        assert not pipeline.error_files
        assert len(pipeline.expected_unpaired) == 1
        summary = pipeline.save_summary(pd.DataFrame()).read_text()
        assert "Expected unpaired masks: 1" in summary

    def test_known_missing_other_channel_is_still_error(self, tmp_path):
        img_dir, msk_dir = self._layout(tmp_path)
        registry = DataExclusions(
            known_missing=(KnownMissing(EXP, "H09", 201, 4, "mCherry", "absent"),)
        )
        pipeline = _pipeline(tmp_path, exclusions=registry)
        _run(pipeline, img_dir, msk_dir)
        assert len(pipeline.error_files) == 1

    def test_images_without_mask_are_not_errors(self, tmp_path):
        img_dir, msk_dir = _dataset(tmp_path, ["a"])
        tifffile.imwrite(img_dir / "z0only_BF.tif", _image())
        pipeline = _pipeline(tmp_path)
        _run(pipeline, img_dir, msk_dir)
        assert not pipeline.error_files


def test_output_dir_from_config_only(tmp_path):
    pipeline = FeatureExtractionPipeline(
        config={
            "method": "incarta",
            "output": {"output_dir": str(tmp_path / "cfg_out")},
        },
        exclusions=DataExclusions.empty(),
    )
    assert isinstance(pipeline.output_dir, Path)
    assert pipeline.output_dir.is_dir()


@pytest.mark.parametrize(
    "pattern, channel, expected",
    [
        ("*_BF.tif", "BF", True),
        ("*_BF_3d.tif", "BF", True),
        ("*_mCherry.tif", "BF", False),
        ("*_BFX.tif", "BF", False),
    ],
)
def test_pattern_has_channel(pattern, channel, expected):
    assert fep._pattern_has_channel(pattern, channel) is expected


def test_cli_writes_summary_when_code_defect_propagates(tmp_path, monkeypatch):
    from tests.feature_extraction.test_run_feature_extraction_cli import (
        _load_script_module,
    )

    img_dir, msk_dir = _dataset(tmp_path, ["a"])

    def broken(mask, intensity_image=None):
        raise NameError("backend bug")

    monkeypatch.setattr(fep, "get_region_properties", broken)
    module = _load_script_module()
    out = tmp_path / "cli_out"
    config = {
        "paths": {
            "image_dir": str(img_dir),
            "mask_dir": str(msk_dir),
            "output_dir": str(out),
        },
        "feature_extraction": {
            "method": "regionprops",
            "n_jobs": 1,
            "image_pattern": "*_BF.tif",
            "mask_pattern": "*_pred_mask.tif",
            "output": {"save_individual_files": False},
        },
        "logging": {},
    }
    with pytest.raises(NameError):
        module.run_feature_extraction_from_config(config)
    assert (out / "feature_extraction_summary.txt").exists()


def test_default_patterns_match_mf5v1_layout(tmp_path):
    img_dir, msk_dir = _dataset(tmp_path, ["pMF5V1_E07_t1_z1"])
    pipeline = _pipeline(tmp_path)
    df = pipeline.process_batch(img_dir, msk_dir)  # no patterns passed
    assert len(df) == 2


def test_no_pairs_is_an_error(tmp_path):
    img_dir, msk_dir = _dataset(tmp_path, ["a"], mask_suffix="_Cells.tif")
    pipeline = _pipeline(tmp_path)
    assert pipeline.process_batch(img_dir, msk_dir).empty
    assert pipeline.error_files == [(str(msk_dir), "No valid image-mask pairs")]


def test_run_treats_null_patterns_as_defaults(tmp_path):
    img_dir, msk_dir = _dataset(tmp_path, ["a"])
    pipeline = FeatureExtractionPipeline(
        config={
            "method": "regionprops",
            "n_jobs": 1,
            "image_pattern": None,
            "mask_pattern": None,
            "output": {"save_individual_files": False},
        },
        output_dir=str(tmp_path / "out"),
        exclusions=DataExclusions.empty(),
    )
    assert len(pipeline.run([img_dir], [msk_dir])) == 2


def test_scportrait_no_images_is_an_error(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    pipeline = FeatureExtractionPipeline(
        config={"method": "scportrait", "output": {}},
        output_dir=str(tmp_path / "out"),
        exclusions=DataExclusions.empty(),
    )
    assert pipeline.process_batch_scportrait(empty).empty
    assert pipeline.error_files == [(str(empty), "No images found")]
