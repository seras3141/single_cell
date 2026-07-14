"""
Unit tests for exporting scPortrait ``.sdata`` labels to a Cellpose-style TIFF.

These cover the converter added to ``feature_extractor_scportrait`` and the
sample-folder-derived export path built by the pipeline. scPortrait is not
installed in CI, so every scPortrait object is faked/mocked; no GPU, no
spatialdata, no real data files.
"""

import numpy as np
import pandas as pd
import pytest
import tifffile
from unittest.mock import MagicMock, patch

from src.feature_extraction import feature_extractor_scportrait as fes
from src.feature_extraction.feature_extractor_scportrait import (
    _export_scportrait_labels_to_tif,
    _resolve_fullres_label_array,
    get_scportrait_features,
)
from src.feature_extraction.feature_extraction_pipeline import (
    SCPORTRAIT_MASK_ROOT_NAME,
    FeatureExtractionPipeline,
    derive_sample_dir,
)


# Dummy workflow classes carrying the right ``__name__`` for the key lookup.
_DummySeg = type("CytosolOnlySegmentationCellpose", (), {})
_DummyExt = type("HDF5CellExtraction", (), {})
_DummyFeat = type("ConvNeXtFeaturizer", (), {})


def _label_array() -> np.ndarray:
    """A small 2D label image with a few distinct non-zero cell IDs."""
    arr = np.zeros((6, 6), dtype=np.uint32)
    arr[0:2, 0:2] = 5
    arr[3:5, 3:5] = 12
    arr[0:2, 4:6] = 7
    return arr


# ---------------------------------------------------------------------------
# _resolve_fullres_label_array
# ---------------------------------------------------------------------------


class TestResolveFullresLabelArray:
    def test_plain_array_passthrough(self):
        """A plain array-like is returned unchanged (as a 2D array)."""
        arr = _label_array()
        out = _resolve_fullres_label_array(arr)
        assert out.shape == (6, 6)
        np.testing.assert_array_equal(out, arr)

    def test_multiscale_selects_finest_scale(self):
        """A multiscale mapping resolves to the finest scale (scale0)."""
        full = _label_array()  # 6x6 -> finest
        half = full[::2, ::2]  # 3x3 -> coarser
        multiscale = {
            "scale0": {"image": full},
            "scale1": {"image": half},
        }
        out = _resolve_fullres_label_array(multiscale)
        assert out.shape == (6, 6)
        np.testing.assert_array_equal(out, full)

    def test_multiscale_s_prefix(self):
        """Scale keys named s0/s1 (zarr-style) are handled too."""
        full = _label_array()
        multiscale = {"s0": {"image": full}, "s1": {"image": full[::2, ::2]}}
        out = _resolve_fullres_label_array(multiscale)
        assert out.shape == (6, 6)

    def test_squeezes_leading_channel_axis(self):
        """A singleton leading (channel) axis is squeezed to 2D."""
        arr = _label_array()[np.newaxis, ...]  # (1, 6, 6)
        out = _resolve_fullres_label_array(arr)
        assert out.shape == (6, 6)


# ---------------------------------------------------------------------------
# _export_scportrait_labels_to_tif
# ---------------------------------------------------------------------------


def _project_with_labels(labels: dict) -> MagicMock:
    project = MagicMock()
    project.sdata.labels = labels  # real dict -> real .keys()/[key]
    return project


class TestExportLabelsToTif:
    def test_roundtrip_preserves_cell_ids(self, tmp_path):
        """Exported TIFF's unique non-zero values == the label cell IDs."""
        arr = _label_array()
        project = _project_with_labels({"seg_all_cytosol": arr})
        out = tmp_path / "sub" / "mask_pred_mask.tif"

        _export_scportrait_labels_to_tif(project, out)

        assert out.exists()
        written = tifffile.imread(str(out))
        assert written.shape == arr.shape
        # Downcast to uint16 to match the pipeline's inference_tracked masks.
        assert written.dtype == np.uint16
        assert set(np.unique(written[written > 0])) == {5, 7, 12}

    def test_large_labels_fall_back_to_int32(self, tmp_path):
        """Cell IDs beyond uint16 range use int32 (still lossless)."""
        arr = np.zeros((4, 4), dtype=np.uint64)
        arr[0, 0] = 70000  # > uint16 max (65535)
        project = _project_with_labels({"seg_all_cytosol": arr})
        out = tmp_path / "mask.tif"
        _export_scportrait_labels_to_tif(project, out)
        written = tifffile.imread(str(out))
        assert written.dtype == np.int32
        assert 70000 in np.unique(written)

    def test_creates_parent_dirs(self, tmp_path):
        project = _project_with_labels({"seg_all_cytosol": _label_array()})
        out = tmp_path / "a" / "b" / "c" / "mask.tif"
        _export_scportrait_labels_to_tif(project, out)
        assert out.exists()

    def test_multiple_keys_uses_first(self, tmp_path):
        """With >1 label key, the first is exported (and a warning logged)."""
        first = _label_array()
        second = np.full((6, 6), 99, dtype=np.uint32)
        project = _project_with_labels(
            {"seg_all_cytosol": first, "seg_all_nucleus": second}
        )
        out = tmp_path / "mask.tif"
        _export_scportrait_labels_to_tif(project, out)
        written = tifffile.imread(str(out))
        assert 99 not in np.unique(written)

    def test_no_labels_raises(self, tmp_path):
        project = _project_with_labels({})
        with pytest.raises(RuntimeError, match="No segmentation labels"):
            _export_scportrait_labels_to_tif(project, tmp_path / "mask.tif")

    def test_non_integer_dtype_is_cast(self, tmp_path):
        """A float label array is cast to integer before writing."""
        arr = _label_array().astype(np.float32)
        project = _project_with_labels({"seg_all_cytosol": arr})
        out = tmp_path / "mask.tif"
        _export_scportrait_labels_to_tif(project, out)
        assert np.issubdtype(tifffile.imread(str(out)).dtype, np.integer)


# ---------------------------------------------------------------------------
# get_scportrait_features export hook
# ---------------------------------------------------------------------------


def _full_mock_project(result_key="ConvNeXtFeaturizer_run") -> MagicMock:
    df = pd.DataFrame({"feat1": [1.0], "feat2": [2.0]})
    mock_table = MagicMock()
    mock_table.to_df.return_value = df
    mock_table.obs = {"scportrait_cell_id": [5]}

    project = MagicMock()
    project.sdata.tables = {result_key: mock_table}
    project.sdata.labels = {"seg_all_cytosol": _label_array()}
    return project


class TestGetScportraitFeaturesExport:
    def test_exports_when_path_given(self, tmp_path):
        out = (
            tmp_path
            / "inference_scportrait"
            / "scportrait"
            / "test"
            / "final_2d"
            / "img_pred_mask.tif"
        )
        with patch.object(fes, "Project", return_value=_full_mock_project()):
            get_scportrait_features(
                image_paths=["bf.tif", "bf.tif"],
                channel_names=["brightfield", "brightfield_ch1"],
                config_path="config.yml",
                project_location="proj",
                segmentation_f=_DummySeg,
                extraction_f=_DummyExt,
                featurization_f=_DummyFeat,
                plots_dir=None,
                scportrait_mask_export_path=str(out),
            )
        assert out.exists()
        written = tifffile.imread(str(out))
        assert set(np.unique(written[written > 0])) == {5, 7, 12}

    def test_no_export_when_path_none(self, tmp_path):
        with patch.object(fes, "Project", return_value=_full_mock_project()):
            get_scportrait_features(
                image_paths=["bf.tif", "bf.tif"],
                channel_names=["brightfield", "brightfield_ch1"],
                config_path="config.yml",
                project_location="proj",
                segmentation_f=_DummySeg,
                extraction_f=_DummyExt,
                featurization_f=_DummyFeat,
                plots_dir=None,
                scportrait_mask_export_path=None,
            )
        # No stray TIFFs written anywhere under tmp_path.
        assert list(tmp_path.rglob("*.tif")) == []


# ---------------------------------------------------------------------------
# derive_sample_dir + pipeline export-path wiring
# ---------------------------------------------------------------------------


def _make_sample_folder(tmp_path, *markers):
    """Create a fake sample folder with the given marker children + an image."""
    sample = tmp_path / "HD1509 MF5V1 0-72h 23-02-26"
    split = sample / "split_data"
    split.mkdir(parents=True)
    for m in markers:
        (sample / m).mkdir(exist_ok=True)
    image = split / "pMF5V1_F09_t131_z20_BF.tif"
    image.write_bytes(b"x")
    return sample, image


class TestDeriveSampleDir:
    def test_finds_sample_dir_via_split_data(self, tmp_path):
        sample, image = _make_sample_folder(tmp_path, "inference_tracked")
        assert derive_sample_dir(image) == sample.resolve()

    def test_works_without_inference_tracked(self, tmp_path):
        """HD1883-like: no inference_tracked, but split_data still marks it."""
        sample, image = _make_sample_folder(tmp_path)  # only split_data
        assert derive_sample_dir(image) == sample.resolve()

    def test_returns_none_when_no_markers(self, tmp_path):
        image = tmp_path / "loose" / "img_BF.tif"
        image.parent.mkdir()
        image.write_bytes(b"x")
        assert derive_sample_dir(image) is None

    def test_symlinked_image_stays_in_processed_tree(self, tmp_path):
        """split_data BF images are symlinks into the raw dataset; derivation
        must resolve the dir (processed side), not follow the file symlink."""
        # Raw tree: a marker-less experiment folder with the real image.
        raw = tmp_path / "raw" / "Ew2-1 raw"
        raw.mkdir(parents=True)
        real_image = raw / "t101_C09_s1_w3_z10.tif"
        real_image.write_bytes(b"x")
        # Processed tree: sample folder with markers; split_data holds a symlink
        # into the raw tree (as the real processed datasets do).
        sample, _ = _make_sample_folder(tmp_path, "inference_tracked")
        link = sample / "split_data" / "pMF5V1_C09_t101_z10_BF.tif"
        link.symlink_to(real_image)
        assert derive_sample_dir(link) == sample.resolve()


class TestScportraitMaskExportPath:
    def _pipeline(self, tmp_path, sc_cfg):
        config = {"method": "scportrait", "scportrait": sc_cfg, "output": {}}
        return FeatureExtractionPipeline(
            config=config, output_dir=str(tmp_path / "out")
        )

    def test_derived_path_mirrors_inference_tracked(self, tmp_path):
        sample, image = _make_sample_folder(tmp_path, "inference_tracked")
        pipeline = self._pipeline(tmp_path, {})
        path = pipeline._scportrait_mask_export_path(image, {})
        assert path == (
            sample
            / SCPORTRAIT_MASK_ROOT_NAME
            / "scportrait"
            / "test"
            / "final_2d"
            / "pMF5V1_F09_t131_z20_pred_mask.tif"
        )

    def test_override_root(self, tmp_path):
        _, image = _make_sample_folder(tmp_path, "inference_tracked")
        override = tmp_path / "custom_root"
        pipeline = self._pipeline(tmp_path, {"mask_export_root": str(override)})
        path = pipeline._scportrait_mask_export_path(
            image, {"mask_export_root": str(override)}
        )
        assert str(path).startswith(str(override))
        assert path.name == "pMF5V1_F09_t131_z20_pred_mask.tif"

    def test_none_when_no_sample_dir(self, tmp_path):
        image = tmp_path / "loose" / "img_BF.tif"
        image.parent.mkdir()
        image.write_bytes(b"x")
        pipeline = self._pipeline(tmp_path, {})
        assert pipeline._scportrait_mask_export_path(image, {}) is None

    def test_dispatch_passes_export_path_kwarg(self, tmp_path):
        """The scPortrait dispatch forwards a derived mask-export path."""
        sample, image = _make_sample_folder(tmp_path, "inference_tracked")
        pipeline = self._pipeline(
            tmp_path,
            {"channel_names": ["brightfield", "brightfield_ch1"], "save_plots": False},
        )
        returned = pd.DataFrame({"feat1": [1.0], "scportrait_cell_id": [0]})
        mock_fn = MagicMock(return_value=returned)
        with patch(
            "src.feature_extraction.feature_extraction_pipeline."
            "get_scportrait_features",
            mock_fn,
        ):
            pipeline.extract_features_from_path(image, None)

        _, kwargs = mock_fn.call_args
        exported = kwargs["scportrait_mask_export_path"]
        assert exported is not None
        assert exported.endswith(
            "inference_scportrait/scportrait/test/final_2d/"
            "pMF5V1_F09_t131_z20_pred_mask.tif"
        )
