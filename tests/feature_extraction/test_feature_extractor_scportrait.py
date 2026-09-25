"""
Unit tests for the scPortrait feature extractor and its pipeline integration.

scPortrait is an optional, GPU/Cellpose-heavy dependency that is not required in
CI. These tests mock scPortrait's ``Project`` class (and the pipeline-level
``get_scportrait_features`` hook) so they run without scportrait installed.
"""

import logging

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.feature_extraction import feature_extractor_scportrait as fes
from src.feature_extraction.feature_extractor_scportrait import get_scportrait_features
from src.feature_extraction.feature_extraction_pipeline import (
    FeatureExtractionPipeline,
    resolve_cellpose_mask,
)


def _make_pipeline(tmp_path, output=None):
    """Build a scPortrait pipeline pointed at a temp output dir.

    ``output`` overrides the ``output`` config block (save gates etc.).
    """
    config = {
        "method": "scportrait",
        "scportrait": {
            "project_location": str(tmp_path / "scportrait_projects"),
            "config_path": "src/feature_extraction/scportrait_project/config.yml",
            "channel_names": ["brightfield", "brightfield_ch1"],
        },
        "output": output if output is not None else {},
    }
    return FeatureExtractionPipeline(config=config, output_dir=str(tmp_path / "out"))


def _make_mock_project(result_key="ConvNeXtFeaturizer_run"):
    """Build a mock scPortrait Project whose featurization table is non-empty."""
    df = pd.DataFrame({"feat1": [1.0, 2.0, 3.0], "feat2": [4.0, 5.0, 6.0]})

    mock_table = MagicMock()
    mock_table.to_df.return_value = df
    mock_table.obs = {"scportrait_cell_id": [10, 20, 30]}

    mock_project = MagicMock()
    # Real dict so .keys()/[key] behave like scPortrait's sdata.tables.
    mock_project.sdata.tables = {result_key: mock_table}
    return mock_project


# Dummy workflow classes carrying the right ``__name__`` for the key lookup.
_DummySeg = type("CytosolOnlySegmentationCellpose", (), {})
_DummyExt = type("HDF5CellExtraction", (), {})
_DummyFeat = type("ConvNeXtFeaturizer", (), {})


class TestGetScportraitFeatures:
    """Tests for the standalone get_scportrait_features() function."""

    def test_get_scportrait_features_returns_dataframe(self):
        """With Project mocked, returns a non-empty DataFrame incl. cell ids."""
        mock_project = _make_mock_project()
        with patch.object(fes, "Project", return_value=mock_project):
            result = get_scportrait_features(
                image_paths=["bf.tif", "bf.tif"],
                channel_names=["brightfield", "brightfield_ch1"],
                config_path="config.yml",
                project_location="proj",
                segmentation_f=_DummySeg,
                extraction_f=_DummyExt,
                featurization_f=_DummyFeat,
                plots_dir=None,
            )

        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert len(result) == 3
        assert "scportrait_cell_id" in result.columns
        assert list(result["scportrait_cell_id"]) == [10, 20, 30]

    def test_missing_scportrait_raises(self):
        """If scportrait is unavailable (Project is None), raise RuntimeError."""
        with patch.object(fes, "Project", None):
            with pytest.raises(RuntimeError, match="scportrait is not installed"):
                get_scportrait_features(
                    image_paths=["bf.tif", "bf.tif"],
                    channel_names=["brightfield", "brightfield_ch1"],
                    config_path="config.yml",
                    project_location="proj",
                )


class TestPipelineDispatch:
    """Tests for FeatureExtractionPipeline dispatch of method='scportrait'."""

    def test_pipeline_dispatch_scportrait(self, tmp_path):
        """method='scportrait' calls get_scportrait_features with expected args."""
        image_path = tmp_path / "p2426_B01_z10_BF.tif"
        mask_path = tmp_path / "p2426_B01_z10_pred_mask.tif"
        image_path.write_bytes(b"fake")
        mask_path.write_bytes(b"fake")

        config = {
            "method": "scportrait",
            "scportrait": {
                "project_location": str(tmp_path / "scportrait_projects"),
                "config_path": "src/feature_extraction/scportrait_project/config.yml",
                "channel_names": ["brightfield", "brightfield_ch1"],
                "overwrite": True,
                "debug": False,
                "save_plots": False,
            },
            "output": {"include_metadata": False},
        }
        pipeline = FeatureExtractionPipeline(
            config=config, output_dir=str(tmp_path / "out")
        )

        returned_df = pd.DataFrame({"feat1": [1.0], "scportrait_cell_id": [0]})
        mock_fn = MagicMock(return_value=returned_df)
        with patch(
            "src.feature_extraction.feature_extraction_pipeline.get_scportrait_features",
            mock_fn,
        ):
            result = pipeline.extract_features_from_path(image_path, mask_path)

        assert result is not None
        assert not result.empty
        mock_fn.assert_called_once()
        _, kwargs = mock_fn.call_args
        assert kwargs["image_paths"] == [str(image_path), str(image_path)]
        assert kwargs["channel_names"] == ["brightfield", "brightfield_ch1"]
        assert kwargs["config_path"].endswith("scportrait_project/config.yml")
        # project_location must be namespaced by the image stem.
        assert kwargs["project_location"].endswith(image_path.stem)
        # save_plots False -> plots_dir disabled.
        assert kwargs["plots_dir"] is None

    def test_pipeline_missing_scportrait_raises(self, tmp_path):
        """A missing scPortrait install fails on the first file (ImportError)."""
        image_path = tmp_path / "bf.tif"
        mask_path = tmp_path / "mask.tif"
        image_path.write_bytes(b"fake")
        mask_path.write_bytes(b"fake")

        pipeline = FeatureExtractionPipeline(
            config={"method": "scportrait"}, output_dir=str(tmp_path / "out")
        )
        with patch(
            "src.feature_extraction.feature_extraction_pipeline.get_scportrait_features",
            None,
        ):
            with pytest.raises(ImportError, match="scportrait"):
                pipeline.extract_features_from_path(image_path, mask_path)
        assert not pipeline.error_files


@pytest.mark.unit
class TestFindImages:
    """Tests for FeatureExtractionPipeline.find_images() (mask-free discovery)."""

    def test_default_pattern_matches_only_bf(self, tmp_path):
        """Default pattern (None -> ['*_BF.tif']) excludes other channels."""
        (tmp_path / "pMF5V1_E07_t1_z10_BF.tif").write_bytes(b"x")
        (tmp_path / "pMF5V1_E07_t1_z10_mCherry.tif").write_bytes(b"x")
        (tmp_path / "pMF5V1_E07_t1_z10_FlipGFP.tif").write_bytes(b"x")

        pipeline = _make_pipeline(tmp_path)
        found = pipeline.find_images(tmp_path)

        assert [p.name for p in found] == ["pMF5V1_E07_t1_z10_BF.tif"]

    def test_custom_pattern_narrows_results(self, tmp_path):
        """A more specific pattern selects only the matching subset."""
        (tmp_path / "pMF5V1_E07_t1_z09_BF.tif").write_bytes(b"x")
        (tmp_path / "pMF5V1_E07_t1_z10_BF.tif").write_bytes(b"x")

        pipeline = _make_pipeline(tmp_path)
        found = pipeline.find_images(tmp_path, image_patterns=["*_z10_BF.tif"])

        assert [p.name for p in found] == ["pMF5V1_E07_t1_z10_BF.tif"]

    def test_multiple_patterns_union_and_dedup(self, tmp_path):
        """Overlapping patterns union results with no duplicates."""
        (tmp_path / "a_BF.tif").write_bytes(b"x")
        (tmp_path / "b_BF.tif").write_bytes(b"x")

        pipeline = _make_pipeline(tmp_path)
        # Both patterns match a_BF.tif; it must appear once.
        found = pipeline.find_images(tmp_path, image_patterns=["*_BF.tif", "a_*.tif"])

        names = [p.name for p in found]
        assert names == ["a_BF.tif", "b_BF.tif"]
        assert len(names) == len(set(names))

    def test_recurses_into_subdirs(self, tmp_path):
        """rglob finds images in nested subdirectories."""
        nested = tmp_path / "well_E07" / "t1"
        nested.mkdir(parents=True)
        (nested / "deep_BF.tif").write_bytes(b"x")

        pipeline = _make_pipeline(tmp_path)
        found = pipeline.find_images(tmp_path)

        assert [p.name for p in found] == ["deep_BF.tif"]

    def test_results_are_sorted(self, tmp_path):
        """Output is sorted regardless of creation order."""
        for name in ("c_BF.tif", "a_BF.tif", "b_BF.tif"):
            (tmp_path / name).write_bytes(b"x")

        pipeline = _make_pipeline(tmp_path)
        found = pipeline.find_images(tmp_path)

        assert [p.name for p in found] == ["a_BF.tif", "b_BF.tif", "c_BF.tif"]

    def test_no_match_returns_empty(self, tmp_path):
        """No matching files -> empty list."""
        (tmp_path / "not_an_image.txt").write_bytes(b"x")

        pipeline = _make_pipeline(tmp_path)
        assert pipeline.find_images(tmp_path) == []


@pytest.mark.unit
class TestProcessBatchScportrait:
    """Tests for FeatureExtractionPipeline.process_batch_scportrait()."""

    def _write_bf(self, tmp_path, *names):
        for n in names:
            (tmp_path / n).write_bytes(b"x")

    def test_happy_path_concatenates(self, tmp_path):
        """Each image's DataFrame is concatenated with a reset index."""
        self._write_bf(tmp_path, "a_BF.tif", "b_BF.tif")
        pipeline = _make_pipeline(tmp_path, output={"save_individual_files": False})

        per_image = pd.DataFrame({"feat": [1.0, 2.0]})
        with patch.object(
            pipeline, "extract_features_from_path", return_value=per_image
        ) as mock_extract:
            combined = pipeline.process_batch_scportrait(tmp_path)

        assert mock_extract.call_count == 2
        # 2 images x 2 rows each, index reset (ignore_index=True).
        assert len(combined) == 4
        assert list(combined.index) == [0, 1, 2, 3]
        # scPortrait is mask-free: every call passes mask_path=None.
        for call in mock_extract.call_args_list:
            assert call.kwargs.get("mask_path") is None

    def test_saves_individual_when_enabled(self, tmp_path):
        """save_individual_files truthy -> save_image_features per image."""
        self._write_bf(tmp_path, "a_BF.tif", "b_BF.tif")
        pipeline = _make_pipeline(tmp_path, output={"save_individual_files": True})

        with patch.object(
            pipeline, "extract_features_from_path", return_value=pd.DataFrame({"f": [1]})
        ), patch.object(pipeline, "save_image_features") as mock_save:
            pipeline.process_batch_scportrait(tmp_path)

        assert mock_save.call_count == 2

    def test_skips_individual_when_disabled(self, tmp_path):
        """save_individual_files False -> save_image_features never called."""
        self._write_bf(tmp_path, "a_BF.tif")
        pipeline = _make_pipeline(tmp_path, output={"save_individual_files": False})

        with patch.object(
            pipeline, "extract_features_from_path", return_value=pd.DataFrame({"f": [1]})
        ), patch.object(pipeline, "save_image_features") as mock_save:
            pipeline.process_batch_scportrait(tmp_path)

        mock_save.assert_not_called()

    def test_no_images_returns_empty(self, tmp_path):
        """Empty directory -> empty DataFrame, extractor never invoked."""
        pipeline = _make_pipeline(tmp_path)

        with patch.object(pipeline, "extract_features_from_path") as mock_extract:
            combined = pipeline.process_batch_scportrait(tmp_path)

        assert combined.empty
        mock_extract.assert_not_called()

    def test_all_none_returns_empty(self, tmp_path):
        """When every extraction returns None -> empty DataFrame."""
        self._write_bf(tmp_path, "a_BF.tif", "b_BF.tif")
        pipeline = _make_pipeline(tmp_path, output={"save_individual_files": False})

        with patch.object(pipeline, "extract_features_from_path", return_value=None):
            combined = pipeline.process_batch_scportrait(tmp_path)

        assert combined.empty

    def test_partial_failures_skip_none(self, tmp_path):
        """A mix of None and DataFrames concatenates only the non-None ones."""
        self._write_bf(tmp_path, "a_BF.tif", "b_BF.tif")
        pipeline = _make_pipeline(tmp_path, output={"save_individual_files": False})

        results = [pd.DataFrame({"f": [1.0]}), None]
        with patch.object(
            pipeline, "extract_features_from_path", side_effect=results
        ):
            combined = pipeline.process_batch_scportrait(tmp_path)

        assert len(combined) == 1


@pytest.mark.unit
class TestProcessSingleImage:
    """Tests for FeatureExtractionPipeline.process_single_image()."""

    def test_happy_path_saves_both(self, tmp_path):
        """Non-empty result -> individual + combined saves; mask forwarded."""
        image_path = tmp_path / "a_BF.tif"
        image_path.write_bytes(b"x")
        pipeline = _make_pipeline(tmp_path, output={"save_individual_files": True})

        df = pd.DataFrame({"f": [1.0]})
        with patch.object(
            pipeline, "extract_features_from_path", return_value=df
        ) as mock_extract, patch.object(
            pipeline, "save_image_features"
        ) as mock_indiv, patch.object(
            pipeline, "save_combined_features"
        ) as mock_combined:
            result = pipeline.process_single_image(image_path, mask_path=None)

        assert result is df
        mock_extract.assert_called_once()
        mock_indiv.assert_called_once()
        mock_combined.assert_called_once()

    def test_empty_result_skips_saves(self, tmp_path):
        """Empty DataFrame -> returned as-is, no save methods called."""
        image_path = tmp_path / "a_BF.tif"
        image_path.write_bytes(b"x")
        pipeline = _make_pipeline(tmp_path, output={"save_individual_files": True})

        with patch.object(
            pipeline, "extract_features_from_path", return_value=pd.DataFrame()
        ), patch.object(pipeline, "save_image_features") as mock_indiv, patch.object(
            pipeline, "save_combined_features"
        ) as mock_combined:
            result = pipeline.process_single_image(image_path)

        assert result is not None and result.empty
        mock_indiv.assert_not_called()
        mock_combined.assert_not_called()

    def test_none_result_skips_saves(self, tmp_path):
        """None result -> returned as None, no save methods called."""
        image_path = tmp_path / "a_BF.tif"
        image_path.write_bytes(b"x")
        pipeline = _make_pipeline(tmp_path)

        with patch.object(
            pipeline, "extract_features_from_path", return_value=None
        ), patch.object(pipeline, "save_image_features") as mock_indiv, patch.object(
            pipeline, "save_combined_features"
        ) as mock_combined:
            result = pipeline.process_single_image(image_path)

        assert result is None
        mock_indiv.assert_not_called()
        mock_combined.assert_not_called()

    def test_individual_disabled_still_saves_combined(self, tmp_path):
        """save_individual_files False -> only combined save runs."""
        image_path = tmp_path / "a_BF.tif"
        image_path.write_bytes(b"x")
        pipeline = _make_pipeline(tmp_path, output={"save_individual_files": False})

        with patch.object(
            pipeline, "extract_features_from_path", return_value=pd.DataFrame({"f": [1]})
        ), patch.object(pipeline, "save_image_features") as mock_indiv, patch.object(
            pipeline, "save_combined_features"
        ) as mock_combined:
            pipeline.process_single_image(image_path)

        mock_indiv.assert_not_called()
        mock_combined.assert_called_once()


# ---------------------------------------------------------------------------
# Milestone 2 — external-mask injection (_inject_mask + mask_path branch)
# ---------------------------------------------------------------------------

def _make_inject_project(seg_name="seg_all_cytosol", frame=(1023, 1023)):
    """Mock project whose filehandler supports mask injection.

    Returns (project, filehandler). The filehandler exposes ``cyto_seg_name``,
    ``_get_input_image`` (with a ``.sizes`` frame) and a spy
    ``_write_segmentation_sdata``.
    """
    proj = MagicMock()
    fh = proj.filehandler
    fh.cyto_seg_name = seg_name
    img = MagicMock()
    img.sizes = {"y": frame[0], "x": frame[1]}
    fh._get_input_image.return_value = img
    return proj, fh


class TestInjectMask:
    """Unit tests for the _inject_mask helper (Milestone 2)."""

    def test_crops_offbyone_and_preserves_ids(self):
        """1024 mask + 1023 frame -> bottom-right crop to 1023; IDs preserved."""
        proj, fh = _make_inject_project(frame=(1023, 1023))
        mask = np.zeros((1024, 1024), dtype=np.uint16)
        mask[10:20, 10:20] = 101
        mask[30:40, 30:40] = 777
        with patch.object(fes.tifffile, "imread", return_value=mask):
            n = fes._inject_mask(proj, "m.tif")

        assert n == 2
        fh._write_segmentation_sdata.assert_called_once()
        args, kwargs = fh._write_segmentation_sdata.call_args
        written = args[0]
        assert written.shape == (1023, 1023)  # cropped, not resized
        assert args[1] == "seg_all_cytosol"  # runtime-resolved seg name
        assert set(int(v) for v in np.unique(written)) == {0, 101, 777}

    def test_exact_frame_no_crop(self):
        """Mask already matching the frame is written unchanged."""
        proj, fh = _make_inject_project(frame=(1023, 1023))
        mask = np.zeros((1023, 1023), dtype=np.uint32)
        mask[5:8, 5:8] = 42
        with patch.object(fes.tifffile, "imread", return_value=mask):
            n = fes._inject_mask(proj, "m.tif")

        assert n == 1
        written = fh._write_segmentation_sdata.call_args[0][0]
        assert written.shape == (1023, 1023)

    def test_big_mismatch_raises(self):
        """A >1px frame mismatch refuses to resize integer labels."""
        proj, fh = _make_inject_project(frame=(1023, 1023))
        mask = np.zeros((512, 512), dtype=np.uint32)
        with patch.object(fes.tifffile, "imread", return_value=mask):
            with pytest.raises(ValueError, match=r">1"):
                fes._inject_mask(proj, "m.tif")
        fh._write_segmentation_sdata.assert_not_called()

    def test_smaller_by_one_raises(self):
        """A mask 1px SMALLER than the frame cannot be cropped into shape.

        It slips through the +/-1 tolerance, and ``mask[:ty, :tx]`` is a no-op,
        so without an explicit check a misregistered label array is written.
        """
        proj, fh = _make_inject_project(frame=(1024, 1024))
        mask = np.zeros((1023, 1023), dtype=np.uint32)
        mask[5:8, 5:8] = 9
        with patch.object(fes.tifffile, "imread", return_value=mask):
            with pytest.raises(ValueError, match="smaller than image frame"):
                fes._inject_mask(proj, "m.tif")
        fh._write_segmentation_sdata.assert_not_called()

    def test_empty_mask_returns_zero(self):
        """An all-background mask reports 0 cells."""
        proj, fh = _make_inject_project()
        mask = np.zeros((1024, 1024), dtype=np.uint32)
        with patch.object(fes.tifffile, "imread", return_value=mask):
            n = fes._inject_mask(proj, "m.tif")
        assert n == 0


class TestMaskPathBranch:
    """Tests for get_scportrait_features() with mask_path set (injection)."""

    def _run(self, mock_project, mask, mask_path="m.tif"):
        with patch.object(fes, "Project", return_value=mock_project), patch.object(
            fes.tifffile, "imread", return_value=mask
        ):
            return get_scportrait_features(
                image_paths=["bf.tif", "bf.tif"],
                channel_names=["brightfield", "brightfield_ch1"],
                config_path="config.yml",
                project_location="proj",
                segmentation_f=_DummySeg,
                extraction_f=_DummyExt,
                featurization_f=_DummyFeat,
                mask_path=mask_path,
                plots_dir=None,
            )

    def test_mask_path_skips_segmentation(self):
        """mask_path set -> segment() skipped; mask injected; extract runs."""
        mock_project = _make_mock_project()
        fh = mock_project.filehandler
        fh.cyto_seg_name = "seg_all_cytosol"
        img = MagicMock()
        img.sizes = {"y": 1023, "x": 1023}
        fh._get_input_image.return_value = img

        mask = np.zeros((1024, 1024), dtype=np.uint32)
        mask[10:20, 10:20] = 5
        result = self._run(mock_project, mask)

        mock_project.segment.assert_not_called()
        fh._write_segmentation_sdata.assert_called_once()
        assert fh._write_segmentation_sdata.call_args[0][1] == "seg_all_cytosol"
        mock_project.extract.assert_called_once()
        mock_project.featurize.assert_called_once()
        assert list(result["scportrait_cell_id"]) == [10, 20, 30]

    def test_mask_path_empty_returns_empty_and_skips_extract(self):
        """An empty injected mask returns an empty DataFrame and skips extraction."""
        mock_project = _make_mock_project()
        fh = mock_project.filehandler
        fh.cyto_seg_name = "seg_all_cytosol"
        img = MagicMock()
        img.sizes = {"y": 1023, "x": 1023}
        fh._get_input_image.return_value = img

        mask = np.zeros((1024, 1024), dtype=np.uint32)
        result = self._run(mock_project, mask)

        assert isinstance(result, pd.DataFrame)
        assert result.empty
        mock_project.segment.assert_not_called()
        mock_project.extract.assert_not_called()
        mock_project.featurize.assert_not_called()


# ---------------------------------------------------------------------------
# Milestone 2 — Phase 2 pipeline/CLI wiring (mask resolution + injection dispatch)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestResolveCellposeMask:
    """Tests for resolve_cellpose_mask() (deterministic stem pairing)."""

    def test_direct_stem_match(self, tmp_path):
        (tmp_path / "pMF5V1_E07_t1_z10_pred_mask.tif").write_bytes(b"x")
        bf = tmp_path / "pMF5V1_E07_t1_z10_BF.tif"
        got = resolve_cellpose_mask(bf, tmp_path)
        assert got.name == "pMF5V1_E07_t1_z10_pred_mask.tif"

    def test_no_match_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            resolve_cellpose_mask(tmp_path / "x_BF.tif", tmp_path)

    def test_two_matches_raises(self, tmp_path):
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        (tmp_path / "a" / "s_pred_mask.tif").write_bytes(b"x")
        (tmp_path / "b" / "s_pred_mask.tif").write_bytes(b"x")
        with pytest.raises(FileNotFoundError):
            resolve_cellpose_mask(tmp_path / "s_BF.tif", tmp_path)

    def test_direct_file_wins_over_nested_duplicate(self, tmp_path):
        """A direct child of mask_root takes precedence over a nested twin.

        The fast path returns before the rglob cardinality check, so this
        precedence is deliberate, not accidental: it avoids an rglob over a
        final_2d tree of thousands of masks for every image.
        """
        root = tmp_path / "masks"
        (root / "nested").mkdir(parents=True)
        direct = root / "s1_pred_mask.tif"
        direct.write_bytes(b"x")
        (root / "nested" / "s1_pred_mask.tif").write_bytes(b"x")

        assert resolve_cellpose_mask(tmp_path / "s1_BF.tif", root) == direct

    def test_custom_pattern(self, tmp_path):
        (tmp_path / "s_mask.tif").write_bytes(b"x")
        got = resolve_cellpose_mask(tmp_path / "s_BF.tif", tmp_path, "{stem}_mask.tif")
        assert got.name == "s_mask.tif"


@pytest.mark.unit
class TestScportraitInjectionDispatch:
    """Tests that a mask supplied for scportrait routes to injection."""

    def test_dispatch_passes_mask_and_injected_root(self, tmp_path):
        image_path = tmp_path / "pMF5V1_E07_t1_z10_BF.tif"
        mask_path = tmp_path / "pMF5V1_E07_t1_z10_pred_mask.tif"
        image_path.write_bytes(b"x")
        mask_path.write_bytes(b"x")
        pipeline = _make_pipeline(tmp_path)

        returned = pd.DataFrame(
            {"convnext_feature_0": [1.0], "scportrait_cell_id": [5]}
        )
        mock_fn = MagicMock(return_value=returned)
        with patch(
            "src.feature_extraction.feature_extraction_pipeline"
            ".get_scportrait_features",
            mock_fn,
        ):
            result = pipeline.extract_features_from_path(image_path, mask_path)

        assert result is not None
        _, kwargs = mock_fn.call_args
        assert kwargs["mask_path"] == str(mask_path)  # injection triggered
        # injected project subtree keeps native + injected runs from colliding
        assert "injected" in kwargs["project_location"]

    def test_missing_injection_mask_returns_none(self, tmp_path):
        image_path = tmp_path / "a_BF.tif"
        image_path.write_bytes(b"x")
        mask_path = tmp_path / "missing_pred_mask.tif"  # not created
        pipeline = _make_pipeline(tmp_path)

        result = pipeline.extract_features_from_path(image_path, mask_path)
        assert result is None
        assert pipeline.error_files

    def test_no_mask_stays_native(self, tmp_path):
        """Without a mask, scportrait dispatch passes mask_path=None (native)."""
        image_path = tmp_path / "a_BF.tif"
        image_path.write_bytes(b"x")
        pipeline = _make_pipeline(tmp_path)

        returned = pd.DataFrame({"f": [1.0], "scportrait_cell_id": [0]})
        mock_fn = MagicMock(return_value=returned)
        with patch(
            "src.feature_extraction.feature_extraction_pipeline"
            ".get_scportrait_features",
            mock_fn,
        ):
            pipeline.extract_features_from_path(image_path, None)

        _, kwargs = mock_fn.call_args
        assert kwargs["mask_path"] is None
        assert "injected" not in kwargs["project_location"]


@pytest.mark.unit
class TestProcessBatchScportraitInjection:
    """Tests for process_batch_scportrait() injection mode."""

    def test_injection_resolves_and_passes_mask(self, tmp_path):
        imgdir = tmp_path / "imgs"
        imgdir.mkdir()
        (imgdir / "s1_BF.tif").write_bytes(b"x")
        maskdir = tmp_path / "masks"
        maskdir.mkdir()
        (maskdir / "s1_pred_mask.tif").write_bytes(b"x")
        pipeline = _make_pipeline(tmp_path, output={"save_individual_files": False})

        with patch.object(
            pipeline,
            "extract_features_from_path",
            return_value=pd.DataFrame({"f": [1]}),
        ) as spy:
            pipeline.process_batch_scportrait(imgdir, mask_dir=maskdir)

        spy.assert_called_once()
        _, kwargs = spy.call_args
        assert Path(kwargs["mask_path"]).name == "s1_pred_mask.tif"

    def test_warns_when_injection_would_overwrite_output(self, tmp_path, caplog):
        """Injected masks are namespaced; the feature CSVs are not.

        Pointing an injected run at a native run's output dir replaces its
        all_features.csv, so the run must say so rather than overwrite quietly.
        """
        imgdir = tmp_path / "imgs"
        imgdir.mkdir()
        (imgdir / "s1_BF.tif").write_bytes(b"x")
        maskdir = tmp_path / "masks"
        maskdir.mkdir()
        (maskdir / "s1_pred_mask.tif").write_bytes(b"x")
        pipeline = _make_pipeline(tmp_path, output={"save_individual_files": False})
        (pipeline.output_dir / "all_features.csv").write_text("pre-existing\n")

        with patch.object(
            pipeline,
            "extract_features_from_path",
            return_value=pd.DataFrame({"f": [1]}),
        ):
            with caplog.at_level(logging.WARNING):
                pipeline.process_batch_scportrait(imgdir, mask_dir=maskdir)

        assert any(
            "will overwrite existing feature output" in r.message
            for r in caplog.records
        )

    def test_warns_on_existing_per_image_output_without_combined(
        self, tmp_path, caplog
    ):
        """The shipped config writes per-image CSVs and no combined file.

        Checking only all_features.csv would miss the common case entirely.
        """
        imgdir = tmp_path / "imgs"
        imgdir.mkdir()
        (imgdir / "s1_BF.tif").write_bytes(b"x")
        maskdir = tmp_path / "masks"
        maskdir.mkdir()
        (maskdir / "s1_pred_mask.tif").write_bytes(b"x")
        pipeline = _make_pipeline(tmp_path, output={"save_individual_files": False})
        native = pipeline.output_dir / "split_data"
        native.mkdir(parents=True, exist_ok=True)
        (native / "s0_BF_features.csv").write_text("pre-existing\n")

        with patch.object(
            pipeline,
            "extract_features_from_path",
            return_value=pd.DataFrame({"f": [1]}),
        ):
            with caplog.at_level(logging.WARNING):
                pipeline.process_batch_scportrait(imgdir, mask_dir=maskdir)

        assert any(
            "will overwrite existing feature output" in r.message
            for r in caplog.records
        )

    def test_glob_pattern_does_not_warn_on_native_run(self, tmp_path, caplog):
        """Native runs resolve no masks, so a glob pattern is irrelevant there."""
        imgdir = tmp_path / "imgs"
        imgdir.mkdir()
        (imgdir / "s1_BF.tif").write_bytes(b"x")
        pipeline = _make_pipeline(tmp_path, output={"save_individual_files": False})

        with patch.object(
            pipeline,
            "extract_features_from_path",
            return_value=pd.DataFrame({"f": [1]}),
        ):
            with caplog.at_level(logging.WARNING):
                pipeline.process_batch_scportrait(
                    imgdir, mask_pattern="*_pred_mask.tif"
                )

        assert not any("Ignoring mask_pattern" in r.message for r in caplog.records)

    def test_no_overwrite_warning_for_native_run(self, tmp_path, caplog):
        """A native (non-injected) run must not emit the injection warning."""
        imgdir = tmp_path / "imgs"
        imgdir.mkdir()
        (imgdir / "s1_BF.tif").write_bytes(b"x")
        pipeline = _make_pipeline(tmp_path, output={"save_individual_files": False})
        (pipeline.output_dir / "all_features.csv").write_text("pre-existing\n")

        with patch.object(
            pipeline,
            "extract_features_from_path",
            return_value=pd.DataFrame({"f": [1]}),
        ):
            with caplog.at_level(logging.WARNING):
                pipeline.process_batch_scportrait(imgdir)

        assert not any(
            "will overwrite existing feature output" in r.message
            for r in caplog.records
        )

    def test_glob_mask_pattern_falls_back_to_template(self, tmp_path):
        """A glob (the other backends' convention) must not silently no-op.

        ``"*_pred_mask.tif".format(stem=...)`` returns itself, which then
        matches every mask in the tree and fails each lookup -- an empty run
        with exit 0. The glob is ignored and the default template used.
        """
        imgdir = tmp_path / "imgs"
        imgdir.mkdir()
        (imgdir / "s1_BF.tif").write_bytes(b"x")
        maskdir = tmp_path / "masks"
        maskdir.mkdir()
        (maskdir / "s1_pred_mask.tif").write_bytes(b"x")
        (maskdir / "s2_pred_mask.tif").write_bytes(b"x")  # glob would match 2
        pipeline = _make_pipeline(tmp_path, output={"save_individual_files": False})

        with patch.object(
            pipeline,
            "extract_features_from_path",
            return_value=pd.DataFrame({"f": [1]}),
        ) as spy:
            pipeline.process_batch_scportrait(
                imgdir, mask_dir=maskdir, mask_pattern="*_pred_mask.tif"
            )

        spy.assert_called_once()
        _, kwargs = spy.call_args
        assert Path(kwargs["mask_path"]).name == "s1_pred_mask.tif"

    def test_glob_mask_pattern_warns_when_ignored(self, tmp_path, caplog):
        """Falling back to the template must be announced, not silent."""
        imgdir = tmp_path / "imgs"
        imgdir.mkdir()
        (imgdir / "s1_BF.tif").write_bytes(b"x")
        maskdir = tmp_path / "masks"
        maskdir.mkdir()
        (maskdir / "s1_pred_mask.tif").write_bytes(b"x")
        pipeline = _make_pipeline(tmp_path, output={"save_individual_files": False})

        with patch.object(
            pipeline,
            "extract_features_from_path",
            return_value=pd.DataFrame({"f": [1]}),
        ):
            with caplog.at_level(logging.WARNING):
                pipeline.process_batch_scportrait(
                    imgdir, mask_dir=maskdir, mask_pattern="*_pred_mask.tif"
                )

        assert any("Ignoring mask_pattern" in r.message for r in caplog.records)

    def test_injection_skips_image_with_no_mask(self, tmp_path):
        imgdir = tmp_path / "imgs"
        imgdir.mkdir()
        (imgdir / "s1_BF.tif").write_bytes(b"x")
        maskdir = tmp_path / "masks"
        maskdir.mkdir()  # no mask present
        pipeline = _make_pipeline(tmp_path)

        with patch.object(pipeline, "extract_features_from_path") as spy:
            out = pipeline.process_batch_scportrait(imgdir, mask_dir=maskdir)

        spy.assert_not_called()
        assert out.empty
        assert pipeline.error_files
