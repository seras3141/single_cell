"""
Unit tests for the scPortrait feature extractor and its pipeline integration.

scPortrait is an optional, GPU/Cellpose-heavy dependency that is not required in
CI. These tests mock scPortrait's ``Project`` class (and the pipeline-level
``get_scportrait_features`` hook) so they run without scportrait installed.
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from src.feature_extraction import feature_extractor_scportrait as fes
from src.feature_extraction.feature_extractor_scportrait import get_scportrait_features
from src.feature_extraction.feature_extraction_pipeline import FeatureExtractionPipeline


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

    def test_pipeline_missing_scportrait_records_error(self, tmp_path):
        """When the hook is None, dispatch records an error and returns None."""
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
            result = pipeline.extract_features_from_path(image_path, mask_path)

        assert result is None
        assert pipeline.error_files
        assert "scportrait" in pipeline.error_files[-1][1]


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
