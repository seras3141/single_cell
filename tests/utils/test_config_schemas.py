"""Tests for src/utils/config_schemas.py — dataclass defaults and validation."""

import pytest

from src.utils.config_schemas import (
    FilterConfig,
    TrackingConfig,
    PipelineConfig,
    CellposeConfig,
    InferenceConfig,
    SegmentationConfig,
    TrainingConfig,
    PostprocessingConfig,
    validate_pipeline_config,
)


# ─── Dataclass default values ────────────────────────────────────────────────

class TestDataclassDefaults:
    def test_filter_config_defaults(self):
        cfg = FilterConfig()
        assert cfg.blur_threshold == 0.5
        assert cfg.invert_threshold is False
        assert cfg.patch_size == 32
        assert cfg.stride_size == 8

    def test_tracking_config_defaults(self):
        cfg = TrackingConfig()
        assert cfg.search_range == 5.0
        assert cfg.memory == 1
        assert cfg.min_area == 10
        assert cfg.max_area == 5000

    def test_cellpose_config_defaults(self):
        cfg = CellposeConfig()
        assert cfg.model_type == "cyto3"
        assert cfg.flow_threshold == 0.4
        assert cfg.cellprob_threshold == 0.0
        assert cfg.gpu is True

    def test_training_config_defaults(self):
        cfg = TrainingConfig()
        assert cfg.learning_rate == 0.1
        assert cfg.batch_size == 8
        assert cfg.n_epochs == 100

    def test_pipeline_config_has_all_sections(self):
        cfg = PipelineConfig()
        assert hasattr(cfg, "paths")
        assert hasattr(cfg, "segmentation")
        assert hasattr(cfg, "training")
        assert hasattr(cfg, "postprocessing")
        assert hasattr(cfg, "feature_extraction")
        assert hasattr(cfg, "logging")


# ─── validate_pipeline_config — valid ────────────────────────────────────────

def test_valid_default_config_does_not_raise():
    validate_pipeline_config(PipelineConfig())


# ─── validate_pipeline_config — invalid training ─────────────────────────────

class TestValidatePipelineConfigTraining:
    def test_zero_learning_rate_raises(self):
        cfg = PipelineConfig()
        cfg.training.learning_rate = 0.0
        with pytest.raises(ValueError, match="learning rate"):
            validate_pipeline_config(cfg)

    def test_negative_learning_rate_raises(self):
        cfg = PipelineConfig()
        cfg.training.learning_rate = -0.01
        with pytest.raises(ValueError, match="learning rate"):
            validate_pipeline_config(cfg)

    def test_zero_batch_size_raises(self):
        cfg = PipelineConfig()
        cfg.training.batch_size = 0
        with pytest.raises(ValueError, match="batch size"):
            validate_pipeline_config(cfg)

    def test_negative_batch_size_raises(self):
        cfg = PipelineConfig()
        cfg.training.batch_size = -1
        with pytest.raises(ValueError, match="batch size"):
            validate_pipeline_config(cfg)

    def test_zero_n_epochs_raises(self):
        cfg = PipelineConfig()
        cfg.training.n_epochs = 0
        with pytest.raises(ValueError, match="epoch"):
            validate_pipeline_config(cfg)


# ─── validate_pipeline_config — invalid segmentation ────────────────────────

class TestValidatePipelineConfigSegmentation:
    def test_flow_threshold_below_zero_raises(self):
        cfg = PipelineConfig()
        cfg.segmentation.cellpose.flow_threshold = -0.1
        with pytest.raises(ValueError, match="flow threshold"):
            validate_pipeline_config(cfg)

    def test_flow_threshold_above_one_raises(self):
        cfg = PipelineConfig()
        cfg.segmentation.cellpose.flow_threshold = 1.1
        with pytest.raises(ValueError, match="flow threshold"):
            validate_pipeline_config(cfg)

    def test_cellprob_threshold_below_zero_raises(self):
        cfg = PipelineConfig()
        cfg.segmentation.cellpose.cellprob_threshold = -0.5
        with pytest.raises(ValueError, match="cellprob threshold"):
            validate_pipeline_config(cfg)

    def test_cellprob_threshold_above_one_raises(self):
        cfg = PipelineConfig()
        cfg.segmentation.cellpose.cellprob_threshold = 1.5
        with pytest.raises(ValueError, match="cellprob threshold"):
            validate_pipeline_config(cfg)


# ─── validate_pipeline_config — invalid tracking ────────────────────────────

class TestValidatePipelineConfigTracking:
    def test_zero_search_range_raises(self):
        cfg = PipelineConfig()
        cfg.postprocessing.tracking.search_range = 0.0
        with pytest.raises(ValueError, match="search range"):
            validate_pipeline_config(cfg)

    def test_negative_search_range_raises(self):
        cfg = PipelineConfig()
        cfg.postprocessing.tracking.search_range = -1.0
        with pytest.raises(ValueError, match="search range"):
            validate_pipeline_config(cfg)

    def test_min_area_equals_max_area_raises(self):
        cfg = PipelineConfig()
        cfg.postprocessing.tracking.min_area = 100
        cfg.postprocessing.tracking.max_area = 100
        with pytest.raises(ValueError, match="min_area"):
            validate_pipeline_config(cfg)

    def test_min_area_greater_than_max_area_raises(self):
        cfg = PipelineConfig()
        cfg.postprocessing.tracking.min_area = 500
        cfg.postprocessing.tracking.max_area = 100
        with pytest.raises(ValueError, match="min_area"):
            validate_pipeline_config(cfg)


# ─── validate_pipeline_config — invalid filtering ───────────────────────────

def test_blur_threshold_above_one_raises():
    cfg = PipelineConfig()
    cfg.postprocessing.filtering.blur_threshold = 1.5
    with pytest.raises(ValueError, match="[Bb]lur threshold"):
        validate_pipeline_config(cfg)


# ─── validate_pipeline_config — invalid feature extraction ──────────────────

def test_invalid_feature_extraction_method_raises():
    cfg = PipelineConfig()
    cfg.feature_extraction.method = "unknown_method"
    with pytest.raises(ValueError, match="Feature extraction method"):
        validate_pipeline_config(cfg)


@pytest.mark.parametrize("method", ["incarta", "regionprops", "pyradiomics"])
def test_valid_feature_extraction_methods(method):
    cfg = PipelineConfig()
    cfg.feature_extraction.method = method
    validate_pipeline_config(cfg)  # Must not raise
