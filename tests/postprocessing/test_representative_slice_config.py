"""Tests for the representative-slice-per-cell config (Phase 1).

Covers the RepresentativeSliceConfig dataclass, its registration in PipelineConfig,
the value-range checks added to validate_pipeline_config, and loading the shipped
config/representative_slice_config.yaml through ConfigManager as a typed section.
"""

from pathlib import Path

import pytest

from src.utils.config import ConfigManager
from src.utils.config_schemas import (
    PipelineConfig,
    RepresentativeSliceConfig,
    validate_pipeline_config,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_YAML = REPO_ROOT / "config" / "representative_slice_config.yaml"


# ─── dataclass defaults ──────────────────────────────────────────────────────
class TestRepresentativeSliceConfigDefaults:
    def test_defaults(self):
        cfg = RepresentativeSliceConfig()
        # paths default empty (validated at stage runtime, not globally)
        assert cfg.input_masks_dir == ""
        assert cfg.bf_3d_dir == ""
        assert cfg.output_dir == ""
        # selection criterion
        assert cfg.sharpness_gate_fraction == 0.7
        assert cfg.selection_metric == "area"
        assert cfg.min_area == 10
        assert cfg.max_area == 100000
        assert cfg.sharpness_erosion_px == 1
        # optional pre-link blur filter (default OFF)
        assert cfg.enable_blur_filter is False
        assert cfg.blur_threshold == 0.5
        assert cfg.blur_invert_threshold is False
        # linker
        assert cfg.search_range == 5.0
        assert cfg.memory == 1
        # data convention / IO / execution
        assert cfg.z_index_offset == 1
        assert cfg.emit_all_slices is False
        assert cfg.mask_pattern == "*_pred_mask_3d.zarr"
        assert cfg.output_label_format == "tif"
        assert cfg.n_jobs is None
        assert cfg.overwrite_existing is False

    def test_registered_in_pipeline_config(self):
        cfg = PipelineConfig()
        assert isinstance(cfg.representative_slice, RepresentativeSliceConfig)


# ─── validate_pipeline_config — valid defaults ───────────────────────────────
def test_valid_default_config_does_not_raise():
    validate_pipeline_config(PipelineConfig())


# ─── validate_pipeline_config — invalid representative_slice ──────────────────
class TestValidateRepresentativeSlice:
    def test_gate_fraction_zero_raises(self):
        cfg = PipelineConfig()
        cfg.representative_slice.sharpness_gate_fraction = 0.0
        with pytest.raises(ValueError, match="sharpness_gate_fraction"):
            validate_pipeline_config(cfg)

    def test_gate_fraction_above_one_raises(self):
        cfg = PipelineConfig()
        cfg.representative_slice.sharpness_gate_fraction = 1.5
        with pytest.raises(ValueError, match="sharpness_gate_fraction"):
            validate_pipeline_config(cfg)

    def test_gate_fraction_one_is_allowed(self):
        cfg = PipelineConfig()
        cfg.representative_slice.sharpness_gate_fraction = 1.0
        validate_pipeline_config(cfg)  # (0, 1] — 1.0 is valid

    def test_min_area_not_less_than_max_area_raises(self):
        cfg = PipelineConfig()
        cfg.representative_slice.min_area = 500
        cfg.representative_slice.max_area = 500
        with pytest.raises(ValueError, match="min_area"):
            validate_pipeline_config(cfg)

    def test_non_positive_search_range_raises(self):
        cfg = PipelineConfig()
        cfg.representative_slice.search_range = 0.0
        with pytest.raises(ValueError, match="search_range"):
            validate_pipeline_config(cfg)

    def test_negative_z_index_offset_raises(self):
        cfg = PipelineConfig()
        cfg.representative_slice.z_index_offset = -1
        with pytest.raises(ValueError, match="z_index_offset"):
            validate_pipeline_config(cfg)

    def test_bad_output_label_format_raises(self):
        cfg = PipelineConfig()
        cfg.representative_slice.output_label_format = "png"
        with pytest.raises(ValueError, match="output_label_format"):
            validate_pipeline_config(cfg)

    def test_bad_selection_metric_raises(self):
        cfg = PipelineConfig()
        cfg.representative_slice.selection_metric = "perimeter"
        with pytest.raises(ValueError, match="selection_metric"):
            validate_pipeline_config(cfg)

    def test_sharpness_selection_metric_allowed(self):
        cfg = PipelineConfig()
        cfg.representative_slice.selection_metric = "sharpness"
        validate_pipeline_config(cfg)


# ─── YAML load via ConfigManager ─────────────────────────────────────────────
class TestYamlLoad:
    def test_yaml_exists(self):
        assert CONFIG_YAML.exists(), f"missing config file: {CONFIG_YAML}"

    def test_loads_as_typed_section(self):
        manager = ConfigManager(config_path=CONFIG_YAML)
        rs = manager.get_structured_config("representative_slice")
        assert isinstance(rs, RepresentativeSliceConfig)
        # Ew2-1 paths came through
        assert rs.input_masks_dir.endswith("inference/cellpose_sam/masks_3d")
        assert rs.bf_3d_dir.endswith("3d_data")
        assert rs.output_dir.endswith("inference_filtered/cellpose_sam")
        # typed scalars survive the YAML round-trip
        assert rs.sharpness_gate_fraction == 0.7
        assert rs.selection_metric == "area"
        assert rs.max_area == 100000
        assert rs.z_index_offset == 1
        assert rs.output_label_format == "tif"
        assert rs.n_jobs is None
