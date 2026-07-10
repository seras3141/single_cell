"""Tests for src/utils/config.py — ConfigManager and module-level helpers."""

import yaml
import pytest
from pathlib import Path

from src.utils.config import ConfigManager, save_config, load_config


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def minimal_valid_yaml(tmp_path) -> Path:
    """Write a minimal YAML that passes validate_pipeline_config."""
    config = {
        "segmentation": {
            "cellpose": {"flow_threshold": 0.4, "cellprob_threshold": 0.0},
            "inference": {"results_folder": "results", "dataset_name": "test"},
        },
        "training": {"learning_rate": 0.1, "batch_size": 8, "n_epochs": 100},
        "postprocessing": {
            "tracking": {"search_range": 5.0, "min_area": 10, "max_area": 5000},
            "filtering": {"blur_threshold": 0.5},
        },
        "feature_extraction": {"method": "incarta"},
    }
    p = tmp_path / "config.yaml"
    with open(p, "w") as f:
        yaml.dump(config, f)
    return p


# ─── ConfigManager.from_dict ──────────────────────────────────────────────────

class TestConfigManagerFromDict:
    def test_empty_dict_uses_schema_defaults(self):
        manager = ConfigManager.from_dict({})
        assert manager.get("segmentation.cellpose.model_type") == "cyto3"

    def test_override_in_dict_is_applied(self):
        manager = ConfigManager.from_dict(
            {"segmentation": {"cellpose": {"model_type": "nuclei"}}}
        )
        assert manager.get("segmentation.cellpose.model_type") == "nuclei"

    def test_get_missing_key_returns_default(self):
        manager = ConfigManager.from_dict({})
        assert manager.get("no.such.key", "fallback") == "fallback"

    def test_get_missing_key_returns_none_by_default(self):
        manager = ConfigManager.from_dict({})
        assert manager.get("no.such.key") is None

    def test_update_changes_value(self):
        manager = ConfigManager.from_dict({})
        manager.update("segmentation.cellpose.model_type", "nuclei")
        assert manager.get("segmentation.cellpose.model_type") == "nuclei"

    def test_to_dict_returns_dict(self):
        manager = ConfigManager.from_dict({})
        d = manager.to_dict()
        assert isinstance(d, dict)
        assert "segmentation" in d

    def test_to_yaml_returns_string_with_content(self):
        manager = ConfigManager.from_dict({})
        yml = manager.to_yaml()
        assert isinstance(yml, str)
        assert "segmentation" in yml

    def test_config_property_returns_dictconfig(self):
        from omegaconf import DictConfig
        manager = ConfigManager.from_dict({})
        assert isinstance(manager.config, DictConfig)


# ─── merge_with_overrides ────────────────────────────────────────────────────

class TestMergeWithOverrides:
    def test_override_applied_to_copy(self):
        manager = ConfigManager.from_dict({})
        merged = manager.merge_with_overrides(
            {"segmentation.cellpose.model_type": "nuclei"}
        )
        assert merged.get("segmentation.cellpose.model_type") == "nuclei"

    def test_original_unchanged(self):
        manager = ConfigManager.from_dict({})
        manager.merge_with_overrides(
            {"segmentation.cellpose.model_type": "nuclei"}
        )
        assert manager.get("segmentation.cellpose.model_type") == "cyto3"

    def test_multiple_overrides_applied(self):
        manager = ConfigManager.from_dict({})
        merged = manager.merge_with_overrides(
            {
                "segmentation.cellpose.model_type": "nuclei",
                "training.n_epochs": 50,
            }
        )
        assert merged.get("segmentation.cellpose.model_type") == "nuclei"
        assert merged.get("training.n_epochs") == 50


# ─── from_cli_args ────────────────────────────────────────────────────────────

class TestFromCliArgs:
    def test_without_base_config_calls_from_dict(self):
        manager = ConfigManager.from_cli_args(
            {"segmentation": {"cellpose": {"model_type": "nuclei"}}}
        )
        assert manager.get("segmentation.cellpose.model_type") == "nuclei"

    def test_with_base_config_path_applies_overrides(self, minimal_valid_yaml):
        manager = ConfigManager.from_cli_args(
            {"segmentation.cellpose.model_type": "nuclei"},
            base_config_path=str(minimal_valid_yaml),
        )
        assert manager.get("segmentation.cellpose.model_type") == "nuclei"


# ─── ConfigManager(path) ─────────────────────────────────────────────────────

class TestConfigManagerFromPath:
    def test_loads_values_from_yaml(self, minimal_valid_yaml):
        manager = ConfigManager(minimal_valid_yaml)
        assert manager.get("segmentation.cellpose.flow_threshold") == 0.4

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ConfigManager(tmp_path / "nonexistent.yaml")

    def test_save_round_trip(self, minimal_valid_yaml, tmp_path):
        manager = ConfigManager(minimal_valid_yaml)
        out = tmp_path / "saved.yaml"
        manager.save(out)
        assert out.exists()
        reloaded = ConfigManager(out)
        assert reloaded.get("segmentation.cellpose.flow_threshold") == pytest.approx(0.4)


# ─── save_config / load_config ────────────────────────────────────────────────

class TestSaveLoadConfig:
    def test_save_config_writes_file(self, tmp_path):
        data = {"key": "value", "num": 42}
        out = tmp_path / "cfg.yaml"
        save_config(data, out)
        assert out.exists()
        with open(out) as f:
            loaded = yaml.safe_load(f)
        assert loaded["key"] == "value"
        assert loaded["num"] == 42

    def test_save_config_creates_parent_dirs(self, tmp_path):
        out = tmp_path / "nested" / "dir" / "cfg.yaml"
        save_config({"x": 1}, out)
        assert out.exists()

    def test_load_config_returns_dict(self, minimal_valid_yaml):
        result = load_config(minimal_valid_yaml)
        assert isinstance(result, dict)
        assert "segmentation" in result
