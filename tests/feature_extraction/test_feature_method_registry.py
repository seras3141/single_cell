"""Tests for the feature-method registry shared by pipeline, schema and CLI."""

import sys
from unittest.mock import patch

import pytest

import src.utils.config_schemas as schemas
from src.feature_extraction.feature_extraction_pipeline import FeatureExtractionPipeline
from src.utils.config_schemas import PipelineConfig, validate_pipeline_config
from src.utils.data_exclusions import DataExclusions


def _pipeline(tmp_path, **config):
    return FeatureExtractionPipeline(
        config={"output": {}, **config},
        output_dir=str(tmp_path / "out"),
        exclusions=DataExclusions.empty(),
    )


@pytest.fixture
def handcrafted_unavailable(monkeypatch):  # no method is unavailable today
    monkeypatch.setitem(schemas.UNAVAILABLE_FEATURE_METHODS, "handcrafted", "test reason")


def test_pyradiomics_is_available(tmp_path):
    assert "pyradiomics" not in schemas.UNAVAILABLE_FEATURE_METHODS
    assert _pipeline(tmp_path, method="pyradiomics").method == "pyradiomics"


def test_unavailable_method_rejected_at_construction(tmp_path, handcrafted_unavailable):
    with pytest.raises(NotImplementedError, match="not yet available: test reason"):
        _pipeline(tmp_path, method="handcrafted")


def test_method_argument_overrides_config(tmp_path, handcrafted_unavailable):
    with pytest.raises(NotImplementedError, match="not yet available"):
        FeatureExtractionPipeline(
            config={"method": "regionprops", "output": {}},
            method="handcrafted",
            output_dir=str(tmp_path / "out"),
            exclusions=DataExclusions.empty(),
        )


def test_unknown_method_raises_value_error(tmp_path):
    with pytest.raises(ValueError, match="Unsupported feature extraction method"):
        _pipeline(tmp_path, method="nope")


def test_the_older_incarta_name_is_an_alias_for_handcrafted(tmp_path):
    from tests.feature_extraction.test_run_feature_extraction_cli import (
        _load_script_module,
    )

    assert schemas.canonical_feature_method("incarta") == "handcrafted"
    assert schemas.canonical_feature_method("regionprops") == "regionprops"
    assert _pipeline(tmp_path, method="incarta").method == "handcrafted"
    cfg = PipelineConfig()
    cfg.feature_extraction.method = "incarta"
    validate_pipeline_config(cfg)
    module = _load_script_module()
    with patch.object(
        sys, "argv", ["run_feature_extraction.py", "--method", "incarta"]
    ):
        assert module.get_args().method == "handcrafted"


def test_cli_reports_unavailable_before_missing_mask_dir(handcrafted_unavailable):
    from tests.feature_extraction.test_run_feature_extraction_cli import (
        _load_script_module,
    )

    module = _load_script_module()
    config = {
        "paths": {"image_dir": "imgs"},  # no mask_dir
        "feature_extraction": {"method": "handcrafted"},
        "logging": {},
    }
    with pytest.raises(NotImplementedError, match="not yet available"):
        module.run_feature_extraction_from_config(config)


def test_method_lists_agree():
    """CLI choices and schema validation are driven by the one registry."""
    from tests.feature_extraction.test_run_feature_extraction_cli import (
        _load_script_module,
    )

    module = _load_script_module()
    for method in schemas.FEATURE_METHODS:
        with patch.object(
            sys, "argv", ["run_feature_extraction.py", "--method", method]
        ):
            assert module.get_args().method == method
        cfg = PipelineConfig()
        cfg.feature_extraction.method = method
        validate_pipeline_config(cfg)
    with patch.object(sys, "argv", ["run_feature_extraction.py", "--method", "nope"]):
        with pytest.raises(SystemExit):
            module.get_args()
