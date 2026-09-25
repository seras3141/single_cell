"""Tests for the feature-method registry (``FEATURE_METHODS`` and friends).

The pipeline, the schema validator and the CLI all read one registry. A method in
``UNAVAILABLE_FEATURE_METHODS`` is recognised but must fail loudly at
construction, not once per file. No method is unavailable today, so the mechanism
is exercised with a temporary entry.
"""

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
def incarta_unavailable(monkeypatch):
    monkeypatch.setitem(schemas.UNAVAILABLE_FEATURE_METHODS, "incarta", "test reason")


def test_pyradiomics_is_available(tmp_path):
    assert "pyradiomics" not in schemas.UNAVAILABLE_FEATURE_METHODS
    assert _pipeline(tmp_path, method="pyradiomics").method == "pyradiomics"


def test_unavailable_method_rejected_at_construction(tmp_path, incarta_unavailable):
    with pytest.raises(NotImplementedError, match="not yet available: test reason"):
        _pipeline(tmp_path, method="incarta")


def test_method_argument_overrides_config(tmp_path, incarta_unavailable):
    with pytest.raises(NotImplementedError, match="not yet available"):
        FeatureExtractionPipeline(
            config={"method": "regionprops", "output": {}},
            method="incarta",
            output_dir=str(tmp_path / "out"),
            exclusions=DataExclusions.empty(),
        )


def test_unknown_method_raises_value_error(tmp_path):
    with pytest.raises(ValueError, match="Unsupported feature extraction method"):
        _pipeline(tmp_path, method="nope")


def test_cli_reports_unavailable_before_missing_mask_dir(incarta_unavailable):
    from tests.feature_extraction.test_run_feature_extraction_cli import (
        _load_script_module,
    )

    module = _load_script_module()
    config = {
        "paths": {"image_dir": "imgs"},  # no mask_dir
        "feature_extraction": {"method": "incarta"},
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
