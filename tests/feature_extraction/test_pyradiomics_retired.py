"""Tests for the retirement of the legacy pyradiomics prototype.

The prototype module is gone, but the ``"pyradiomics"`` method name is kept for
its in-repo replacement. Until that lands, selecting it must fail loudly at
construction, not once per file.
"""

import sys
from unittest.mock import patch

import pytest

from src.feature_extraction.feature_extraction_pipeline import FeatureExtractionPipeline
from src.utils.config_schemas import PipelineConfig, validate_pipeline_config


def test_pipeline_rejects_pyradiomics_at_construction(tmp_path):
    with pytest.raises(NotImplementedError, match="not yet available"):
        FeatureExtractionPipeline(
            config={"method": "pyradiomics", "output": {}},
            output_dir=str(tmp_path / "out"),
        )


def test_method_argument_overrides_config(tmp_path):
    with pytest.raises(NotImplementedError, match="not yet available"):
        FeatureExtractionPipeline(
            config={"method": "incarta", "output": {}},
            method="pyradiomics",
            output_dir=str(tmp_path / "out"),
        )


def test_unknown_method_still_raises_value_error(tmp_path):
    with pytest.raises(ValueError, match="Unsupported feature extraction method"):
        FeatureExtractionPipeline(
            config={"method": "nope", "output": {}},
            output_dir=str(tmp_path / "out"),
        )


def test_method_name_still_accepted_by_schema():
    cfg = PipelineConfig()
    cfg.feature_extraction.method = "pyradiomics"
    validate_pipeline_config(cfg)  # the name is reserved, not removed


def test_method_name_still_accepted_by_cli():
    from tests.feature_extraction.test_run_feature_extraction_cli import (
        _load_script_module,
    )

    module = _load_script_module()
    with patch.object(
        sys, "argv", ["run_feature_extraction.py", "--method", "pyradiomics"]
    ):
        args = module.get_args()
    assert args.method == "pyradiomics"


def test_cli_reports_unavailable_before_missing_mask_dir():
    from tests.feature_extraction.test_run_feature_extraction_cli import (
        _load_script_module,
    )

    module = _load_script_module()
    config = {
        "paths": {"image_dir": "imgs"},  # no mask_dir
        "feature_extraction": {"method": "pyradiomics"},
        "logging": {},
    }
    with pytest.raises(NotImplementedError, match="not yet available"):
        module.run_feature_extraction_from_config(config)


def test_method_lists_agree():
    """CLI choices and schema validation are driven by the one registry."""
    from src.utils.config_schemas import FEATURE_METHODS
    from tests.feature_extraction.test_run_feature_extraction_cli import (
        _load_script_module,
    )

    module = _load_script_module()
    for method in FEATURE_METHODS:
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
