"""CLI-layer tests for ``scripts/run_feature_extraction.py``.

Covers the config/CLI merge only -- no pipeline, no scPortrait, no GPU.
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_feature_extraction.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("_rfe_cli", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(**overrides):
    base = dict(
        config=None,
        image_dir=None,
        mask_dir=None,
        image_file=None,
        mask_file=None,
        output_dir=None,
        n_jobs=None,
        method=None,
        image_pattern=None,
        mask_pattern=None,
        log_level="INFO",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.mark.unit
class TestMaskDirIsOptIn:
    """scPortrait injection must be opt-in via --mask-dir, not via config."""

    def test_cli_mask_dir_sets_the_opt_in_key(self):
        module = _load_script_module()
        config = module.load_config(_args(mask_dir="/masks"))
        assert config["paths"]["mask_dir"] == "/masks"
        assert config["paths"]["mask_dir_cli"] == "/masks"

    def test_config_only_mask_dir_does_not_opt_in(self, tmp_path):
        """A config carrying paths.mask_dir must NOT trigger injection.

        Every shipped config sets that key, so gating on it would flip
        scPortrait from native segmentation to injection for existing runs.
        """
        module = _load_script_module()
        config = module.load_config(_args())
        config.setdefault("paths", {})["mask_dir"] = "/from-config"
        assert "mask_dir_cli" not in config["paths"]
