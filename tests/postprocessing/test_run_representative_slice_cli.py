"""Tests for the representative-slice CLI (scripts/run_representative_slice.py).

Covers the arg -> config building path (ConfigManager load + overrides + re-validation)
without running the pipeline. The script is loaded via importlib since scripts/ is not a
package.
"""

import importlib.util
from pathlib import Path

import pytest

from src.utils.config_schemas import RepresentativeSliceConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_representative_slice.py"
CONFIG_YAML = REPO_ROOT / "config" / "representative_slice_config.yaml"


def _load_cli():
    spec = importlib.util.spec_from_file_location(
        "run_representative_slice", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cli = _load_cli()


class TestBuildConfig:
    def test_named_and_generic_overrides_applied(self, tmp_path):
        args = cli.get_args(
            [
                "--config",
                str(CONFIG_YAML),
                "--output-dir",
                str(tmp_path),
                "--n-jobs",
                "3",
            ]
        )
        cfg = cli.build_config(args)
        assert isinstance(cfg, RepresentativeSliceConfig)
        assert cfg.output_dir == str(tmp_path)
        assert cfg.n_jobs == 3  # coerced from string to int by OmegaConf
        # a non-overridden field keeps the YAML's Ew2-1 default
        assert cfg.input_masks_dir.endswith("inference/cellpose_sam/masks_3d")
        assert cfg.sharpness_gate_fraction == 0.7

    def test_overwrite_flag_sets_override(self):
        args = cli.get_args(["--config", str(CONFIG_YAML), "--overwrite"])
        cfg = cli.build_config(args)
        assert cfg.overwrite_existing is True

    def test_bad_override_fails_revalidation(self):
        args = cli.get_args(
            [
                "--config",
                str(CONFIG_YAML),
                "-O",
                "representative_slice.sharpness_gate_fraction=2.0",
            ]
        )
        with pytest.raises(ValueError, match="sharpness_gate_fraction"):
            cli.build_config(args)
