"""CLI-layer tests for ``scripts/run_feature_extraction.py``.

Covers the config/CLI merge only -- no pipeline, no scPortrait, no GPU.
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
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


@pytest.mark.unit
class TestScportraitDispatch:
    """The CLI must not pre-filter the mask pattern the pipeline validates."""

    def _run(self, module, tmp_path, config):
        pipeline = MagicMock()
        pipeline.output_dir = tmp_path
        pipeline.process_batch_scportrait.return_value = pd.DataFrame()
        with (
            patch.object(
                module.FeatureExtractionPipeline, "from_config", return_value=pipeline
            ),
            patch.object(module, "add_file_handler"),
            patch.object(module, "validate_inputs"),
        ):
            module.run_feature_extraction_from_config(config)
        return pipeline.process_batch_scportrait.call_args.kwargs

    def test_glob_pattern_is_forwarded_when_injecting(self, tmp_path):
        """Nulling the glob here would suppress the pipeline's warning.

        The pipeline is the single validation point; it falls back to the
        template and says so. Filtering at the call site made that silent.
        """
        module = _load_script_module()
        kwargs = self._run(
            module,
            tmp_path,
            {
                "paths": {"image_dir": str(tmp_path), "mask_dir_cli": "/masks"},
                "feature_extraction": {
                    "method": "scportrait",
                    "mask_pattern": "*_pred_mask.tif",
                },
                "output": {},
                "logging": {},
            },
        )
        assert kwargs["mask_dir"] == "/masks"
        assert kwargs["mask_pattern"] == "*_pred_mask.tif"

    def test_pattern_withheld_for_native_run(self, tmp_path):
        """Native runs resolve no masks, so the pattern must not reach them."""
        module = _load_script_module()
        kwargs = self._run(
            module,
            tmp_path,
            {
                "paths": {"image_dir": str(tmp_path), "mask_dir": "/from-config"},
                "feature_extraction": {
                    "method": "scportrait",
                    "mask_pattern": "*_pred_mask.tif",
                },
                "output": {},
                "logging": {},
            },
        )
        assert kwargs["mask_dir"] is None
        assert kwargs["mask_pattern"] is None


@pytest.mark.unit
def test_manifest_records_the_image_dir_as_input(tmp_path):
    module = _load_script_module()
    run_dir = tmp_path / "run"
    argv = [
        "run_feature_extraction.py",
        "--image-dir", "/imgs",
        "--mask-dir", "/masks",
        "--output-dir", str(tmp_path / "out"),
        "--run-dir", str(run_dir),
    ]  # fmt: skip
    extracted = pd.DataFrame({"cell_id": [1]})
    with patch.object(sys, "argv", argv), patch.object(module, "setup_logging"):
        with patch.object(module, "create_or_load_manifest") as create:
            with patch.object(
                module, "run_feature_extraction_from_config", return_value=extracted
            ):
                module.main()
    assert create.call_args.args[:2] == (str(run_dir), "/imgs")


@pytest.mark.unit
@pytest.mark.parametrize(
    "config_text",
    [None, "feature_extraction: {}\n", "paths:\n  image_dir: /configured/batch\n"],
    ids=["no-config", "schema-default-image-dir", "configured-image-dir"],
)
def test_manifest_records_the_image_folder_in_single_file_mode(tmp_path, config_text):
    module = _load_script_module()
    run_dir = tmp_path / "run"
    config_args = []
    if config_text is not None:
        config_file = tmp_path / "feature_extraction.yaml"
        config_file.write_text(config_text)
        config_args = ["--config", str(config_file)]
    argv = [
        "run_feature_extraction.py",
        *config_args,
        "--image-file", "/imgs/a_BF.tif",
        "--mask-file", "/masks/a_pred_mask.tif",
        "--output-dir", str(tmp_path / "out"),
        "--run-dir", str(run_dir),
    ]  # fmt: skip
    extracted = pd.DataFrame({"cell_id": [1]})
    with patch.object(sys, "argv", argv), patch.object(module, "setup_logging"):
        with patch.object(module, "create_or_load_manifest") as create:
            with patch.object(
                module, "run_feature_extraction_from_config", return_value=extracted
            ):
                module.main()
    assert create.call_args.args[:2] == (str(run_dir), "/imgs")


@pytest.mark.unit
@pytest.mark.parametrize("cli, expected", [([], 8), (["--n-jobs", "2"], 2)])
def test_n_jobs_overrides_the_config_only_when_given(tmp_path, cli, expected):
    module = _load_script_module()
    config_file = tmp_path / "feature_extraction.yaml"
    config_file.write_text("feature_extraction:\n  n_jobs: 8\n")
    argv = ["run_feature_extraction.py", "--config", str(config_file), *cli]
    with patch.object(sys, "argv", argv):
        args = module.get_args()
    assert module.load_config(args)["feature_extraction"]["n_jobs"] == expected


@pytest.mark.unit
def test_cli_only_run_records_the_default_n_jobs():
    module = _load_script_module()
    with patch.object(sys, "argv", ["run_feature_extraction.py", "--image-dir", "x"]):
        args = module.get_args()
    assert module.load_config(args)["feature_extraction"]["n_jobs"] == -1


@pytest.mark.unit
def test_snapshot_lists_only_the_inputs_of_the_mode_used():
    module = _load_script_module()
    paths = {"image_dir": "/batch/imgs", "mask_dir": "/batch/masks"}
    batch = module._get_extract_snapshot({"paths": paths})
    single = module._get_extract_snapshot(
        {"paths": {**paths, "image_file": "/imgs/a_BF.tif", "mask_file": "/m/a.tif"}}
    )
    assert {"image_dir", "mask_dir"} <= batch.keys()
    assert single.keys() & {"image_dir", "mask_dir"} == set()
    assert single["image_file"] == "/imgs/a_BF.tif"


@pytest.mark.unit
@pytest.mark.parametrize("mask_dir_cli, expected", [(None, None), ("/masks", "/masks")])
def test_scportrait_snapshot_records_only_injected_masks(mask_dir_cli, expected):
    module = _load_script_module()
    paths = {"image_dir": "/bf", "mask_dir": "/configured/masks"}
    if mask_dir_cli:
        paths["mask_dir_cli"] = mask_dir_cli
    snapshot = module._get_extract_snapshot(
        {"paths": paths, "feature_extraction": {"method": "scportrait"}}
    )
    assert snapshot.get("mask_dir") == expected


@pytest.mark.unit
def test_invalid_inputs_are_rejected_before_the_manifest_is_written(tmp_path):
    module = _load_script_module()
    argv = ["run_feature_extraction.py", "--run-dir", str(tmp_path / "run")]
    with patch.object(sys, "argv", argv), patch.object(module, "setup_logging"):
        with patch.object(module, "create_or_load_manifest") as create:
            with pytest.raises(SystemExit) as exited:
                module.main()
    assert exited.value.code == 1
    create.assert_not_called()


@pytest.mark.unit
def test_unavailable_method_is_reported_before_missing_inputs(
    tmp_path, monkeypatch, caplog
):
    import src.utils.config_schemas as config_schemas

    monkeypatch.setitem(
        config_schemas.UNAVAILABLE_FEATURE_METHODS, "regionprops", "test"
    )
    module = _load_script_module()
    argv = [
        "run_feature_extraction.py",
        "--method", "regionprops",
        "--image-dir", "/bf",
        "--run-dir", str(tmp_path / "run"),
    ]  # fmt: skip
    with patch.object(sys, "argv", argv), patch.object(module, "setup_logging"):
        with patch.object(module, "create_or_load_manifest") as create:
            with pytest.raises(SystemExit):
                module.main()
    assert "not yet available" in caplog.text
    create.assert_not_called()


@pytest.mark.unit
def test_n_jobs_below_minus_one_is_rejected_before_the_manifest(tmp_path, caplog):
    module = _load_script_module()
    argv = [
        "run_feature_extraction.py",
        "--image-dir", "/bf",
        "--mask-dir", "/masks",
        "--n-jobs", "-2",
        "--run-dir", str(tmp_path / "run"),
    ]  # fmt: skip
    with patch.object(sys, "argv", argv), patch.object(module, "setup_logging"):
        with patch.object(module, "create_or_load_manifest") as create:
            with pytest.raises(SystemExit) as exited:
                module.main()
    assert exited.value.code == 1
    assert "n_jobs=-2" in caplog.text
    create.assert_not_called()
