#!/usr/bin/env python3
"""CLI for the representative-slice-per-cell stage (the representative-slice branch).

Loads ``config/representative_slice_config.yaml`` (a central config section, via
``ConfigManager``), applies CLI overrides, and runs
``src.postprocessing.representative_slice.run``.

Examples
--------
    # run the Ew2-1 defaults baked into the YAML
    uv run scripts/run_representative_slice.py

    # override paths for another experiment + parallelism
    uv run scripts/run_representative_slice.py \\
        --input-masks-dir ".../inference/cellpose_sam/masks_3d" \\
        --bf-3d-dir ".../3d_data" \\
        --output-dir ".../inference_filtered_sharpness/cellpose_sam" \\
        --n-jobs 8 --overwrite

    # arbitrary dot-notation override
    uv run scripts/run_representative_slice.py \\
        -O representative_slice.sharpness_gate_fraction=0.6
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from omegaconf import OmegaConf

# Add the project root to the path (matches scripts/run_postprocessing.py).
sys.path.append(str(Path(__file__).parent.parent))

from src.postprocessing.representative_slice import run  # noqa: E402
from src.utils.config import ConfigManager  # noqa: E402
from src.utils.config_schemas import (  # noqa: E402
    PipelineConfig,
    RepresentativeSliceConfig,
    validate_pipeline_config,
)
from src.utils.logging_utils import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

# named flags that map onto representative_slice.<field> overrides
_NAMED_OVERRIDES = {
    "input_masks_dir": "input_masks_dir",
    "bf_3d_dir": "bf_3d_dir",
    "output_dir": "output_dir",
    "mask_pattern": "mask_pattern",
    "n_jobs": "n_jobs",
    "selection_metric": "selection_metric",
}


def get_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Select one representative z-slice per cell (representative-slice branch).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("config/representative_slice_config.yaml"),
        help="YAML config with a top-level representative_slice: section.",
    )
    parser.add_argument("--input-masks-dir", help="Dir of raw 3D masks (masks_3d).")
    parser.add_argument("--bf-3d-dir", help="Dir of 3D BF stacks (3d_data).")
    parser.add_argument("--output-dir", help="Output dir (inference_<filter>_<chooser>/<model>).")
    parser.add_argument("--mask-pattern", help="Glob for input 3D mask files.")
    parser.add_argument(
        "--selection-metric",
        choices=["area", "sharpness"],
        help="Chooser among gated slices (default from config: area).",
    )
    parser.add_argument("--n-jobs", type=int, help="Parallel workers over stacks.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate outputs even if selection.csv already exists.",
    )
    parser.add_argument(
        "--override",
        "-O",
        action="append",
        default=[],
        help="Arbitrary dot-notation override, e.g. "
        "representative_slice.sharpness_gate_fraction=0.6 (repeatable).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> RepresentativeSliceConfig:
    """Load the config section and apply CLI overrides, then re-validate.

    ``ConfigManager.merge_with_overrides`` type-coerces against the schema but does NOT
    re-run ``validate_pipeline_config``; we re-validate here so bad CLI overrides fail
    fast with a clear message.
    """
    overrides: List[str] = list(args.override)
    for arg_name, field_name in _NAMED_OVERRIDES.items():
        value = getattr(args, arg_name)
        if value is not None:
            overrides.append(f"representative_slice.{field_name}={value}")
    if args.overwrite:
        overrides.append("representative_slice.overwrite_existing=true")

    manager = ConfigManager(config_path=args.config)
    if overrides:
        override_dict = OmegaConf.to_container(OmegaConf.from_dotlist(overrides))
        assert isinstance(override_dict, dict)
        manager = manager.merge_with_overrides(cast(Dict[str, Any], override_dict))

    config = manager.get_structured_config("representative_slice")
    if not isinstance(config, RepresentativeSliceConfig):
        raise TypeError(
            "Expected a representative_slice section in "
            f"{args.config}; got {type(config)!r}"
        )
    # Re-enforce the branch value-range checks against any overrides.
    validate_pipeline_config(PipelineConfig(representative_slice=config))
    return config


def main(argv: Optional[List[str]] = None) -> None:
    args = get_args(argv)
    setup_logging(level=args.log_level)
    config = build_config(args)
    logger.info(
        "representative_slice: masks=%s bf=%s out=%s",
        config.input_masks_dir,
        config.bf_3d_dir,
        config.output_dir,
    )
    run(config)


if __name__ == "__main__":
    main()
