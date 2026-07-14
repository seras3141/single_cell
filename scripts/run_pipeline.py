#!/usr/bin/env python3
"""
Complete single cell analysis pipeline.

This script provides a command-line interface for the complete pipeline
from raw data to analyzed results.

TODO : Update this script to use the new modules
"""

import argparse
import logging
import sys
from pathlib import Path
import os
from glob import glob
from typing import Optional

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.utils.logging_utils import setup_logging
from src.utils.config import get_config_manager
from src.dataset_analysis.run_manifest import create_or_load_manifest
from src.utils.file_utils import EXPERIMENT_WAVELENGTH_MAPPINGS


def run_data_preparation(input_dir: str, output_dir: str, config) -> None:
    """Run data preparation and train/test splitting."""    
    from run_preprocessing import run_preprocessing_from_config

    logger = logging.getLogger(__name__)

    try:
        run_preprocessing_from_config(config, input_dir, output_dir)
    except Exception as e:
        logger.error(f"Data preparation failed: {e}")
        raise


def run_2d_segmentation(split_dir: str, output_dir: str, config : dict) -> None:
    """Run 2D cell segmentation using Cellpose."""
    from run_inference import run_inference_from_config

    logger = logging.getLogger(__name__)

    try:
        logger.info("Starting 2D segmentation...")

        # Update config paths
        config["paths"]["input_dir"] = split_dir
        config["paths"]["output_dir"] = output_dir

        run_inference_from_config(config)

        logger.info("2D segmentation completed")
    except Exception as e:
        logger.error(f"2D segmentation failed: {e}")
        raise


def run_3d_segmentation(input_dir: str, output_dir: str, config) -> None:
    """Run 3D cell segmentation."""
    logger = logging.getLogger(__name__)
    logger.info("3D segmentation is not currently supported in the Cellpose-only environment.")
    
    # Create output directory but don't process anything
    os.makedirs(output_dir, exist_ok=True)
    
    # Skip actual processing
    logger.info("3D segmentation skipped (not available in Cellpose-only environment)")


def run_cell_tracking(image_dir: str, segmentation_dir: str, blur_dir : str, output_dir:str, config:dict) -> None:
    """Run cell tracking across z-stacks."""
    from src.postprocessing.tracking_processor import CellTrackingPipeline
    from run_postprocessing import run_batch_postprocessing, create_postprocessing_config
    
    logger = logging.getLogger(__name__)

    try:
        logger.info("Starting cell tracking...")
        postprocessing_config = create_postprocessing_config(config)
        pipeline = CellTrackingPipeline(postprocessing_config)

        run_batch_postprocessing(pipeline, image_dir, segmentation_dir, blur_dir, output_dir)            
        logger.info("Cell tracking completed")
    except Exception as e:
        logger.error(f"Cell tracking failed: {e}")
        raise
    


def run_feature_extraction(image_dir: str, ground_truth_dir: Optional[str] = None, inference_dir : Optional[str] = None, output_folder : Optional[str] = None, config : dict = {}) -> None:
    """Run feature extraction pipeline."""
    from run_feature_extraction import run_feature_extraction_from_config
    
    logger = logging.getLogger(__name__)
    logger.info("Starting feature extraction...")

    logger.info(f"Image directory: {image_dir}")

    if output_folder:
        config["feature_extraction"]["output"]["folder"] = output_folder

    if ground_truth_dir:
        # Use ground truth directory to extract features
        logger.info(f"Using ground truth directory: {ground_truth_dir}")
        config["paths"]["image_dir"] = image_dir
        config["paths"]["mask_dir"] = ground_truth_dir
        config["paths"]["output_dir"] = ground_truth_dir
        run_feature_extraction_from_config(config)
        
    if inference_dir:
        # Use inference directory to extract features
        logger.info(f"Using inference directory: {inference_dir}")
        config["paths"]["image_dir"] = image_dir
        config["paths"]["mask_dir"] = inference_dir
        config["paths"]["output_dir"] = inference_dir
        run_feature_extraction_from_config(config)
        
    logger.info("Feature extraction completed")

def get_3d_folder(config: dict) -> str:
    """Name of the 2D-to-3D output folder, honoring ``preprocessing.out_3d_folder``.

    Must match the value used by ``run_preprocessing.run_preprocessing_from_config``
    so the ``track`` step reads the folder that ``prepare`` actually wrote.
    """
    return config.get("preprocessing", {}).get("out_3d_folder", "3d_images")


def get_track_output_dir(output_dir: str, config: dict) -> str:
    """Root directory for cell-tracking outputs.

    Single source of truth shared by the ``track`` step (which writes ``final/``,
    ``final_2d/`` etc. under this root) and the ``extract`` step (which reads
    ``final_2d/`` from it). Layout matches ``src/dataset_analysis/processed_inventory.py``
    and the per-stage SLURM scripts: ``inference_tracked/<model_type>[/<dataset_name>]``.
    ``dataset_name`` is optional — when empty (the MF5V1 convention) no dataset-split
    subfolder is added, mirroring the inference outputs (``inference/<model_type>/``).
    """
    model_type = config["segmentation"]["cellpose"]["model_type"]
    dataset_name = config["segmentation"]["inference"]["dataset_name"]
    parts = [output_dir, "inference_tracked", model_type]
    if dataset_name:
        parts.append(dataset_name)
    return os.path.join(*parts)


def get_pipeline_legacy_args(args):
    """Get legacy CLI arguments for the pipeline."""
    legacy_overrides = {}
    
    if args.get("input_dir"):
        legacy_overrides["paths.input_dir"] = args["input_dir"]
    if args.get("output_dir"):
        legacy_overrides["paths.output_dir"] = args["output_dir"]

    if args.get("log_level"):
        legacy_overrides["logging.level"] = args["log_level"]

    if args.get("experiment_name"):
        legacy_overrides["preprocessing.wavelength_mappings"] = EXPERIMENT_WAVELENGTH_MAPPINGS[args["experiment_name"]]

    if args.get("plate"):
        legacy_overrides["preprocessing.plate_number"] = args["plate"]

    return legacy_overrides

def get_pipeline_args():
    """Parse command line arguments for the pipeline."""
    parser = argparse.ArgumentParser(description="Single cell analysis pipeline")
    parser.add_argument("--input-dir", type=str, required=False, help="Input directory containing raw microscopy data")
    parser.add_argument("--output-dir", type=str, required=False, help="Output directory for all results")
    parser.add_argument("--config", type=str, help="Path to configuration file (YAML)")
    parser.add_argument("--steps", type=str, nargs="+", choices=["prepare", "segment-2d", "track", "mcherry", "extract"], default=None, help="Pipeline steps to run")
    parser.add_argument("--experiment-name", type=str, choices=list(EXPERIMENT_WAVELENGTH_MAPPINGS.keys()), help="Experiment name — automatically sets preprocessing wavelength mappings (e.g. 'Ew2-1', 'HD1509')")
    parser.add_argument("--plate", type=str, help="Plate number for file renaming (overrides auto-detection from filepath, e.g. 'MF5V1')")
    parser.add_argument("--log-level", type=str, default=None, choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--override", type=str, action="append", help="Override config values using dot notation (e.g., model.diameter=30)")
    
    return parser.parse_args()


def parse_config_and_args():

    args = get_pipeline_args()
    cli_args = vars(args)

    # Get config manager with CLI args and legacy overrides
    config_manager = get_config_manager(cli_args, legacy_args_function=get_pipeline_legacy_args)

    # Get final config as dict for backward compatibility
    config_dict = config_manager.to_dict()
    logging.info("Final merged configuration:")
    logging.info(config_dict)

    config_dict["steps"] = cli_args.get("steps", ["prepare", "segment-2d", "track"])

    return config_dict


def main():
    config = parse_config_and_args()

    paths_config = config.get("paths", {})
    log_config = config.get("logging", {})

    input_dir = paths_config.get("input_dir", "")
    output_dir = paths_config.get("output_dir", "")

    steps = config.get("steps", ["prepare", "segment-2d", "track"])

    setup_logging(log_config=log_config)
    logger = logging.getLogger(__name__)

    logger.info("Starting single cell analysis pipeline")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Steps to run: {steps}")

    os.makedirs(output_dir, exist_ok=True)

    manifest = create_or_load_manifest(output_dir, input_dir, config)
    logger.info(f"Manifest for '{manifest.experiment_id}'")

    try:
        # Step 1: Data preparation
        if "prepare" in steps:
            manifest.start_stage("prepare")
            try:
                run_data_preparation(input_dir, output_dir, config)
                manifest.complete_stage("prepare", output_dir=output_dir)
            except Exception as e:
                manifest.fail_stage("prepare", error=str(e))
                raise

        # Step 2: 2D Segmentation
        if "segment-2d" in steps:
            split_folder = config.get("preprocessing", {}).get("split_folder", "split_data")
            split_dir = os.path.join(output_dir, split_folder)
            mask_dir = output_dir
            seg_config_snapshot = {
                "model_type": config.get("segmentation", {}).get("cellpose", {}).get("model_type"),
                "flow_threshold": config.get("segmentation", {}).get("cellpose", {}).get("flow_threshold"),
            }
            manifest.start_stage("segment-2d", config=seg_config_snapshot)
            try:
                run_2d_segmentation(split_dir, mask_dir, config)
                manifest.complete_stage("segment-2d", output_dir=mask_dir)
            except Exception as e:
                manifest.fail_stage("segment-2d", error=str(e))
                raise

        # Step 3: Cell tracking (postprocessing)
        if "track" in steps:
            results_folder = config["segmentation"]["inference"]["results_folder"]
            model_type = config["segmentation"]["cellpose"]["model_type"]
            dataset_name = config["segmentation"]["inference"]["dataset_name"]

            image_dir = os.path.join(output_dir, get_3d_folder(config))
            blur_dir = os.path.join(output_dir, "blur_heatmaps")

            mask_base_dir = os.path.join(output_dir, results_folder, model_type, dataset_name)
            mask_dir = os.path.join(mask_base_dir, "masks_3d")
            track_dir = get_track_output_dir(output_dir, config)

            manifest.start_stage("track")
            try:
                run_cell_tracking(image_dir, mask_dir, blur_dir, track_dir, config)
                manifest.complete_stage("track", output_dir=track_dir)
            except Exception as e:
                manifest.fail_stage("track", error=str(e))
                raise

        # Step 5: mCherry metrics
        if "mcherry" in steps:
            mcherry_dir = paths_config.get("mcherry_dir")
            if not mcherry_dir:
                manifest.skip_stage("mcherry", reason="paths.mcherry_dir not set in config — run run_mcherry_metrics.py directly")
                logger.info("mCherry metrics skipped: paths.mcherry_dir not configured.")
            else:
                from src.mcherry_metrics.config import ExtractionConfig
                from src.mcherry_metrics.core.batch import run_extraction
                mcherry_output = os.path.join(output_dir, "mcherry_metrics")
                manifest.start_stage("mcherry")
                try:
                    run_extraction(
                        mcherry_dir=Path(mcherry_dir),
                        output_dir=Path(mcherry_output),
                        config=ExtractionConfig(),
                    )
                    manifest.complete_stage("mcherry", output_dir=mcherry_output)
                except Exception as e:
                    manifest.fail_stage("mcherry", error=str(e))
                    raise

        # Step 6: Feature extraction
        if "extract" in steps:
            split_folder = config.get("preprocessing", {}).get("split_folder", "split_data")

            image_dir = os.path.join(output_dir, split_folder)
            ground_truth_dir = os.path.join(output_dir, split_folder)

            # Read the tracked 2D masks from the same root the `track` step wrote to.
            inference_dir = os.path.join(get_track_output_dir(output_dir, config), "final_2d")

            manifest.start_stage("extract")
            try:
                run_feature_extraction(image_dir, ground_truth_dir, inference_dir, config=config)
                manifest.complete_stage("extract", output_dir=inference_dir)
            except Exception as e:
                manifest.fail_stage("extract", error=str(e))
                raise

        logger.info("Pipeline completed successfully!")
        logger.info(manifest.summary())

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.info(manifest.summary())
        sys.exit(1)

if __name__ == "__main__":
    main()
