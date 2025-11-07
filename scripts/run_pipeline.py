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
    from run_feature_extraction import run_feature_extraction_pipeline
    
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
        run_feature_extraction_pipeline(config)
        
    if inference_dir:
        # Use inference directory to extract features
        logger.info(f"Using inference directory: {inference_dir}")
        config["paths"]["image_dir"] = image_dir
        config["paths"]["mask_dir"] = inference_dir
        config["paths"]["output_dir"] = inference_dir
        run_feature_extraction_pipeline(config)
        
    logger.info("Feature extraction completed")

def get_pipeline_legacy_args(args):
    """Get legacy CLI arguments for the pipeline."""
    legacy_overrides = {}
    
    if args.get("input_dir"):
        legacy_overrides["paths.input_dir"] = args["input_dir"]
    if args.get("output_dir"):
        legacy_overrides["paths.output_dir"] = args["output_dir"]

    if args.get("log_level"):
        legacy_overrides["logging.level"] = args["log_level"]
    
    return legacy_overrides

def get_pipeline_args():
    """Parse command line arguments for the pipeline."""
    parser = argparse.ArgumentParser(description="Single cell analysis pipeline")
    parser.add_argument("--input-dir", type=str, required=False, help="Input directory containing raw microscopy data")
    parser.add_argument("--output-dir", type=str, required=False, help="Output directory for all results")
    parser.add_argument("--config", type=str, help="Path to configuration file (YAML)")
    parser.add_argument("--steps", type=str, nargs="+", choices=["prepare", "segment-2d", "segment-3d", "track", "extract"], default=None, help="Pipeline steps to run")
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

    try:
        # Step 1: Data preparation
        if "prepare" in steps:
            run_data_preparation(input_dir, output_dir, config)

        # Step 2: 2D Segmentation
        if "segment-2d" in steps:
            split_folder = config.get("preprocessing", {}).get("split_folder", "split_data")
            split_dir = os.path.join(output_dir, split_folder)
            mask_dir = output_dir
            run_2d_segmentation(split_dir, mask_dir, config)

        # Step 3: 3D Segmentation
        if "segment-3d" in steps:
            # TODO : Will be implemented in a future update
            raise NotImplementedError("3D segmentation is not implemented in this Cellpose-only version of the pipeline.")
            test_dir = os.path.join(output_dir, "split", "test")
            seg_output = os.path.join(output_dir, "segmentation_3d")
            run_3d_segmentation(test_dir, seg_output, config)

        # Step 4: Cell tracking (postprocessing)
        if "track" in steps:
            # TODO : Update hardcoded paths
            results_folder = config["segmentation"]["inference"]["results_folder"]
            model_type = config["segmentation"]["cellpose"]["model_type"]
            dataset_name = config["segmentation"]["inference"]["dataset_name"]

            image_dir = os.path.join(output_dir, "3d_images")
            blur_dir = os.path.join(output_dir, "blur_heatmaps")

            mask_base_dir = os.path.join(output_dir, results_folder, model_type, dataset_name)
            mask_dir = os.path.join(mask_base_dir, "masks_3d")
            track_dir = os.path.join(mask_base_dir, "tracking")

            # Use new postprocessing pipeline
            run_cell_tracking(image_dir, mask_dir, blur_dir, track_dir, config)

        # Step 5: Feature extraction
        if "extract" in steps:
            split_folder = config.get("preprocessing", {}).get("split_folder", "split_data")

            image_dir = os.path.join(output_dir, split_folder)
            ground_truth_dir = os.path.join(output_dir, split_folder)

            results_folder = config["segmentation"]["inference"]["results_folder"]
            model_type = config["segmentation"]["cellpose"]["model_type"]
            dataset_name = config["segmentation"]["inference"]["dataset_name"]
            inference_dir = os.path.join(output_dir, results_folder, model_type, dataset_name, "tracking", "final_2d")

            run_feature_extraction(image_dir, ground_truth_dir, inference_dir, config=config)

        logger.info("Pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
