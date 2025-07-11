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
import tifffile

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.utils.logging_utils import setup_logging


def run_data_preparation(input_dir: str, output_dir: str, config) -> None:
    """Run data preparation and train/test splitting."""
    logger = logging.getLogger(__name__)
    logger.info("Starting data preparation...")
    
    from src.utils.file_utils import BF_IF_FileHandler
    from src.utils.conversion import combine_2d_to_3d
    from src.preprocessing.dataset_split import train_test_split_directory
    from src.preprocessing.blur_analysis import measure_dataset_blur_heatmaps

    # Step 1: Split dataset
    split_dir = os.path.join(output_dir, "split")
    logger.info(f"Splitting dataset into train/test at {split_dir} ...")
    train_test_split_directory(
        data_dir=input_dir,
        output_dir=split_dir,
        test_size=config.get('file_processing.test_size', 0.2),
        random_state=config.get('file_processing.random_state', 42),
        image_pattern=config.get('file_processing.raw_data_patterns.brightfield', "t1_*_w1_*.tif"),
        mask_pattern=config.get('file_processing.raw_data_patterns.mask', "Cells_*.tif"),
        file_handler=BF_IF_FileHandler()
    )

    # Step 2: Combine 2D to 3D (for train set only)
    input_2d_dir = split_dir
    output_3d_dir = os.path.join(output_dir, "split_3d")
    logger.info(f"Combining 2D images into 3D stacks at {output_3d_dir} ...")
    combine_2d_to_3d(
        input_dir=input_2d_dir,
        output_dir=output_3d_dir,
        # pattern=args.combine_pattern,
        recursive=True,
    )

    # Step 3: Generate blur heatmaps
    blur_dir = os.path.join(output_dir, "blur_heatmaps")
    logger.info(f"Generating blur heatmaps at {blur_dir} ...")
    measure_dataset_blur_heatmaps(
        input_dir=output_3d_dir,
        output_dir=blur_dir,
        pattern="*_BF_3d.tif",
        patch_size=config.get('quality.blur_detection.patch_size', 32),
        stride_size=config.get('quality.blur_detection.stride_size', 8),
        normalize=True,
        # overwrite=args.overwrite
    )
    logger.info("Preprocessing complete.")


def run_2d_segmentation(test_dir: str, output_dir: str, config) -> None:
    """Run 2D cell segmentation using Cellpose."""
    logger = logging.getLogger(__name__)
    logger.info("Starting 2D segmentation...")
    
    # Use direct import from the same directory for run_inference
    from run_inference import run_inference

    # Concatenate dataset_name to input_dir
    input_dir = test_dir
    
    try:
        run_inference(
            input_dir=input_dir,
            output_dir=output_dir,
            dataset_name="test",
            file_pattern="*_BF.tif",
            config=config,
        )

        logger.info("2D segmentation completed")

    except Exception as e:
        logger.error(f"Inference failed: {e}")


def run_3d_segmentation(input_dir: str, output_dir: str, config) -> None:
    """Run 3D cell segmentation."""
    logger = logging.getLogger(__name__)
    logger.info("3D segmentation is not currently supported in the Cellpose-only environment.")
    
    # Create output directory but don't process anything
    os.makedirs(output_dir, exist_ok=True)
    
    # Skip actual processing
    logger.info("3D segmentation skipped (not available in Cellpose-only environment)")


def run_cell_tracking(image_dir: str, segmentation_dir: str, blur_dir : str, output_dir:str, config) -> None:
    """Run cell tracking across z-stacks."""
    from src.postprocessing.tracking_processor import CellTrackingPipeline
    from run_postprocessing import run_batch_postprocessing, create_postprocessing_config_from_args
    
    logger = logging.getLogger(__name__)
    logger.info("Starting cell tracking...")

    postprocessing_config = create_postprocessing_config_from_args(config)

    print(image_dir, segmentation_dir, blur_dir, output_dir) # Debug print

    pipeline = CellTrackingPipeline(postprocessing_config)
    try:
        run_batch_postprocessing(pipeline, image_dir, segmentation_dir, blur_dir, output_dir)
            
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
    
    logger.info("Cell tracking completed")



def run_feature_extraction(image_dir: str, mask_dir: str, output_dir: str, config) -> None:
    """Run feature extraction using PyRadiomics."""
    raise NotImplementedError("Feature extraction is not implemented in this script. Will be included in a future update.")
    from src.utils.feature_extractor import extract_features
    from glob import glob
    import pandas as pd
    
    logger = logging.getLogger(__name__)
    logger.info("Starting feature extraction...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find matching image-mask pairs
    image_files = glob(f"{image_dir}/**/*.tif", recursive=True)
    
    all_features = []
    
    for image_file in image_files:
        # Find corresponding mask file
        mask_file = image_file.replace(image_dir, mask_dir).replace('_BF.tif', '_Cells.tif')
        
        if os.path.exists(mask_file):
            logger.info(f"Extracting features from {os.path.basename(image_file)}")
            
            try:
                features = extract_features(image_file, mask_file)
                features['image_file'] = os.path.basename(image_file)
                all_features.append(features)
            except Exception as e:
                logger.warning(f"Failed to extract features from {image_file}: {e}")
    
    # Save combined features
    if all_features:
        df = pd.concat(all_features, ignore_index=True)
        output_file = os.path.join(output_dir, "extracted_features.csv")
        df.to_csv(output_file, index=False)
        logger.info(f"Features saved to {output_file}")
    
    logger.info("Feature extraction completed")


def parse_config_and_args():
    import yaml
    parser = argparse.ArgumentParser(description="Single cell analysis pipeline")
    parser.add_argument("--input-dir", type=str, required=False, help="Input directory containing raw microscopy data")
    parser.add_argument("--output-dir", type=str, required=False, help="Output directory for all results")
    parser.add_argument("--config", type=str, help="Path to configuration file (YAML)")
    parser.add_argument("--steps", type=str, nargs="+", choices=["prepare", "segment-2d", "segment-3d", "track", "extract"], default=None, help="Pipeline steps to run")
    parser.add_argument("--log-level", type=str, default=None, choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    # Add more CLI args as needed for pipeline parameters
    args = parser.parse_args()

    config = {}
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    # CLI args override config file
    for key in ["input_dir", "output_dir", "steps", "log_level"]:
        arg_val = getattr(args, key.replace('-', '_'), None)
        if arg_val is not None:
            config[key] = arg_val
    # Set defaults if not present
    config.setdefault("steps", ["prepare", "segment-2d", "track"])
    return config


def main():
    config = parse_config_and_args()
    input_dir = config["input_dir"]
    output_dir = config["output_dir"]
    steps = config["steps"]

    log_level = config.get("log_level", "INFO")
    log_file = os.path.join(output_dir, "logs", "pipeline.log")
    setup_logging(log_level, log_file)
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
            test_dir = os.path.join(output_dir, "split", "test")
            seg_dir = os.path.join(output_dir, "segmentation_2d")
            run_2d_segmentation(test_dir, seg_dir, config)

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
            image_dir = os.path.join(output_dir, "split_3d")
            blur_dir = os.path.join(output_dir, "blur_heatmaps")
            pred_dir = os.path.join(output_dir, "segmentation_2d", "cyto3", "test")
            seg_dir = os.path.join(pred_dir, "masks_3d")
            track_output = os.path.join(pred_dir, "tracking")
            # Use new postprocessing pipeline
            run_cell_tracking(image_dir, seg_dir, blur_dir, track_output, config)

        # Step 5: Feature extraction
        if "extract" in steps:
            image_dir = os.path.join(output_dir, "2d_dataset", "test")
            mask_dir = os.path.join(output_dir, "segmentation")
            feature_output = os.path.join(output_dir, "features")
            run_feature_extraction(image_dir, mask_dir, feature_output, config)

        logger.info("Pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
