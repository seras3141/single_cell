#!/usr/bin/env python3
"""
Complete single cell analysis pipeline.

This script provides a command-line interface for the complete pipeline
from raw data to analyzed results.
"""

import argparse
import logging
import sys
from pathlib import Path
import os

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def setup_logging(level: str = "INFO", log_file: str = None) -> None:
    """Setup logging configuration."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
    else:
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format=log_format
        )


def run_data_preparation(input_dir: str, output_dir: str, config) -> None:
    """Run data preparation and train/test splitting."""
    from src.utils.train_test_split import DatasetSplitter, DatasetPaths
    from src.utils.file_utils import BF_IF_FileHandler
    
    logger = logging.getLogger(__name__)
    logger.info("Starting data preparation...")
    
    # Setup paths for different experiment types
    paths = DatasetPaths(
        image_path=f"{input_dir}/**/t1_*_w1_*.tif",
        mask_path=f"{input_dir}/**/Cells_*.tif",
        output_dir=f"{output_dir}/2d_dataset"
    )
    
    file_handler = BF_IF_FileHandler()
    splitter = DatasetSplitter(
        paths=paths,
        file_handler=file_handler,
        test_size=config.get('file_processing.test_size', 0.2),
        random_state=config.get('file_processing.random_state', 42)
    )
    splitter.process()
    logger.info("Data preparation completed")


def run_2d_segmentation(test_dir: str, output_dir: str, config) -> None:
    """Run 2D cell segmentation using Cellpose."""
    logger = logging.getLogger(__name__)
    logger.info("Starting 2D segmentation...")
    
    # Import and run 2D prediction
    import subprocess
    
    cmd = [
        sys.executable, "src/models/cellpose/cellpose_2D_prediction.py",
        "--test-dir", test_dir,
        "--output-dir", output_dir,
        "--flow-threshold", str(config.get('segmentation.cellpose.flow_threshold', 0.4))
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"2D segmentation failed: {result.stderr}")
        raise RuntimeError("2D segmentation failed")
    
    logger.info("2D segmentation completed")


def run_3d_segmentation(input_dir: str, output_dir: str, config) -> None:
    """Run 3D cell segmentation."""
    logger = logging.getLogger(__name__)
    logger.info("3D segmentation is not currently supported in the Cellpose-only environment.")
    
    # Create output directory but don't process anything
    os.makedirs(output_dir, exist_ok=True)
    
    # Skip actual processing
    logger.info("3D segmentation skipped (not available in Cellpose-only environment)")


def run_cell_tracking(input_dir: str, output_dir: str, config) -> None:
    """Run cell tracking across z-stacks."""
    from src.utils.track_cells import track_3d_centers
    from glob import glob
    import tifffile
    
    logger = logging.getLogger(__name__)
    logger.info("Starting cell tracking...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all 3D segmentation files
    input_files = glob(f"{input_dir}/**/*_3d.tif", recursive=True)
    
    for input_file in input_files:
        logger.info(f"Tracking cells in {input_file}")
        
        # Load segmentation stack
        segmentation = tifffile.imread(input_file).astype(int)
        
        # Track cells
        tracked = track_3d_centers(
            segmentation,
            blur_thresh=config.get('tracking.blur_threshold', 0.5)
        )
        
        # Save results
        output_file = os.path.join(
            output_dir,
            os.path.basename(input_file).replace('.tif', '_tracked.tif')
        )
        tifffile.imwrite(output_file, tracked)
    
    logger.info("Cell tracking completed")


def run_feature_extraction(image_dir: str, mask_dir: str, output_dir: str, config) -> None:
    """Run feature extraction using PyRadiomics."""
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


def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(description="Single cell analysis pipeline")
    parser.add_argument("--input-dir", type=str, required=True,
                       help="Input directory containing raw microscopy data")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for all results")
    parser.add_argument("--config", type=str,
                       help="Path to configuration file")
    parser.add_argument("--steps", type=str, nargs="+",
                       choices=["prepare", "segment-2d", "segment-3d", "track", "extract"],
                       default=["prepare", "segment-2d", "track", "extract"],
                       help="Pipeline steps to run")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = os.path.join(args.output_dir, "logs", "pipeline.log")
    setup_logging(args.log_level, log_file)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting single cell analysis pipeline")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Steps to run: {args.steps}")
    
    # Load configuration
    try:
        from utils.confignfig import get_config, set_config
        if args.config:
            set_config(args.config)
        config = get_config()
    except Exception as e:
        logger.warning(f"Could not load configuration: {e}")
        logger.info("Using default parameters")
        config = type('Config', (), {'get': lambda self, key, default=None: default})()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Step 1: Data preparation
        if "prepare" in args.steps:
            run_data_preparation(args.input_dir, args.output_dir, config)
        
        # Step 2: 2D Segmentation
        if "segment-2d" in args.steps:
            test_dir = os.path.join(args.output_dir, "2d_dataset", "test")
            seg_output = os.path.join(args.output_dir, "segmentation", "2d")
            run_2d_segmentation(test_dir, seg_output, config)
        
        # Step 3: 3D Segmentation
        if "segment-3d" in args.steps:
            test_dir = os.path.join(args.output_dir, "2d_dataset", "test")
            seg_output = os.path.join(args.output_dir, "segmentation", "3d")
            run_3d_segmentation(test_dir, seg_output, config)
        
        # Step 4: Cell tracking
        if "track" in args.steps:
            seg_dir = os.path.join(args.output_dir, "segmentation")
            track_output = os.path.join(args.output_dir, "tracking")
            run_cell_tracking(seg_dir, track_output, config)
        
        # Step 5: Feature extraction
        if "extract" in args.steps:
            image_dir = os.path.join(args.output_dir, "2d_dataset", "test")
            mask_dir = os.path.join(args.output_dir, "segmentation")
            feature_output = os.path.join(args.output_dir, "features")
            run_feature_extraction(image_dir, mask_dir, feature_output, config)
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
