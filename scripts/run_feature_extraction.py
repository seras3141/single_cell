#!/usr/bin/env python3
"""
Feature Extraction Script for Single Cell Analysis

Extracts comprehensive features from 2D segmentation masks using the
feature_extractor_2d module. Processes entire datasets and saves
features to CSV files with configurable options.

Usage:
    python scripts/run_feature_extraction.py --config config/feature_extraction_config.yaml
    python scripts/run_feature_extraction.py --image-dir data/sample_data --mask-dir data/sample_data --output-dir data/features_output
    python scripts/run_feature_extraction.py --image-dir data/sample_data --mask-dir data/sample_data/mask --mask-pattern "Cells_*.tif" --image-pattern "t1_*_w1_*.tif"
    python scripts/run_feature_extraction.py --image-dir data/sample_data --mask-dir data/sample_data --output-dir data/features_output --method regionprops
"""

import argparse
import logging
import os
import sys
from typing import Dict, Any
import pandas as pd

from src.feature_extraction.feature_extraction_pipeline import FeatureExtractionPipeline
from src.utils.logging_utils import setup_logging
from src.utils.config import ConfigManager
from src.dataset_analysis.run_manifest import create_or_load_manifest

def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Extract features from cell segmentation datasets")
    
    # Config file option
    parser.add_argument("--config", "-c", type=str,
                       help="Path to configuration file")
    
    # Direct options (override config)
    parser.add_argument("--image-dir", "-i", type=str,
                       help="Directory containing images")
    parser.add_argument("--mask-dir", "-m", type=str,
                       help="Directory containing masks")
    parser.add_argument("--output-dir", "-o", type=str,
                       help="Output directory for feature files")
    parser.add_argument("--image-pattern", type=str,
                       help="Glob pattern for image files")
    parser.add_argument("--mask-pattern", type=str,
                       help="Glob pattern for mask files")
    parser.add_argument("--n-jobs", type=int, default=-1,
                       help="Number of parallel jobs")
    parser.add_argument("--batch-size", type=int, default=50,
                       help="Batch size for processing")
    parser.add_argument("--method", type=str, default=None,
                       choices=["incarta", "regionprops", "pyradiomics"],
                       help="Feature extraction method (overrides config). "
                            "Options: incarta, regionprops, pyradiomics")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--run-dir", type=str, default=None,
                       help="Experiment root directory where manifest.json lives. If omitted, manifest is not updated.")

    return parser.parse_args()


def load_config(args) -> Dict[str, Any]:
    """Load configuration from file or command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Configuration dictionary
    """
    logger = logging.getLogger(__name__)
    if args.config:
        # Load from config file
        config_manager = ConfigManager(args.config)
        config = config_manager.to_dict()
        logger.info(f"Loaded configuration from {args.config}")
    else:
        # Create config from command line arguments
        config = {
            'paths': {},
            'feature_extraction': {},
            'output': {},
            'processing': {},
            'logging': {}
        }
    
    # Override with command line arguments
    if args.image_dir:
        config['paths']['image_dir'] = args.image_dir
    if args.mask_dir:
        config['paths']['mask_dir'] = args.mask_dir
    if args.output_dir:
        config['paths']['output_dir'] = args.output_dir
    if args.n_jobs:
        config['feature_extraction']['n_jobs'] = args.n_jobs
    if args.method:
        config['feature_extraction']['method'] = args.method
    if args.image_pattern:
        config['feature_extraction']['image_pattern'] = args.image_pattern
    if args.mask_pattern:
        config['feature_extraction']['mask_pattern'] = args.mask_pattern

    config['logging']['level'] = args.log_level
    
    return config


def run_feature_extraction_from_config(config: Dict[str, Any]) -> pd.DataFrame:
    """Run the feature extraction pipeline with the given configuration.
    
    Args:
        config: Configuration dictionary containing all pipeline settings
        
    Returns:
        Combined features DataFrame
    """
    pipeline = FeatureExtractionPipeline.from_config(config)

    paths_config = config.get('paths', {})
    feature_config = config.get('feature_extraction', {})
    image_dir = paths_config.get('image_dir', 'data/sample_data')
    mask_dir = paths_config.get('mask_dir', 'data/sample_data')
    image_pattern = feature_config.get('image_pattern')
    mask_pattern = feature_config.get('mask_pattern')
    features_df = pipeline.process_batch(
        image_dir=image_dir,
        mask_dir=mask_dir,
        image_patterns=[image_pattern] if image_pattern else None,
        mask_patterns=[mask_pattern] if mask_pattern else None,
    )

    return features_df


def _get_extract_snapshot(config: Dict[str, Any]) -> Dict[str, Any]:
    feature_config = config.get("feature_extraction", {})
    return {k: v for k, v in {
        "method": feature_config.get("method"),
        "n_jobs": feature_config.get("n_jobs"),
        "image_dir": config.get("paths", {}).get("image_dir"),
        "mask_dir": config.get("paths", {}).get("mask_dir"),
    }.items() if v is not None}


def main():
    """Main function."""
    args = get_args()
    config = load_config(args)

    manifest = None
    if args.run_dir is not None:
        output_dir = config.get("paths", {}).get("output_dir", "")
        os.makedirs(args.run_dir, exist_ok=True)
        manifest = create_or_load_manifest(args.run_dir, output_dir, config)
        snapshot = _get_extract_snapshot(config)
        manifest.start_stage("extract", config=snapshot)

    try:
        features_df = run_feature_extraction_from_config(config)
    except Exception as e:
        logging.error(f"Feature extraction failed: {e}")
        if manifest is not None:
            manifest.fail_stage("extract", error=str(e))
            logging.info(manifest.summary())
        sys.exit(1)

    if not features_df.empty:
        print(f"Feature extraction completed successfully!")
        print(f"Total instances: {len(features_df)}")
        print(f"Features per instance: {len([col for col in features_df.columns if col not in ['instance_id', 'image_filename', 'mask_filename', 'processing_timestamp', 'feature_extraction_version', 'dataset_name']])}")
        if manifest is not None:
            output_dir = config.get("paths", {}).get("output_dir", "")
            manifest.complete_stage("extract", output_dir=output_dir)
            logging.info(manifest.summary())
    else:
        print("No features were extracted. Check logs for errors.")
        if manifest is not None:
            manifest.fail_stage("extract", error="No features extracted")
            logging.info(manifest.summary())
        sys.exit(1)


if __name__ == "__main__":
    main()
