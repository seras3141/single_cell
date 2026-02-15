#!/usr/bin/env python3
"""
Feature Extraction Script for Single Cell Analysis

This script extracts comprehensive features from 2D indef run_feature_extraction_from_config(config: Dict[str, Any]) -> pd.DataFrame:
    pipeline = FeatureExtractionPipeline(config=config)
    features_df = pipeline.process_batch()

    return features_dfions using
the feature_extractor_2d module. It processes entire datasets and saves
features to CSV files with configurable options.

Usage:
    python scripts/run_feature_extraction.py --config config/feature_extraction_config.yaml
    python scripts/run_feature_extraction.py --image-dir data/sample_data --mask-dir data/sample_data --output-dir data/features_output
    python scripts/run_feature_extraction.py --image-dir data/sample_data --mask-dir data/sample_data/mask --mask-pattern "Cells_*.tif" --image-pattern "t1_*_w1_*.tif"
"""

import argparse
import logging
import sys
import time
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from src.feature_extraction.feature_extraction_pipeline import FeatureExtractionPipeline
from src.utils.logging_utils import setup_logging
from src.utils.config import ConfigManager

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
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
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
    
    config['logging']['level'] = args.log_level
    
    return config


def run_feature_extraction_pipeline(image_dir: Path, mask_dir: Path) -> pd.DataFrame:
    """Run the feature extraction pipeline with the given configuration.
    
    Args:
        image_dir: Directory containing images
        mask_dir: Directory containing masks
        
    Returns:
        Combined features DataFrame
    """

    pipeline = FeatureExtractionPipeline(method='incarta')

    # Find image-mask pairs
    pairs = pipeline.find_image_mask_pairs(image_dir, mask_dir)

    # Process pairs
    all_features = []
        
    for image_path, mask_path in tqdm(pairs, desc="Processing files"):
        features_df = pipeline.extract_features_from_path(image_path, mask_path)

        if features_df is not None:
            # batch_features.append(features_df)
            all_features.append(features_df)
            
            # Save individual file if configured
            if pipeline.output_config.get('save_individual_files', True):
                pipeline.save_image_features(features_df, image_path)

    combined_df = pd.concat(all_features, ignore_index=True)

    return combined_df


def run_feature_extraction_from_config(config: Dict[str, Any]) -> pd.DataFrame:
    """Run the feature extraction pipeline with the given configuration.
    
    Args:
        config: Configuration dictionary containing all pipeline settings
        
    Returns:
        Combined features DataFrame
    """
    pipeline = FeatureExtractionPipeline.from_config(config)

    paths_config = config.get('paths', {})
    image_dir = paths_config.get('image_dir', 'data/sample_data')
    mask_dir = paths_config.get('mask_dir', 'data/sample_data')
    features_df = pipeline.process_batch(
        image_dir=image_dir,
        mask_dir=mask_dir
    )

    return features_df


def main():
    """Main function."""
    args = get_args()
    config = load_config(args)
    
    # Initialize and run pipeline
    features_df = run_feature_extraction_from_config(config)

    if not features_df.empty:
        print(f"Feature extraction completed successfully!")
        print(f"Total instances: {len(features_df)}")
        print(f"Features per instance: {len([col for col in features_df.columns if col not in ['instance_id', 'image_filename', 'mask_filename', 'processing_timestamp', 'feature_extraction_version', 'dataset_name']])}")
        # print(f"Output directory: {pipeline.output_dir}")
    else:
        print("No features were extracted. Check logs for errors.")
        sys.exit(1)


if __name__ == "__main__":
    main()
