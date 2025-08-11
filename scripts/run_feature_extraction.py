#!/usr/bin/env python3
"""
Feature Extraction Script for Single Cell Analysis

This script extracts comprehensive features from 2D instance segmentations using
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
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2
from datetime import datetime
import glob
import re

from src.features.feature_extractor_2d import extract_all_instance_features
from src.utils.logging_utils import setup_logging
from src.utils.config import ConfigManager
from src.utils.file_utils import DatasetPaths

class FeatureExtractionPipeline:
    """Pipeline for extracting features from datasets of segmented cells."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize feature extraction pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Extract configuration sections
        self.paths_config = config.get('paths', {})
        self.feature_config = config.get('feature_extraction', {})

        self.pattern_config = self.feature_config.get('file_patterns', {})
        self.output_config = self.feature_config.get('output', {})
        self.processing_config = self.feature_config.get('processing', {})
        
        # Setup output directory first
        self.output_dir = Path(self.paths_config.get('output_dir', 'data/features_output'))

        if self.output_config.get('folder_name'):
            self.output_dir = self.output_dir / self.output_config['folder_name']

        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Now setup logging (which needs output_dir)
        self.setup_logging()

        
        # Initialize counters and results
        self.processed_files = 0
        self.skipped_files = 0
        self.error_files = []
        self.all_features = []
        
    def setup_logging(self):
        """Setup logging configuration."""
        from src.utils.logging_utils import setup_logging

        log_config = self.config.get('logging', {})
        log_level = log_config.get('level', 'INFO')        

        # Add file handler if configured
        log_file = self.output_dir / log_config.get('filename', 'feature_extraction.log')

        setup_logging(log_level, log_file=log_file)
        self.logger = logging.getLogger(__name__)


    def find_image_mask_pairs(self, image_dir: Path, mask_dir: Path) -> List[Tuple[Path, Path]]:
        """Find matching image and mask file pairs.
        
        Args:
            image_dir: Directory containing images
            mask_dir: Directory containing masks

        Returns:
            List of (image_path, mask_path) tuples
        """
        logger = self.logger

        pairs = []
        
        # Get file patterns
        image_patterns = self.pattern_config.get('images', ['*_BF.tif'])
        mask_patterns = self.pattern_config.get('masks', ['*_Cells.tif'])
        
        # Ensure patterns are lists
        if isinstance(image_patterns, str):
            image_patterns = [image_patterns]
        if isinstance(mask_patterns, str):
            mask_patterns = [mask_patterns]

        logger.debug(f"Searching for image patterns: {image_patterns}")
        logger.debug(f"Searching for mask patterns: {mask_patterns}")
        
        # Find all image and mask files
        image_files = []
        mask_files = []
        
        for pattern in image_patterns:
            image_files.extend(image_dir.rglob(pattern))

        for pattern in mask_patterns:
            mask_files.extend(mask_dir.rglob(pattern))

        logger.info(f"Found {len(image_files)} potential image files and {len(mask_files)} mask files")

        # Match files based on configuration
        pairs = self.match_files(image_files, mask_files)

        logger.info(f"Successfully paired {len(pairs)} image-mask pairs")
        return pairs
    
    def find_image_given_mask(self, mask_path: Path, image_files: List[Path]) -> Optional[Path]:
        """Find corresponding image file for a given mask.
        
        Args:
            mask_path: Path to the mask file
            image_files: List of available image files
            
        Returns:
            Path to the matching image file, or None if not found
        """
        logger = self.logger

        import re
        # Convert mask path to str
        mask_name = mask_path.name
        mask_patterns = self.pattern_config.get('masks', ['*_Cells.tif'])

        for mask_pattern in mask_patterns:
            mask_pattern = mask_pattern.replace('*', '(.*)')
            match = re.match(mask_pattern, mask_name)
            if match:
                mask_prefix = match.group(1)
                logger.debug(f"Mask prefix extracted: {mask_prefix}")

                for image in image_files:
                    image_stem = image.stem
                    if image_stem.startswith(mask_prefix):
                        logger.debug(f"Matched {image.name} with {mask_path.name} based on prefix {mask_prefix}")
                        return image

        logger.warning(f"No matching image found for mask: {mask_path.name}")
        return None

    def match_files(self, image_files: List[Path], mask_files: List[Path]) -> List[Tuple[Path, Path]]:
        """Match image files with corresponding mask files based on patterns.
        
        Args:
            image_files: List of image file paths
            mask_files: List of mask file paths
            
        Returns:
            List of matched (image_path, mask_path) tuples
        """
        pairs = []

        for mask in mask_files:
            # Find corresponding image for each mask
            image = self.find_image_given_mask(mask, image_files)

            if image:
                pairs.append((image, mask))
        
        return pairs

    def load_image_and_mask(self, image_path: Path, mask_path: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Load image and mask files.
        
        Args:
            image_path: Path to image file
            mask_path: Path to mask file
            
        Returns:
            Tuple of (image, mask) arrays, or (None, None) if loading fails
        """
        logger = self.logger

        try:
            # Load image
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return None, None
            
            # Load mask
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if mask is None:
                logger.error(f"Failed to load mask: {mask_path}")
                return None, None
            
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            
            # Validate dimensions match
            if image.shape != mask.shape:
                logger.error(f"Image and mask dimensions don't match: {image.shape} vs {mask.shape}")
                return None, None
            
            # Apply preprocessing if configured
            preprocessing = self.feature_config.get('preprocessing', {})
            if preprocessing.get('normalize_intensity', False):
                image = image.astype(np.float32) / 255.0
            
            clip_percentiles = preprocessing.get('clip_percentiles')
            if clip_percentiles:
                lower, upper = clip_percentiles
                p_low, p_high = np.percentile(image, [lower, upper]) # type: ignore
                image = np.clip(image, p_low, p_high)
            
            return image, mask
            
        except Exception as e:
            logger.error(f"Error loading {image_path} and {mask_path}: {str(e)}")
            return None, None
    
    '''
    def validate_mask(self, mask: np.ndarray, image_path: Path) -> bool:
        """Validate mask meets quality criteria.
        
        Args:
            mask: Mask array
            image_path: Path to corresponding image (for logging)
            
        Returns:
            True if mask is valid, False otherwise
        """
        validation_config = self.processing_config.get('validation', {})
        
        # Count instances
        unique_labels = np.unique(mask)
        n_instances = len(unique_labels) - 1 if 0 in unique_labels else len(unique_labels)
        
        min_instances = validation_config.get('min_instances_per_image', 1)
        max_instances = validation_config.get('max_instances_per_image', 10000)
        
        if n_instances < min_instances:
            logger.warning(f"Skipping {image_path.name}: too few instances ({n_instances})")
            return False
        
        if n_instances > max_instances:
            logger.warning(f"Skipping {image_path.name}: too many instances ({n_instances})")
            return False
        
        return True
    '''
    
    def extract_features_from_pair(self, image_path: Path, mask_path: Path) -> Optional[pd.DataFrame]:
        """Extract features from a single image-mask pair.
        
        Args:
            image_path: Path to image file
            mask_path: Path to mask file
            
        Returns:
            DataFrame with extracted features, or None if extraction fails
        """
        logger = self.logger

        try:
            # Load image and mask
            image, mask = self.load_image_and_mask(image_path, mask_path)
            if image is None or mask is None:
                return None
                        
            # Extract features using the main function
            n_jobs = self.feature_config.get('n_jobs', -1)
            features_df = extract_all_instance_features(mask, image, n_jobs=n_jobs)
            
            if features_df.empty:
                logger.warning(f"No features extracted from {mask_path.name}")
                return None
            
            # Add metadata if configured
            if self.output_config.get('include_metadata', True):
                features_df['image_filename'] = image_path.name
                features_df['mask_filename'] = mask_path.name
                features_df['processing_timestamp'] = datetime.now().isoformat()
                # features_df['feature_extraction_version'] = '1.0'
                features_df['dataset_name'] = image_path.parent.name
            
            logger.debug(f"Extracted {len(features_df)} instances from {image_path.name}")
            self.processed_files += 1
            
            return features_df
            
        except Exception as e:
            logger.error(f"Error extracting features from {image_path}: {str(e)}")
            self.error_files.append((str(image_path), str(e)))
            return None
    
    def save_individual_features(self, features_df: pd.DataFrame, image_path: Path):
        """Save features for individual image to CSV file.
        
        Args:
            features_df: Features DataFrame
            image_path: Original image path (for naming output file)
        """
        logger = self.logger
        if not self.output_config.get('save_individual_files', True):
            return
        
        # Create output filename
        output_format = self.output_config.get('individual_format', '{image_name}_features.csv')
        output_name = output_format.format(image_name=image_path.stem)
        
        # Create subdirectory if configured
        output_path = self.output_dir
        if self.output_config.get('create_subdirs', True):
            subdir = image_path.parent.name
            output_path = output_path / subdir
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Save file
        output_file = output_path / output_name
        features_df.to_csv(output_file, index=False)
        logger.debug(f"Saved individual features to {output_file}")

    def process_dataset(self, image_dir: Optional[Path] = None, mask_dir: Optional[Path] = None) -> pd.DataFrame:
        """Process entire dataset and extract features.
        
        Args:
            image_dir: Directory containing images (if None, uses config)
            mask_dir: Directory containing masks (if None, uses config)
            
        Returns:
            Combined DataFrame with all features
        """
        logger = self.logger

        image_dir = image_dir or Path(self.paths_config.get('image_dir', 'data/sample_data'))
        mask_dir = mask_dir or Path(self.paths_config.get('mask_dir', 'data/sample_data'))

        logger.info(f"Processing dataset: {mask_dir} with images from {image_dir}")

        # Find image-mask pairs
        pairs = self.find_image_mask_pairs(image_dir, mask_dir)
        if not pairs:
            logger.error(f"No valid image-mask pairs found in {image_dir}")
            return pd.DataFrame()
        
        # Process pairs
        all_features = []
        batch_size = self.feature_config.get('batch_size', 50)
        
        for i in tqdm(range(0, len(pairs), batch_size), desc="Processing batches"):
            batch_pairs = pairs[i:i+batch_size]
            batch_features = []
            
            for image_path, mask_path in tqdm(batch_pairs, desc="Processing files", leave=False):
                features_df = self.extract_features_from_pair(image_path, mask_path)
                
                if features_df is not None:
                    batch_features.append(features_df)
                    
                    # Save individual file if configured
                    self.save_individual_features(features_df, image_path)
            
            # Combine batch features
            if batch_features:
                batch_combined = pd.concat(batch_features, ignore_index=True)
                all_features.append(batch_combined)
                logger.info(f"Completed batch {i//batch_size + 1}: {len(batch_combined)} total instances")
        
        # Combine all features
        if all_features:
            combined_df = pd.concat(all_features, ignore_index=True)
            logger.info(f"Total features extracted: {len(combined_df)} instances from {self.processed_files} images")
        else:
            combined_df = pd.DataFrame()
            logger.warning("No features extracted from any files")
        
        return combined_df
    
    def save_combined_features(self, features_df: pd.DataFrame):
        """Save combined features to CSV file.
        
        Args:
            features_df: Combined features DataFrame
        """
        if not self.output_config.get('save_combined_file', True):
            return
        
        if features_df.empty:
            self.logger.warning("No features to save")
            return
        
        # Save combined file
        combined_filename = self.output_config.get('combined_filename', 'all_features.csv')
        output_file = self.output_dir / combined_filename
        
        features_df.to_csv(output_file, index=False)
        self.logger.info(f"Saved combined features to {output_file}")

        # Save summary statistics
        summary_file = self.output_dir / 'feature_extraction_summary.txt'
        with open(summary_file, 'w') as f:
            f.write(f"Feature Extraction Summary\n")
            f.write(f"========================\n\n")
            f.write(f"Processing completed: {datetime.now()}\n")
            f.write(f"Total files processed: {self.processed_files}\n")
            f.write(f"Files skipped: {self.skipped_files}\n")
            f.write(f"Files with errors: {len(self.error_files)}\n")
            f.write(f"Total instances: {len(features_df)}\n")
            f.write(f"Total features per instance: {len(features_df.columns)}\n\n")
            
            if self.error_files:
                f.write("Error Files:\n")
                for filepath, error in self.error_files:
                    f.write(f"  {filepath}: {error}\n")
            
            f.write(f"\nFeature Columns:\n")
            for col in features_df.columns:
                f.write(f"  {col}\n")

        self.logger.info(f"Saved processing summary to {summary_file}")

    def run(self, image_dirs: List[Path] = [], mask_dirs: List[Path] = []) -> pd.DataFrame:
        """Run the complete feature extraction pipeline.
        
        Args:
            image_dirs: List of image directories (if None, uses config)
            mask_dirs: List of mask directories (if None, uses config)

        Returns:
            Combined features DataFrame
        """
        start_time = time.time()
        self.logger.info("Starting feature extraction pipeline")

        all_datasets_features = []


        # Handle multiple image directories
        if image_dirs is None and mask_dirs is None:
            # Check if multiple dataset paths are specified in config
            image_dirs = [Path(self.paths_config.get('image_dir', 'data/sample_data'))]
            mask_dirs = [Path(self.paths_config.get('mask_dir', 'data/sample_data'))]
        else:
            assert len(image_dirs) == len(mask_dirs), "Image and mask directories must have the same length"

        # Process each directory
        for image_dir, mask_dir in zip(image_dirs, mask_dirs):
            self.logger.info(f"Processing directory: {image_dir}")
            features_df = self.process_dataset(image_dir, mask_dir)

            if not features_df.empty:
                all_datasets_features.append(features_df)
        
        # Combine all datasets
        if all_datasets_features:
            final_features = pd.concat(all_datasets_features, ignore_index=True)
        else:
            final_features = pd.DataFrame()
        
        # Save results
        self.save_combined_features(final_features)
        
        # Log completion
        elapsed_time = time.time() - start_time
        self.logger.info(f"Feature extraction completed in {elapsed_time:.2f} seconds")
        self.logger.info(f"Final results: {len(final_features)} instances from {self.processed_files} images")

        return final_features


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
            'file_patterns': {},
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
    if args.image_pattern:
        config['file_patterns']['images'] = [args.image_pattern]
    if args.mask_pattern:
        config['file_patterns']['masks'] = [args.mask_pattern]
    if args.n_jobs:
        config['feature_extraction']['n_jobs'] = args.n_jobs
    if args.batch_size:
        config['feature_extraction']['batch_size'] = args.batch_size
    
    config['logging']['level'] = args.log_level
    
    return config

def run_feature_extraction_pipeline(config: Dict[str, Any]) -> pd.DataFrame:
    """Run the feature extraction pipeline with the given configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Combined features DataFrame
    """
    print(config)  # Debugging line to check config
    pipeline = FeatureExtractionPipeline(config)
    features_df = pipeline.process_dataset()

    return features_df


def main():
    """Main function."""
    args = get_args()
    config = load_config(args)
    
    # Initialize and run pipeline
    features_df = run_feature_extraction_pipeline(config)

    if not features_df.empty:
        print(f"Feature extraction completed successfully!")
        # print(f"Extracted features from {pipeline.processed_files} images")
        print(f"Total instances: {len(features_df)}")
        print(f"Features per instance: {len([col for col in features_df.columns if col not in ['instance_id', 'image_filename', 'mask_filename', 'processing_timestamp', 'feature_extraction_version', 'dataset_name']])}")
        # print(f"Output directory: {pipeline.output_dir}")
    else:
        print("No features were extracted. Check logs for errors.")
        sys.exit(1)


if __name__ == "__main__":
    main()
