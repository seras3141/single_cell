#!/usr/bin/env python3
"""
3D Cell Tracking CLI - Command-line interface for 3D cell tracking with blur filtering.

This script provides a command-line interface for the 3D cell tracking postprocessing
pipeline, maintaining compatibility with the original track_cells.py workflow while
using the new modular architecture.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any

from src.postprocessing import (
    TrackingProcessor,
    TrackingProcessorConfig,
    TrackingConfig,
    FilterConfig,
    run_tracking_pipeline
)


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def create_config_from_args(args: argparse.Namespace) -> TrackingProcessorConfig:
    """Create configuration from command line arguments."""
    
    # Create tracking configuration
    tracking_config = TrackingConfig(
        search_range=args.search_range,
        memory=args.memory,
        min_track_length=args.min_track_length,
        min_area=args.min_area,
        max_area=args.max_area
    )
    
    # Create filter configuration
    filter_config = FilterConfig(
        patch_size=args.patch_size,
        stride_size=args.stride_size,
        blur_threshold=args.blur_threshold,
        invert_threshold=args.invert_threshold
    )
    
    # Create processor configuration
    config = TrackingProcessorConfig(
        mask_pattern=args.mask_pattern,
        blur_heatmap_suffix=args.blur_suffix,
        blur_threshold=args.blur_threshold,
        invert_blur_threshold=args.invert_threshold,
        tracking_config=tracking_config,
        filter_config=filter_config,
        create_output_dirs=True,
        overwrite_existing=args.overwrite,
        save_tracking_data=args.save_tracking_data
    )
    
    return config


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="3D Cell Tracking with Blur Filtering",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required directories
    parser.add_argument(
        "--image-directory", "-i",
        type=str,
        required=True,
        help="Directory containing source images (e.g., *_BF_3d.tif files)"
    )
    
    parser.add_argument(
        "--mask-directory", "-m", 
        type=str,
        required=True,
        help="Directory containing 3D segmentation masks (e.g., *_3d.tif files)"
    )
    
    parser.add_argument(
        "--output-directory", "-o",
        type=str,
        required=True,
        help="Directory for tracked output files"
    )
    
    parser.add_argument(
        "--blur-directory", "-b",
        type=str,
        required=True,
        help="Directory for blur heatmap cache"
    )
    
    # File patterns and naming
    parser.add_argument(
        "--mask-pattern",
        type=str,
        default="*_3d.tif",
        help="Pattern to match mask files"
    )
    
    parser.add_argument(
        "--blur-suffix",
        type=str, 
        default="_blur_heatmap_32_8.tif",
        help="Suffix for blur heatmap files"
    )
    
    # Blur filtering parameters
    parser.add_argument(
        "--blur-threshold", "-t",
        type=float,
        default=0.5,
        help="Threshold for blur filtering (lower = sharper)"
    )
    
    parser.add_argument(
        "--invert-threshold",
        action="store_true",
        help="Invert blur threshold comparison (keep cells with blur > threshold)"
    )
    
    parser.add_argument(
        "--patch-size",
        type=int,
        default=32,
        help="Patch size for blur measurement"
    )
    
    parser.add_argument(
        "--stride-size",
        type=int,
        default=8,
        help="Stride size for blur measurement"
    )
    
    # Tracking parameters
    parser.add_argument(
        "--search-range",
        type=float,
        default=5.0,
        help="Maximum distance features can move between frames"
    )
    
    parser.add_argument(
        "--memory",
        type=int,
        default=1,
        help="Number of frames to remember a particle"
    )
    
    parser.add_argument(
        "--min-track-length",
        type=int,
        default=3,
        help="Minimum length of tracks to keep"
    )
    
    # Cell filtering parameters
    parser.add_argument(
        "--min-area",
        type=int,
        default=10,
        help="Minimum cell area to consider"
    )
    
    parser.add_argument(
        "--max-area", 
        type=int,
        default=5000,
        help="Maximum cell area to consider"
    )
    
    # Output options
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files"
    )
    
    parser.add_argument(
        "--save-tracking-data",
        action="store_true",
        help="Save intermediate tracking data as CSV files"
    )
    
    # Logging options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Validate directories
    for dir_arg, dir_name in [
        (args.image_directory, "image directory"),
        (args.mask_directory, "mask directory")
    ]:
        if not Path(dir_arg).exists():
            logger.error(f"{dir_name} does not exist: {dir_arg}")
            sys.exit(1)
    
    # Create configuration
    config = create_config_from_args(args)
    
    logger.info("Starting 3D cell tracking pipeline")
    logger.info(f"Image directory: {args.image_directory}")
    logger.info(f"Mask directory: {args.mask_directory}")
    logger.info(f"Output directory: {args.output_directory}")
    logger.info(f"Blur directory: {args.blur_directory}")
    logger.info(f"Blur threshold: {args.blur_threshold}")
    logger.info(f"Invert threshold: {args.invert_threshold}")
    
    try:
        # Run tracking pipeline
        results = run_tracking_pipeline(
            mask_directory=args.mask_directory,
            image_directory=args.image_directory,
            output_directory=args.output_directory,
            blur_directory=args.blur_directory,
            config=config
        )
        
        # Print summary
        print(f"\nTracking pipeline completed successfully!")
        print(f"Total files: {results['total_files']}")
        print(f"Successful: {results['successful']}")
        print(f"Failed: {results['failed']}")
        print(f"Skipped: {results['skipped']}")
        print(f"Success rate: {results['success_rate']:.1%}")
        
        if results['failed_files']:
            print(f"\nFailed files:")
            for failed_file in results['failed_files']:
                print(f"  - {failed_file}")
        
        # Exit with appropriate code
        sys.exit(0 if results['failed'] == 0 else 1)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
