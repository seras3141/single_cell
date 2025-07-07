#!/usr/bin/env python3
"""
Command-line interface for cell tracking postprocessing pipeline.

This script provides an easy-to-use CLI for running the complete postprocessing
pipeline on segmentation results.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from src.postprocessing.pipeline import CellTrackingPipeline, PipelineConfig
from src.postprocessing.cell_tracking import TrackingConfig
from src.postprocessing.blur_filtering import FilterConfig


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('cell_tracking_pipeline.log')
        ]
    )


def create_config_from_args(args) -> PipelineConfig:
    """Create pipeline configuration from command line arguments."""
    
    # Tracking configuration
    tracking_config = TrackingConfig(
        search_range=args.search_range,
        memory=args.memory,
        min_track_length=args.min_track_length,
        min_area=args.min_area,
        max_area=args.max_area
    )
    
    # Filter configuration
    filter_config = FilterConfig(
        patch_size=args.blur_patch_size,
        stride_size=args.blur_stride_size,
        blur_threshold=args.blur_threshold,
        invert_threshold=args.invert_blur_threshold,
        cache_blur_maps=args.cache_blur_maps
    )
    
    
    # Pipeline configuration
    pipeline_config = PipelineConfig(
        tracking_config=tracking_config,
        filter_config=filter_config,
        enable_blur_filtering=args.enable_blur_filtering,
        filter_before_tracking=args.filter_before_tracking,
        save_intermediate_results=args.save_intermediate
    )
    
    return pipeline_config


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="3D Cell Tracking and Analysis Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output arguments
    parser.add_argument(
        "--input-dir", 
        type=str, 
        help="Input directory containing segmentation files"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        required=True,
        help="Output directory for results"
    )
    parser.add_argument(
        "--segmentation-file",
        type=str,
        help="Single segmentation file to process"
    )
    parser.add_argument(
        "--image-file",
        type=str,
        help="Corresponding image file (required if using --segmentation-file)"
    )
    
    # File patterns
    parser.add_argument(
        "--seg-pattern",
        type=str,
        default="*_3d.tif",
        help="Glob pattern for segmentation files"
    )
    parser.add_argument(
        "--img-pattern",
        type=str,
        default="*_BF_3d.tif",
        help="Glob pattern for image files"
    )
    
    # Blur filtering arguments
    parser.add_argument(
        "--enable-blur-filtering",
        action="store_true",
        default=True,
        help="Enable blur-based filtering"
    )
    parser.add_argument(
        "--disable-blur-filtering",
        action="store_true",
        help="Disable blur-based filtering"
    )
    parser.add_argument(
        "--blur-threshold",
        type=float,
        default=0.5,
        help="Threshold for blur filtering"
    )
    parser.add_argument(
        "--invert-blur-threshold",
        action="store_true",
        help="Invert blur threshold (keep blurry cells)"
    )
    parser.add_argument(
        "--blur-patch-size",
        type=int,
        default=32,
        help="Patch size for blur measurement"
    )
    parser.add_argument(
        "--blur-stride-size",
        type=int,
        default=8,
        help="Stride size for blur measurement"
    )
    parser.add_argument(
        "--blur-cache-dir",
        type=str,
        help="Directory to cache blur heatmaps"
    )
    parser.add_argument(
        "--cache-blur-maps",
        action="store_true",
        default=True,
        help="Cache blur heatmaps to disk"
    )
    
    # Tracking arguments
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
        help="Minimum track length to keep"
    )
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
        
    # Pipeline options
    parser.add_argument(
        "--filter-before-tracking",
        action="store_true",
        default=True,
        help="Apply blur filtering before tracking"
    )
    parser.add_argument(
        "--track-before-filtering",
        action="store_true",
        help="Apply tracking before blur filtering"
    )
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        default=True,
        help="Save intermediate processing results"
    )
    
    # General options
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Validate arguments
    if args.segmentation_file and not args.image_file:
        parser.error("--image-file is required when using --segmentation-file")
    
    if not args.segmentation_file and not args.input_dir:
        parser.error("Either --segmentation-file or --input-dir must be specified")
    
    # Handle conflicting flags
    if args.disable_blur_filtering:
        args.enable_blur_filtering = False
        
    if args.track_before_filtering:
        args.filter_before_tracking = False
    
    # Create configuration
    config = create_config_from_args(args)
    
    # Initialize pipeline
    pipeline = CellTrackingPipeline(config)
    
    try:
        if args.segmentation_file:
            # Process single file
            logger.info(f"Processing single file: {args.segmentation_file}")
            
            result = pipeline.process_single_file(
                segmentation_path=args.segmentation_file,
                image_path=args.image_file,
                output_dir=args.output_dir,
                blur_cache_dir=args.blur_cache_dir
            )
            
            logger.info("Processing completed successfully")
            logger.info(f"Results saved to: {result.get('final_output', 'N/A')}")
            
        else:
            # Process batch
            logger.info(f"Processing batch from directory: {args.input_dir}")
            
            results = pipeline.process_batch(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                segmentation_pattern=args.seg_pattern,
                image_pattern=args.img_pattern,
                blur_cache_dir=args.blur_cache_dir
            )
            
            successful = len([r for r in results if 'error' not in r])
            logger.info(f"Batch processing completed: {successful}/{len(results)} files successful")
            
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)
    
    logger.info("Pipeline execution completed successfully")


if __name__ == "__main__":
    main()
