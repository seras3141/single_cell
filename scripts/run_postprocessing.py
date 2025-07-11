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
import yaml  # For YAML config file support

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from src.postprocessing.tracking_processor import CellTrackingPipeline, PostprocessingConfig
from src.postprocessing.cell_tracking import TrackingConfig
from src.postprocessing.blur_filtering import FilterConfig
from src.utils.logging_utils import setup_logging

# TODO : Add typing to all files

def get_tracking_arguments(args: Optional[dict] = None):
    if 'tracking' in args:
        return args['tracking']
    
    # Separate argument groups for flexibility
    tracking_kwargs = dict(
        search_range=getattr(args, 'search_range', None),
        memory=getattr(args, 'memory', None),
        min_track_length=getattr(args, 'min_track_length', None),
        min_area=getattr(args, 'min_area', None),
        max_area=getattr(args, 'max_area', None),
    )
    # Remove None values
    tracking_kwargs = {k: v for k, v in tracking_kwargs.items() if v is not None}

    return tracking_kwargs

def get_filtering_arguments(args: Optional[dict] = None):
    if 'filtering' in args:
        return args['filtering']

    filter_kwargs = dict(
        patch_size=getattr(args, 'blur_patch_size', None),
        stride_size=getattr(args, 'blur_stride_size', None),
        blur_threshold=getattr(args, 'blur_threshold', None),
        invert_threshold=getattr(args, 'invert_blur_threshold', None),
        cache_blur_maps=getattr(args, 'cache_blur_maps', None),
    )
    filter_kwargs = {k: v for k, v in filter_kwargs.items() if v is not None}

    return filter_kwargs

def get_postprocessing_arguments(args: Optional[dict] = None):
    if 'postprocessing' in args:
        return args['postprocessing']
    postprocessing_kwargs = dict(
        enable_blur_filtering=getattr(args, 'enable_blur_filtering', True),
        filter_before_tracking=getattr(args, 'filter_before_tracking', True),
        save_intermediate_results=getattr(args, 'save_intermediate', False),
    )
    postprocessing_kwargs = {k: v for k, v in postprocessing_kwargs.items() if v is not None}
    return postprocessing_kwargs

def create_postprocessing_config_from_args(args) -> PostprocessingConfig:
    """Create pipeline configuration from command line arguments or merged config."""
    tracking_kwargs = get_tracking_arguments(args)
    tracking_config = TrackingConfig(**tracking_kwargs)

    filtering_kwargs = get_filtering_arguments(args)
    filter_config = FilterConfig(**filtering_kwargs)

    postprocessing_kwargs = get_postprocessing_arguments(args)
    postprocessing_config = PostprocessingConfig(
        tracking_config=tracking_config,
        filter_config=filter_config,
        **postprocessing_kwargs
    )
    return postprocessing_config

def load_config_file(config_path):
    """Load configuration from a YAML file if provided."""
    if not config_path:
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}

def merge_config_and_args(config_dict, args_namespace):
    """Merge config file dictionary and argparse Namespace, CLI args take priority."""
    merged = config_dict.copy()
    for key, value in vars(args_namespace).items():
        if value is not None:
            merged[key] = value
    return argparse.Namespace(**merged)


def get_postprocessing_args():
    parser = argparse.ArgumentParser(
        description="3D Cell Tracking and Analysis Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML configuration file"
    )
    # Input/Output arguments
    # Batch mode
    parser.add_argument(
        "-i", "--image-dir", 
        type=str, 
        help="Input directory containing image files"
    )
    parser.add_argument(
        "-s", "--seg-dir", 
        type=str, 
        help="Input directory containing segmentation files"
    )
    parser.add_argument(
        "-o", "--output-dir", 
        type=str, 
        required=True,
        help="Output directory for results"
    )
    # Single file mode
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
        "--blur-dir",
        type=str,
        help="Directory to read or create blur heatmaps"
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
    
    return parser.parse_args(), parser

def validate_and_prepare_args(args, parser):
    """Validate and prepare command line arguments."""
    if args.segmentation_file and not args.image_file:
        parser.error("--image-file is required when using --segmentation-file")
    if not args.segmentation_file and not args.seg_dir:
        parser.error("Either --segmentation-file or --seg-dir must be specified")
    if args.disable_blur_filtering:
        args.enable_blur_filtering = False
    if args.track_before_filtering:
        args.filter_before_tracking = False
    return args

def run_single_file_postprocessing(pipeline, args):
    """Run postprocessing pipeline on a single file."""
    logger = logging.getLogger(__name__)
    logger.info(f"Processing single file: {args.segmentation_file}")
    result = pipeline.process_single_file(
        segmentation_path=args.segmentation_file,
        image_path=args.image_file,
        output_dir=args.output_dir,
        blur_cache_dir=args.blur_cache_dir
    )
    logger.info("Processing completed successfully")
    logger.info(f"Results saved to: {result.get('final_output', 'N/A')}")

def run_batch_postprocessing(pipeline, image_dir, seg_dir, blur_dir, output_dir):
    """Run postprocessing pipeline on a batch of files."""
    logger = logging.getLogger(__name__)
    logger.info(f"Processing batch from directory: {seg_dir}")
    results = pipeline.process_batch(
        image_dir=image_dir,
        mask_dir=seg_dir,
        output_dir=output_dir,
        blur_cache_dir=blur_dir,
    )
    successful = len([r for r in results if 'error' not in r])
    logger.info(f"Batch processing completed: {successful}/{len(results)} files successful")

def main():
    """Main CLI function."""
    args, parser = get_postprocessing_args()

    # Load config file if provided and merge with CLI args
    config_dict = load_config_file(getattr(args, "config", None))
    merged_args = merge_config_and_args(config_dict, args)

    setup_logging(merged_args.log_level, 'cell_tracking_pipeline.log')
    logger = logging.getLogger(__name__)

    merged_args = validate_and_prepare_args(merged_args, parser)
    postprocessing_config = create_postprocessing_config_from_args(merged_args)

    pipeline = CellTrackingPipeline(postprocessing_config)
    try:
        if merged_args.segmentation_file:
            run_single_file_postprocessing(pipeline, merged_args)
        else:
            run_batch_postprocessing(pipeline, merged_args.image_dir, merged_args.seg_dir, merged_args.blur_dir, merged_args.output_dir)
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)
    logger.info("Pipeline execution completed successfully")


if __name__ == "__main__":
    main()
