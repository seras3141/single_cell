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

from src.postprocessing.tracking_processor import CellTrackingPipeline
from src.utils.config_schemas import PostprocessingConfig, TrackingConfig, FilterConfig
from src.utils.config import ConfigManager
from src.utils.logging_utils import setup_logging

def get_tracking_config(config: dict) -> TrackingConfig:
    """Extract tracking arguments from config or CLI args."""

    if 'tracking' in config:
        tracking_kwargs = config['tracking']
    else:
        tracking_kwargs = {}

    # Remove None values
    tracking_kwargs = {k: v for k, v in tracking_kwargs.items() if v is not None}

    tracking_config = TrackingConfig(**tracking_kwargs)

    # # Separate argument groups for flexibility
    # tracking_kwargs = dict(
    #     search_range=getattr(args, 'search_range', tracking_config.get('search_range', None)),
    #     memory=getattr(args, 'memory', None),
    #     min_track_length=getattr(args, 'min_track_length', None),
    #     min_area=getattr(args, 'min_area', None),
    #     max_area=getattr(args, 'max_area', None),
    # )

    return tracking_config

def get_filtering_config(config: dict) -> FilterConfig:
    if 'filtering' in config:
        filtering_kwargs = config['filtering']
    else:
        filtering_kwargs = {}

    # Remove None values
    filtering_kwargs = {k: v for k, v in filtering_kwargs.items() if v is not None}

    filter_config = FilterConfig(**filtering_kwargs)

    # filter_kwargs = dict(
    #     patch_size=getattr(args, 'blur_patch_size', None),
    #     stride_size=getattr(args, 'blur_stride_size', None),
    #     blur_threshold=getattr(args, 'blur_threshold', None),
    #     invert_threshold=getattr(args, 'invert_blur_threshold', None),
    #     cache_blur_maps=getattr(args, 'cache_blur_maps', None),
    # )

    return filter_config

def get_postprocessing_arguments(config: dict = {}):
    if 'postprocessing' in config:
        postprocessing_kwargs = config['postprocessing']
    else:
        postprocessing_kwargs = {}

    postprocessing_kwargs = {
        "enable_blur_filtering": bool(postprocessing_kwargs.get("enable_blur_filtering", True)),
        "filter_before_tracking": bool(postprocessing_kwargs.get("filter_before_tracking", True)),
        "save_intermediate_results": bool(postprocessing_kwargs.get("save_intermediate_results", False)),
        # "mask_pattern": str(postprocessing_kwargs.get("mask_pattern", "")),
        # "image_pattern": str(postprocessing_kwargs.get("image_pattern", "")),
        # "blur_heatmap_suffix": str(postprocessing_kwargs.get("blur_heatmap_suffix", "")),
        # "output_suffix": str(postprocessing_kwargs.get("output_suffix", "")),
    }

    postprocessing_kwargs = {k: v for k, v in postprocessing_kwargs.items() if v is not None}
    return postprocessing_kwargs


def create_postprocessing_config(config : dict) -> PostprocessingConfig:
    """Create pipeline configuration from command line arguments or merged config."""
    tracking_config = get_tracking_config(config)

    filter_config = get_filtering_config(config)

    postprocessing_kwargs = get_postprocessing_arguments(config)

    postprocessing_config = PostprocessingConfig(
        tracking_config=tracking_config,
        filter_config=filter_config,
        **postprocessing_kwargs, # type: ignore
    )
    return postprocessing_config


def merge_config_and_args(config_dict, args_namespace):
    """Merge config file dictionary and argparse Namespace, CLI args take priority."""
    merged = config_dict.copy()
    for key, value in vars(args_namespace).items():
        if value is not None:
            merged[key] = value
    return argparse.Namespace(**merged)


def get_postprocessing_args():
    parser = argparse.ArgumentParser( description="3D Cell Tracking and Analysis Pipeline", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    def optional_arg(*args, **kwargs):
        kwargs['default'] = argparse.SUPPRESS  # Don't set a default so we can detect user input
        return parser.add_argument(*args, **kwargs)
    
    optional_arg("--config", type=str, help="Path to YAML configuration file")
    # Input/Output arguments
    # Batch mode
    optional_arg("-i", "--image-dir", type=str, help="Input directory containing image files")
    optional_arg("-s", "--seg-dir", type=str, help="Input directory containing segmentation files")
    optional_arg("-o", "--output-dir", type=str, help="Output directory for results")
    # Single file mode
    optional_arg("--segmentation-file", type=str, help="Single segmentation file to process")
    optional_arg("--image-file", type=str, help="Corresponding image file (required if using --segmentation-file)")
    # File patterns
    optional_arg("--img-pattern", type=str, default="*_BF_3d.tif", help="Glob pattern for image files")
    optional_arg("--seg-pattern", type=str, default="*_3d.tif", help="Glob pattern for segmentation files")
    # Blur filtering arguments
    optional_arg("--enable-blur-filtering", action="store_true", default=True, help="Enable blur-based filtering")
    optional_arg("--disable-blur-filtering", action="store_true", help="Disable blur-based filtering")
    optional_arg("--blur-threshold", type=float, default=0.5, help="Threshold for blur filtering")
    optional_arg("--invert-blur-threshold", action="store_true", help="Invert blur threshold (keep blurry cells)")
    optional_arg("--blur-patch-size", type=int, default=32, help="Patch size for blur measurement")
    optional_arg("--blur-stride-size", type=int, default=8, help="Stride size for blur measurement")
    optional_arg("--blur-dir", type=str, help="Directory to read or create blur heatmaps")
    optional_arg("--cache-blur-maps", action="store_true", default=True, help="Cache blur heatmaps to disk")
    # Tracking arguments
    optional_arg("--search-range", type=float, default=5.0, help="Maximum distance features can move between frames")
    optional_arg("--memory", type=int, default=1, help="Number of frames to remember a particle")
    optional_arg("--min-track-length", type=int, default=3, help="Minimum track length to keep")
    optional_arg("--min-area", type=int, default=10, help="Minimum cell area to consider")
    optional_arg("--max-area", type=int, default=5000, help="Maximum cell area to consider")
    # Pipeline options
    optional_arg("--filter-before-tracking", action="store_true", default=True, help="Apply blur filtering before tracking")
    optional_arg("--track-before-filtering", action="store_true", help="Apply tracking before blur filtering")
    optional_arg("--save-intermediate", action="store_true", default=True, help="Save intermediate processing results")
    # General options
    optional_arg("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    return parser.parse_args(), parser


def get_legacy_args(args: dict):
    """Extract legacy CLI arguments for backward compatibility."""
    legacy_args = {}
    # Input and Output directories
    if "image_dir" in args:
        legacy_args["paths.image_dir"] = args["image_dir"]
    if "seg_dir" in args:
        legacy_args["paths.seg_dir"] = args["seg_dir"]
    if "output_dir" in args:
        legacy_args["paths.output_dir"] = args["output_dir"]
    if "blur_dir" in args:
        legacy_args["paths.blur_dir"] = args["blur_dir"]
    # Single file processing
    if "image_file" in args:
        legacy_args["image_file"] = args["image_file"]
    if "segmentation_file" in args: 
        legacy_args["segmentation_file"] = args["segmentation_file"]
    # Blur filtering options
    if "blur_threshold" in args:
        legacy_args["filtering.blur_threshold"] = args["blur_threshold"]
    if "invert_blur_threshold" in args:
        legacy_args["filtering.invert_blur_threshold"] = args["invert_blur_threshold"]
    if "blur_patch_size" in args:
        legacy_args["filtering.blur_patch_size"] = args["blur_patch_size"]
    if "blur_stride_size" in args:
        legacy_args["filtering.blur_stride_size"] = args["blur_stride_size"]
    if "cache_blur_maps" in args:
        legacy_args["filtering.cache_blur_maps"] = args["cache_blur_maps"]
    # Tracking options
    if "search_range" in args:
        legacy_args["tracking.search_range"] = args["search_range"]
    if "memory" in args:
        legacy_args["tracking.memory"] = args["memory"]
    if "min_track_length" in args:
        legacy_args["tracking.min_track_length"] = args["min_track_length"]
    if "min_area" in args:
        legacy_args["tracking.min_area"] = args["min_area"]
    if "max_area" in args:
        legacy_args["tracking.max_area"] = args["max_area"]
    # Pipeline options
    if "enable_blur_filtering" in args:
        legacy_args["postprocessing.enable_blur_filtering"] = args["enable_blur_filtering"]
    if "filter_before_tracking" in args:
        legacy_args["postprocessing.filter_before_tracking"] = args["filter_before_tracking"]
    if "save_intermediate" in args:
        legacy_args["postprocessing.save_intermediate"] = args["save_intermediate"]
    # Logging options
    if "log_level" in args:
        legacy_args["log_level"] = args["log_level"]

    return legacy_args

def check_input(config, cli_args):
    paths_config = config.get('paths', {})
    if 'image_dir' not in paths_config or not paths_config['image_dir']:
        if 'image_file' not in cli_args:
            raise ValueError("Image directory is required. Please specify --image-dir or set 'paths.image_dir' in the config. Alternatively, you can use --image-file for single file processing.")

    if 'seg_dir' not in paths_config or not paths_config['seg_dir']:
        if 'segmentation_file' not in cli_args:
            raise ValueError("Segmentation directory is required. Please specify --seg-dir or set 'paths.seg_dir' in the config. Alternatively, you can use --segmentation-file for single file processing.")
        
    if 'segmentation_file' in cli_args and 'image_file' not in cli_args:
        raise ValueError("When using --segmentation-file, you must also specify --image-file to provide the corresponding image.")
    
    if 'seg_dir' in paths_config and 'image_dir' not in paths_config:
        raise ValueError("When using a segmentation directory, you must also specify an image directory. Please set 'paths.image_dir' in the config or use --image-dir.")
    
    if 'segmentation_file' in cli_args and 'image_file' in cli_args:
        config['batch_mode'] = False
    elif 'seg_dir' in paths_config and 'image_dir' in paths_config:
        config['batch_mode'] = True
    else:
        raise ValueError("Either --segmentation-file or --seg-dir must be specified for batch processing. If using single file mode, both --segmentation-file and --image-file are required.")
    
    return config
    
    

def validate_and_prepare_args(config : dict, args: dict):

    config = check_input(config, args)
    
    return config

def validate_and_prepare_args_legacy(args, parser):
    """Validate and prepare command line arguments."""
    if not "segmentation_file" in args and not "seg_dir" in args:
        parser.error("Either --segmentation-file or --seg-dir must be specified")
    # If using single file mode, image file is required
    if "segmentation_file" in args and not "image_file" in args:
        parser.error("--image-file is required when using --segmentation-file")
    # If using batch mode, both directories are required
    if "seg_dir" in args and not "image_dir" in args:
        parser.error("--image-dir is required when using --seg-dir")

    # Handle boolean flags
    if args.get("disable_blur_filtering"):
        args["enable_blur_filtering"] = False
    if args.get("track_before_filtering"):
        args["filter_before_tracking"] = False

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
    cli_args = vars(args)

    # Set up logging
    setup_logging(cli_args.get("log_level", "INFO"))
    logger = logging.getLogger(__name__)

    # 1: Load base config (from YAML or default)
    if "config" in cli_args:
        config_manager = ConfigManager(cli_args["config"])
        logging.info(f"Loaded configuration from {args.config}")
    else:
        config_manager = ConfigManager()  # Use defaults
        logging.info("Using default configuration")

    # 2: Apply dotlist overrides from CLI
    from omegaconf import OmegaConf
    if "override" in cli_args:
        overrides = OmegaConf.from_dotlist(cli_args["override"])
        override_dict = OmegaConf.to_container(overrides)
        logging.info(f"Applying CLI overrides: {cli_args['override']}")
        config_manager = config_manager.merge_with_overrides(override_dict) #type: ignore
        # config_dict.update(override_dict)

    # 3: Apply legacy overrides and Merge config and CLI args        
    legacy_overrides = get_legacy_args(cli_args)
    if legacy_overrides:
        config_manager = config_manager.merge_with_overrides(legacy_overrides)
        logging.info(f"Applied legacy CLI overrides: {list(legacy_overrides.keys())}")

    # Get final config as dict for backward compatibility
    config_dict = config_manager.to_dict()
    config_dict = validate_and_prepare_args(config_dict, cli_args)

    postprocessing_config = create_postprocessing_config(config_dict)

    pipeline = CellTrackingPipeline(postprocessing_config)

    try:
        if config_dict["batch_mode"]:
            paths_config = config_dict.get("paths", {})
            # Use paths from config or CLI args
            image_dir = Path(paths_config.get("image_dir", "."))
            seg_dir = Path(paths_config.get("seg_dir", "."))
            blur_dir = Path(paths_config.get("blur_dir", "."))
            output_dir = Path(paths_config.get("output_dir", "."))
            # TODO : Add checks for directories
            run_batch_postprocessing(pipeline, image_dir, seg_dir, blur_dir, output_dir)
        else:
            run_single_file_postprocessing(pipeline, config_dict)
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)
    logger.info("Pipeline execution completed successfully")


if __name__ == "__main__":
    main()
