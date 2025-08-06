import argparse
import logging
from pathlib import Path
import os, sys
from typing import Any, Dict, Optional, Union
from src.preprocessing.dataset_split import train_test_split_directory
from src.utils.config import ConfigManager
from src.utils.file_utils import BF_IF_FileHandler
from src.utils.conversion import combine_2d_to_3d
from src.preprocessing.blur_analysis import measure_dataset_blur_heatmaps
from src.utils.logging_utils import setup_logging

def get_preprocessing_args():
    parser = argparse.ArgumentParser(description="Run preprocessing pipeline for single cell datasets.")
    parser.add_argument("-i", "--input-dir", help="Path to the raw dataset directory (input)")
    parser.add_argument("-o", "--output-dir", help="Root output directory for all processed data")
    parser.add_argument("--test-size", type=float, help="Fraction of data for test set")
    parser.add_argument("--random-seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--split-folder", type=str, help="Folder name for split datasets")
    # blur detection parameters
    parser.add_argument("--patch-size", type=int, help="Patch size for blur detection")
    parser.add_argument("--stride-size", type=int, help="Stride size for blur detection")
    # Input patterns
    parser.add_argument("--combine-pattern", help="Regex for 2D to 3D grouping")
    parser.add_argument("--image-pattern", help="Glob pattern for image files")
    parser.add_argument("--mask-pattern", help="Glob pattern for mask files")
    # misc
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--override", nargs='*', help="Override configuration values in dot notation (e.g., paths.input_dir=data/input)")
    return parser.parse_args()

def get_preprocessing_legacy_args(args):
    """
    Extract legacy CLI arguments that are not part of the new config schema.
    This is for backward compatibility with existing scripts.
    """
    legacy_args = {}
    if args.input_dir:
        legacy_args['path.input_dir'] = args.input_dir
    if args.output_dir:
        legacy_args['path.output_dir'] = args.output_dir
    if args.test_size:
        legacy_args['preprocessing.test_size'] = args.test_size
    if args.random_seed:
        legacy_args['preprocessing.random_state'] = args.random_seed
    if args.split_folder:
        legacy_args['preprocessing.split_folder'] = args.split_folder
    if args.patch_size:
        legacy_args['quality.blur_detection.patch_size'] = args.patch_size
    if args.stride_size:
        legacy_args['quality.blur_detection.stride_size'] = args.stride_size
    if args.combine_pattern:
        legacy_args['preprocessing.combine_pattern'] = args.combine_pattern
    if args.image_pattern:
        legacy_args['preprocessing.raw_data_patterns.brightfield'] = args.image_pattern
    if args.mask_pattern:
        legacy_args['preprocessing.raw_data_patterns.masks'] = args.mask_pattern
    # if args.nuclei_pattern:
    #     legacy_args['preprocessing.raw_data_patterns.nuclei'] = args.nuclei_pattern

    return legacy_args

def run_preprocessing_from_config(config: Dict[str, Any], input_dir : Optional[Union[str, Path]] = None, output_dir: Optional[Union[str, Path]] = None):
    paths_config = config.get('paths', {})
    preprocessing_config = config.get('preprocessing', {})

    if input_dir is None:
        input_dir = paths_config.get('input_dir', 'data/input/')
    if output_dir is None:
        output_dir = paths_config.get('output_dir', 'data/output/')

    input_dir = Path(input_dir) # type: ignore
    assert input_dir.exists(), f"Input directory {input_dir} does not exist."

    output_dir = Path(output_dir) # type: ignore

    logger = logging.getLogger(__name__)
    logger.info("Starting data preparation...")

    # Step 1: Split dataset
    split_dir = output_dir / preprocessing_config.get('split_folder', 'split_data')
    logger.info(f"Splitting dataset into train/test at {split_dir} ...")
    train_test_split_directory(
        data_dir=input_dir,
        output_dir=split_dir,
        test_size=preprocessing_config.get('test_size', 0.2),
        random_state=preprocessing_config.get('random_state', 42),
        image_pattern=preprocessing_config.get('raw_data_patterns', {}).get('brightfield', 't1_*_w1_*.tif'),
        mask_pattern=preprocessing_config.get('raw_data_patterns', {}).get('masks', 'Cells_*.tif'),
        file_handler=BF_IF_FileHandler()
    )

    # Step 2: Combine 2D to 3D
    input_2d_dir = split_dir
    out_3d_folder = preprocessing_config.get('out_3d_folder', '3d_images')
    output_3d_dir = output_dir / out_3d_folder
    logger.info(f"Combining 2D images into 3D stacks at {output_3d_dir} ...")
    combine_2d_to_3d(
        input_dir=input_2d_dir,
        output_dir=output_3d_dir,
        pattern=preprocessing_config.get('combine_pattern', r"(.+?)_z(\d+)(?:_(BF|Cells))?\.(tif)"),
        recursive=True,
    )

    # Step 3: Generate blur heatmaps
    blur_config = config.get('quality', {}).get('blur_detection', {})
    blur_dir = output_dir / "blur_heatmaps"
    logger.info(f"Generating blur heatmaps at {blur_dir} ...")
    measure_dataset_blur_heatmaps(
        input_dir=output_3d_dir,
        output_dir=blur_dir,
        pattern="*_BF_3d.tif",
        patch_size=blur_config.get('patch_size', 32),
        stride_size=blur_config.get('stride_size', 16),
        normalize=True,
        overwrite=True,
    )
    logger.info("Preprocessing complete.")

def run_preprocessing_pipeline(args):
    """ 
    Legacy function to run the preprocessing pipeline from args.
    Run the preprocessing pipeline based on command-line arguments.
    This function orchestrates the entire preprocessing workflow.
    """
    # Step 1: Split dataset
    split_dir = os.path.join(args.output_root, "split")
    print(f"Splitting dataset into train/test at {split_dir} ...")
    train_test_split_directory(
        data_dir=args.dataset_path,
        output_dir=split_dir,
        test_size=args.test_size,
        random_state=args.random_seed,
        image_pattern=args.image_pattern,
        mask_pattern=args.mask_pattern,
        file_handler=BF_IF_FileHandler()
    )

    # Step 2: Combine 2D to 3D
    input_2d_dir = split_dir
    output_3d_dir = os.path.join(args.output_root, "3d_images")
    print(f"Combining 2D images into 3D stacks at {output_3d_dir} ...")
    combine_2d_to_3d(
        input_dir=input_2d_dir,
        output_dir=output_3d_dir,
        pattern=args.combine_pattern,
        recursive=True,
    )

    # Step 3: Generate blur heatmaps
    blur_dir = os.path.join(args.output_root, "blur_heatmaps")
    print(f"Generating blur heatmaps at {blur_dir} ...")
    measure_dataset_blur_heatmaps(
        input_dir=output_3d_dir,
        output_dir=blur_dir,
        pattern="*_BF_3d.tif",
        patch_size=args.patch_size,
        stride_size=args.stride_size,
        normalize=True,
        overwrite=args.overwrite
    )
    print("Preprocessing complete.")

def main():
    args = get_preprocessing_args()
    cli_args = vars(args)

    # Remove all None values from cli_args
    cli_args = {k: v for k, v in cli_args.items() if v is not None}

    print(cli_args) # Debugging line to see parsed arguments

    # Set up logging
    setup_logging(cli_args.get("log_level", "INFO"))

    # 1: Load base config (from YAML or default)
    if "config" in cli_args:
        config_manager = ConfigManager(cli_args["config"])
        logging.info(f"Loaded configuration from {args.config}")
    else:
        config_manager = ConfigManager()  # Use defaults
        logging.info("Using default configuration")

    # 2: Apply dotlist overrides from CLI
    if "override" in cli_args:
        from omegaconf import OmegaConf
        overrides = OmegaConf.from_dotlist(cli_args["override"])
        override_dict = OmegaConf.to_container(overrides)
        logging.info(f"Applying CLI overrides: {cli_args['override']}")
        config_manager = config_manager.merge_with_overrides(override_dict) #type: ignore
        # config_dict.update(override_dict)

    # 3: Apply legacy overrides and Merge config and CLI args
    legacy_overrides = get_preprocessing_legacy_args(args)
    if legacy_overrides:
        config_manager = config_manager.merge_with_overrides(legacy_overrides)
        logging.info(f"Applied legacy CLI overrides: {list(legacy_overrides.keys())}")

    # Get final config as dict for backward compatibility
    merged_config = config_manager.to_dict()
    logging.info("Final merged configuration:")
    
    try:
        logging.info("Starting preprocessing...")
        run_preprocessing_from_config(merged_config)

    except Exception as e:
        logging.error(f"Preprocessing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

