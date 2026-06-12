import argparse
import logging
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Union

from src.preprocessing.dataset_split import train_test_split_directory
from src.utils.config import get_config_manager
from src.utils.file_utils import BF_IF_FileHandler, ConfigurableFileHandler, EXPERIMENT_WAVELENGTH_MAPPINGS
from src.utils.conversion import combine_2d_to_3d
from src.preprocessing.blur_analysis import generate_blur_heatmap_batch
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
    # File handler
    parser.add_argument(
        "--experiment-name",
        type=str,
        choices=list(EXPERIMENT_WAVELENGTH_MAPPINGS.keys()),
        help="Experiment name — automatically sets wavelength mappings (e.g. 'Ew2-1', 'HD1509'). Overridden by --wavelengths if both are provided."
    )
    parser.add_argument("--wavelengths", type=str, help="Wavelength-to-channel mappings as comma-separated key:value pairs (e.g. '1:FlipGFP,2:mCherry,3:BF'). Overrides --experiment-name.")
    parser.add_argument("--plate", type=str, help="Default plate number for file renaming (overrides auto-detection from filepath)")
    # Z-range for 2D-to-3D combination
    parser.add_argument("--z-min", type=int, help="Minimum z-index to include when combining 2D to 3D (default: 1, skipping z0 which is the 2D projection)")
    parser.add_argument("--z-max", type=int, help="Maximum z-index to include when combining 2D to 3D (default: no limit)")
    # misc
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--override", nargs='*', help="Override configuration values in dot notation (e.g., paths.input_dir=data/input)")
    # Step selection (all enabled by default; pass flags to disable individual steps)
    step_group = parser.add_argument_group('step selection', 'Control which pipeline steps are executed')
    step_group.add_argument("--skip-split", action="store_true", help="Skip train/test split step")
    step_group.add_argument("--skip-3d", action="store_true", help="Skip 2D-to-3D combination step")
    step_group.add_argument("--skip-blur", action="store_true", help="Skip blur heatmap generation step")
    return parser.parse_args()

def get_preprocessing_legacy_args(vargs: dict) -> Dict[str, Any]:
    """
    Extract legacy CLI arguments that are not part of the new config schema.
    This is for backward compatibility with existing scripts.
    """
    legacy_mapping = {
        # No direct legacy args for now; placeholder for future use
        'input_dir': 'paths.input_dir',
        'output_dir': 'paths.output_dir',
        'test_size': 'preprocessing.test_size',
        'random_seed': 'preprocessing.random_state',
        'split_folder': 'preprocessing.split_folder',
        'combine_pattern': 'preprocessing.combine_pattern',
        'patch_size': 'quality.blur_detection.patch_size',
        'stride_size': 'quality.blur_detection.stride_size',
        'plate': 'preprocessing.plate_number',
        'z_min': 'preprocessing.z_min',
        'z_max': 'preprocessing.z_max',
        'skip_split': 'preprocessing.skip_split',
        'skip_3d': 'preprocessing.skip_3d',
        'skip_blur': 'preprocessing.skip_blur',
        'overwrite': 'preprocessing.overwrite',
    }

    legacy_args = {}

    for k, v in legacy_mapping.items():
        if k in vargs:
            legacy_args[v] = vargs[k]

    # --experiment-name sets wavelength mappings from the known-experiment lookup table
    if vargs.get('experiment_name'):
        legacy_args['preprocessing.wavelength_mappings'] = EXPERIMENT_WAVELENGTH_MAPPINGS[vargs['experiment_name']]

    # --wavelengths overrides --experiment-name if both are provided
    if vargs.get('wavelengths'):
        try:
            mappings = {}
            for pair in vargs['wavelengths'].split(','):
                key, val = pair.strip().split(':')
                mappings[int(key.strip())] = val.strip()
            legacy_args['preprocessing.wavelength_mappings'] = mappings
        except (ValueError, AttributeError) as e:
            raise ValueError(
                f"Invalid --wavelengths format '{vargs['wavelengths']}'. "
                f"Expected '1:FlipGFP,2:mCherry,3:BF'. Error: {e}"
            )

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

    run_split = not preprocessing_config.get('skip_split', False)
    run_3d    = not preprocessing_config.get('skip_3d', False)
    run_blur  = not preprocessing_config.get('skip_blur', False)
    overwrite = preprocessing_config.get('overwrite', False)
    logger.info(f"Steps enabled — split: {run_split}, 3d: {run_3d}, blur: {run_blur}, overwrite: {overwrite}")

    # Step 1: Split dataset
    split_dir = output_dir / preprocessing_config.get('split_folder', 'split_data')
    if run_split:
        logger.info(f"Splitting dataset into train/test at {split_dir} ...")
        # Build ConfigurableFileHandler with optional wavelength mappings and plate number from config/CLI
        wl_mappings_raw = preprocessing_config.get('wavelength_mappings') or {}
        wl_mappings = {int(k): v for k, v in wl_mappings_raw.items()} if wl_mappings_raw else None
        plate_number = preprocessing_config.get('plate_number') or None
        file_handler = ConfigurableFileHandler(
            wavelength_mappings=wl_mappings,
            plate_number=plate_number,
        )
        train_test_split_directory(
            data_dir=input_dir,
            output_dir=split_dir,
            test_size=preprocessing_config.get('test_size', 0.2),
            random_state=preprocessing_config.get('random_state', 42),
            file_handler=file_handler,
            overwrite=overwrite,
        )
    else:
        logger.info("Skipping split step.")

    # Step 2: Combine 2D to 3D
    input_2d_dir = split_dir
    out_3d_folder = preprocessing_config.get('out_3d_folder', '3d_images')
    output_3d_dir = output_dir / out_3d_folder
    if run_3d:
        logger.info(f"Combining 2D images into 3D stacks at {output_3d_dir} ...")
        z_min = preprocessing_config.get('z_min', 1)
        z_max = preprocessing_config.get('z_max', None)
        logger.info(f"Z-range filter: z_min={z_min}, z_max={z_max} (z0 is the 2D projection and is skipped by default)")
        combine_2d_to_3d(
            input_dir=input_2d_dir,
            output_dir=output_3d_dir,
            pattern=preprocessing_config.get('combine_pattern', r"(.+?)_z(\d+)(?:_(BF|Cells))?\.(tif)"),
            recursive=True,
            z_min=z_min,
            z_max=z_max,
            overwrite=overwrite,
        )
    else:
        logger.info("Skipping 2D-to-3D combination step.")

    # Step 3: Generate blur heatmaps
    blur_config = config.get('quality', {}).get('blur_detection', {})
    blur_dir = output_dir / "blur_heatmaps"
    if run_blur:
        logger.info(f"Generating blur heatmaps at {blur_dir} ...")
        # Config stores patch/stride as [h, w] lists; the function expects a single int
        patch_size = blur_config.get('patch_size', 32)
        stride_size = blur_config.get('stride_size', 16)
        if isinstance(patch_size, (list, tuple)):
            patch_size = int(patch_size[0])
        if isinstance(stride_size, (list, tuple)):
            stride_size = int(stride_size[0])
        generate_blur_heatmap_batch(
            input_dir=output_3d_dir,
            output_dir=blur_dir,
            pattern="*_BF_3d.tif",
            patch_size=patch_size,
            stride_size=stride_size,
            normalize=True,
            overwrite=overwrite,
        )
    else:
        logger.info("Skipping blur heatmap generation step.")

    logger.info("Preprocessing complete.")

    # Step 4 : Optional - Create symlinks for images from input directory to output_dir
    if preprocessing_config.get('create_symlinks', False):
        symlink_dir = output_dir / "input_data_symlinks"
        logger.info("Creating symlinks for images in output directory...")
        for image_path in input_2d_dir.glob("**/*.tif"):
            image_name = image_path.name
            relative_path = image_path.relative_to(input_2d_dir)
            symlink_path = symlink_dir / image_name
            print(f"Creating symlink: {symlink_path} -> {relative_path}")
            # symlink_path.parent.mkdir(parents=True, exist_ok=True)
            # if not symlink_path.exists():
            #     symlink_path.symlink_to(image_path)
            #     logger.info(f"Created symlink: {symlink_path} -> {image_path}")
    
def main():
    args = get_preprocessing_args()
    cli_args = vars(args)

    # Set up logging
    setup_logging(cli_args.get("log_level", "INFO"))
    logger = logging.getLogger(__name__)

    config_manager = get_config_manager(cli_args=cli_args, legacy_args_function=get_preprocessing_legacy_args)

    # Get final config as dict for backward compatibility
    merged_config = config_manager.to_dict()
    logger.info("Final merged configuration:")
    logger.info(merged_config)
    
    try:
        logger.info("Starting preprocessing...")
        run_preprocessing_from_config(merged_config)

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

