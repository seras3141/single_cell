#!/usr/bin/env python3
"""
Run inference on test dataset using trained Cellpose models.

This script provides a command-line interface for running cell segmentation
inference on test datasets with organized output structure.

#TODO : Update usage
Usage:
    python scripts/run_inference.py --input-dir data/test --output-dir results --model-name cyto3
    
    python scripts/run_inference.py --config config/inference_config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.inference.inference_pipeline import InferencePipeline
from src.inference.cellpose_predictor import CellposePredictor
from src.inference.output_manager import OutputManager

from src.utils.logging_utils import setup_logging
from src.utils.config import ConfigManager

def get_inference_args():
    parser = argparse.ArgumentParser(description="Run inference on cell segmentation test dataset")

    def optional_arg(*args, **kwargs):
        kwargs['default'] = argparse.SUPPRESS  # Don't set a default so we can detect user input
        return parser.add_argument(*args, **kwargs)

    # Define args like this:
    optional_arg("--input-dir", "-i", type=str, help="Directory containing test images")
    optional_arg("--output-dir", "-o", type=str, help="Base output directory for results")
    # TODO : model_type argument is not used in v4.0.1+. Ignoring this argument...
    optional_arg("--model-name", "-m", type=str, help="Model name/type to use (default: cyto3)")
    optional_arg("--dataset-name", type=str, help="Dataset name (default: test)")
    optional_arg("--file-pattern", type=str, help="File pattern (default: *_BF.tif)")
    optional_arg("--process-z-stacks", action="store_true", help="Process images as Z-stacks")
    optional_arg("--no-overlays", action="store_true", help="Skip overlay visualizations")
    optional_arg("--no-metadata", action="store_true", help="Skip saving prediction metadata")
    optional_arg("--model-path", type=str, help="Path to custom model weights")
    optional_arg("--gpu", dest="gpu", action="store_true", help="Use GPU for inference")
    optional_arg("--no-gpu", dest="gpu", action="store_false", help="Disable GPU usage")
    optional_arg("--flow-threshold", type=float, help="Flow error threshold")
    optional_arg("--cellprob-threshold", type=float, help="Cell probability threshold")
    optional_arg("--min-size", type=int, help="Minimum cell size")
    optional_arg("--diameter", type=float, help="Expected cell diameter")
    optional_arg("--config", type=str, help="Path to YAML config file")
    optional_arg("--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    optional_arg("--override", "-O", action="append", help="Config overrides in dot notation")
    
    return parser.parse_args()

    '''
    parser.add_argument(
        "--model-name", "-m", 
        type=str, 
        default="cyto3",
        help="Model name/type to use (default: cyto3)"
    )
    parser.add_argument(
        "--dataset-name", 
        type=str, 
        default="test",
        help="Name of the dataset being processed (default: test)"
    )
    
    # Processing arguments
    parser.add_argument(
        "--file-pattern", 
        type=str, 
        default="*_BF.tif",
        help="Glob pattern for input files (default: *_BF.tif)"
    )
    parser.add_argument(
        "--process-z-stacks", 
        action="store_true",
        help="Process images as Z-stacks"
    )
    parser.add_argument(
        "--no-overlays", 
        action="store_true",
        help="Skip saving overlay visualizations"
    )
    parser.add_argument(
        "--no-metadata", 
        action="store_true",
        help="Skip saving prediction metadata"
    )
    
    # Model arguments
    parser.add_argument(
        "--model-path", 
        type=str,
        help="Path to custom trained model weights"
    )
    parser.add_argument(
        "--gpu", 
        action="store_true", 
        default=True,
        help="Use GPU for inference (default: True)"
    )
    parser.add_argument(
        "--no-gpu", 
        dest="gpu", 
        action="store_false",
        help="Disable GPU usage"
    )
    
    # Cellpose parameters
    parser.add_argument(
        "--flow-threshold", 
        type=float, 
        default=0.4,
        help="Flow error threshold for Cellpose (default: 0.4)"
    )
    parser.add_argument(
        "--cellprob-threshold", 
        type=float, 
        default=0.0,
        help="Cell probability threshold (default: 0.0)"
    )
    parser.add_argument(
        "--min-size", 
        type=int, 
        default=30,
        help="Minimum cell size in pixels (default: 30)"
    )
    parser.add_argument(
        "--diameter", 
        type=float,
        help="Expected cell diameter (default: auto)"
    )
    
    # Configuration file
    parser.add_argument(
        "--config", 
        type=str,
        help="Path to configuration YAML file"
    )
    
    # Logging
    parser.add_argument(
        "--log-level", 
        type=str, 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    # Configuration overrides (support dot notation)
    parser.add_argument(
        "--override", "-O",
        action="append",
        help="Configuration overrides in dot notation (e.g., 'segmentation.cellpose.gpu=false')"
    )
    
    return parser.parse_args()
    '''

def get_legacy_args(cli_args):
    # Apply legacy CLI argument overrides for backward compatibility
    legacy_overrides = {}
    # Legacy arguments for input/output directories
    if 'input_dir' in cli_args:
        legacy_overrides['paths.input_dir'] = cli_args['input_dir']
    if 'output_dir' in cli_args:
        legacy_overrides['paths.output_dir'] = cli_args['output_dir']
    # Legacy arguments for inference parameters
    if 'dataset_name' in cli_args:
        legacy_overrides['segmentation.inference.dataset_name'] = cli_args['dataset_name']
    if 'file_pattern' in cli_args:
        legacy_overrides['segmentation.inference.file_pattern'] = cli_args['file_pattern']
    if 'no_overlays' in cli_args:
        legacy_overrides['segmentation.inference.save_overlays'] = not cli_args['no_overlays']
    if 'no_metadata' in cli_args:
        legacy_overrides['segmentation.inference.save_metadata'] = not cli_args['no_metadata']
    if 'process_z_stacks' in cli_args:
        legacy_overrides['segmentation.inference.process_z_stacks'] = cli_args['process_z_stacks']
    # Legacy arguments for Cellpose parameters
    if 'gpu' in cli_args:
        legacy_overrides['segmentation.cellpose.gpu'] = cli_args['gpu']
    if 'flow_threshold' in cli_args:
        legacy_overrides['segmentation.cellpose.flow_threshold'] = cli_args['flow_threshold']
    if 'cellprob_threshold' in cli_args:
        legacy_overrides['segmentation.cellpose.cellprob_threshold'] = cli_args['cellprob_threshold']
    if 'min_size' in cli_args:
        legacy_overrides['segmentation.cellpose.min_size'] = cli_args['min_size']
    if 'diameter' in cli_args:
        legacy_overrides['segmentation.cellpose.diameter'] = cli_args['diameter']
    if 'model_name' in cli_args:
        legacy_overrides['segmentation.cellpose.model_type'] = cli_args['model_name']

    return legacy_overrides

def run_inference(
    input_dir,
    output_dir,
    model_name="cyto3",
    dataset_name="test",
    file_pattern="*_BF.tif",
    process_z_stacks=False,
    save_overlays=False,
    save_metadata=False,
    model_path=None,
    gpu=True,
    flow_threshold=0.4,
    cellprob_threshold=0.0,
    min_size=30,
    diameter=None,
    config=None
):
    """Run the inference pipeline with the provided parameters."""
    logging.info(f"Starting inference on {input_dir}")

    # TODO : Check if this is the right order to get arguments vs config
    predictor_kwargs = {
        'model_type': model_name,
        'gpu': gpu,
        'flow_threshold': flow_threshold,
        'cellprob_threshold': cellprob_threshold,
        'min_size': min_size,
        'diameter': diameter
    }

    # Override with config if available
    if config and 'segmentation' in config and 'cellpose' in config['segmentation']:
        cellpose_config = config['segmentation']['cellpose']
        for key, value in cellpose_config.items():
            if key not in predictor_kwargs or predictor_kwargs[key] is None:
                predictor_kwargs[key] = value
    predictor = CellposePredictor(**predictor_kwargs)

    if model_path:
        predictor.load_model(model_path)
        logging.info(f"Loaded custom model from {model_path}")

    output_manager = OutputManager(
        base_output_dir=output_dir,
        model_name=predictor_kwargs['model_type'],
        dataset_name=dataset_name
    )

    pipeline = InferencePipeline(predictor, output_manager, config or {})
    validation = pipeline.validate_setup()
    if not validation['overall']:
        logging.error("Pipeline validation failed")
        sys.exit(1)

    model_info = pipeline.get_model_info()
    logging.info(f"Model information: {model_info}")

    results = pipeline.run_inference(
        input_dir=input_dir,
        file_pattern=file_pattern,
        process_z_stacks=process_z_stacks,
        save_overlays=save_overlays,
        save_metadata=save_metadata
    )

    print("\n" + "="*50)
    print("INFERENCE COMPLETED")
    print("="*50)
    print(f"Total files processed: {len(results['processed_files'])}")
    print(f"Total cells detected: {results['total_cells']}")
    print(f"Failed files: {len(results['failed_files'])}")
    print(f"Results saved to: {output_manager.output_dir}")
    print(f"Summary report: {results['summary_path']}")
    if results['failed_files']:
        print("\nFailed files:")
        for failed in results['failed_files']:
            print(f"  - {failed['file']}: {failed['error']}")
    logging.info("Inference completed successfully")

def run_inference_from_config(config : Dict[str, Any], input_dir: Optional[str] = None, dataset_name : Optional[str] = None):
    """
    Run inference using a configuration file and command-line arguments.
    Merges config file with CLI args, with CLI args taking precedence.
    """

    pipeline = InferencePipeline.from_config(config=config)

    inference_config = config.get('segmentation', {}).get('inference', {})

    if input_dir is None:
        input_dir = config.get('paths', {}).get('input_dir', 'data/test')
    if dataset_name is None:
        dataset_name = inference_config.get('dataset_name', 'test')

    pipeline.run_inference(
        input_dir=Path(input_dir) / dataset_name, # type: ignore
        file_pattern=inference_config.get('file_pattern', '*_BF.tif'),
        process_z_stacks=inference_config.get('process_z_stacks', False),
        save_overlays=inference_config.get('save_overlays', False),
        save_metadata=inference_config.get('save_metadata', False)
    )

def main():
    """Main inference function with OmegaConf configuration support."""
    args = get_inference_args()
    cli_args = vars(args)

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
    merged_config = config_manager.to_dict()
    logging.info("Final merged configuration:")
    
    try:
        logging.info("Starting inference...")
        run_inference_from_config(merged_config)

    except Exception as e:
        logging.error(f"Inference failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
