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

from src.utils.logging_utils import setup_logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.inference.inference_pipeline import InferencePipeline
from src.inference.cellpose_predictor import CellposePredictor
from src.inference.output_manager import OutputManager
from src.utils.config import load_config


def get_inference_args():
    parser = argparse.ArgumentParser(description="Run inference on cell segmentation test dataset")
    
    # Input/Output arguments
    parser.add_argument(
        "--input-dir", "-i", 
        type=str, 
        required=True,
        help="Directory containing test images"
    )
    parser.add_argument(
        "--output-dir", "-o", 
        type=str, 
        required=True,
        help="Base output directory for results"
    )
    # TODO : model_type argument is not used in v4.0.1+. Ignoring this argument...
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
    
    return parser.parse_args()


def merge_config_and_args(config: dict, args) -> dict:
    """
    Merge config file dictionary and argparse.Namespace, with CLI args taking precedence.
    Returns a flat dictionary for easy access.
    """
    merged = dict(config) if config else {}
    for key, value in vars(args).items():
        if value is not None:
            merged[key] = value
    return merged


def run_inference(
    input_dir,
    output_dir,
    model_name="cyto3",
    dataset_name="test",
    file_pattern="*_BF.tif",
    process_z_stacks=False,
    no_overlays=False,
    no_metadata=False,
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
        model_name=model_name,
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
        save_overlays=not no_overlays,
        save_metadata=not no_metadata
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


def main():
    """Main inference function."""
    args = get_inference_args()
    
    # Set up logging
    setup_logging(args)
    
    # Concatenate dataset_name to input_dir
    input_dir = str(Path(args.input_dir) / args.dataset_name)
    
    try:
        # Load configuration if provided
        config = {}
        if args.config:
            config = load_config(args.config)
            logging.info(f"Loaded configuration from {args.config}")
        
        # Merge config and args, CLI args take precedence
        merged = merge_config_and_args(config, args)
        
        run_inference(
            input_dir=input_dir,
            output_dir=merged['output_dir'],
            model_name=merged.get('model_name', 'cyto3'),
            dataset_name=merged.get('dataset_name', 'test'),
            file_pattern=merged.get('file_pattern', '*_BF.tif'),
            process_z_stacks=merged.get('process_z_stacks', False),
            no_overlays=merged.get('no_overlays', False),
            no_metadata=merged.get('no_metadata', False),
            model_path=merged.get('model_path'),
            gpu=merged.get('gpu', True),
            flow_threshold=merged.get('flow_threshold', 0.4),
            cellprob_threshold=merged.get('cellprob_threshold', 0.0),
            min_size=merged.get('min_size', 30),
            diameter=merged.get('diameter'),
            config=merged
        )
    except Exception as e:
        logging.error(f"Inference failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
