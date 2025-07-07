#!/usr/bin/env python3
"""
Run inference on test dataset using trained Cellpose models.

This script provides a command-line interface for running cell segmentation
inference on test datasets with organized output structure.

Usage:
    python scripts/run_inference.py --input-dir data/test --output-dir results --model-name cyto3
    
    python scripts/run_inference.py --config config/inference_config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.inference.inference_pipeline import InferencePipeline
from src.inference.cellpose_predictor import CellposePredictor
from src.inference.output_manager import OutputManager
from src.utils.config import load_config


def main():
    """Main inference function."""
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
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Load configuration if provided
        config = {}
        if args.config:
            config = load_config(args.config)
            logging.info(f"Loaded configuration from {args.config}")
        
        # Initialize predictor
        predictor_kwargs = {
            'model_type': args.model_name,
            'gpu': args.gpu,
            'flow_threshold': args.flow_threshold,
            'cellprob_threshold': args.cellprob_threshold,
            'min_size': args.min_size
        }
        
        if args.diameter:
            predictor_kwargs['diameter'] = args.diameter
        
        # Override with config if available
        if 'segmentation' in config and 'cellpose' in config['segmentation']:
            cellpose_config = config['segmentation']['cellpose']
            for key, value in cellpose_config.items():
                if key not in predictor_kwargs or predictor_kwargs[key] is None:
                    predictor_kwargs[key] = value
        
        predictor = CellposePredictor(**predictor_kwargs)
        
        # Load custom model if specified
        if args.model_path:
            predictor.load_model(args.model_path)
            logging.info(f"Loaded custom model from {args.model_path}")
        
        # Initialize output manager
        output_manager = OutputManager(
            base_output_dir=args.output_dir,
            model_name=args.model_name,
            dataset_name=args.dataset_name
        )
        
        # Initialize pipeline
        pipeline = InferencePipeline(predictor, output_manager, config)
        
        # Validate setup
        validation = pipeline.validate_setup()
        if not validation['overall']:
            logging.error("Pipeline validation failed")
            sys.exit(1)
        
        # Log model information
        model_info = pipeline.get_model_info()
        logging.info(f"Model information: {model_info}")
        
        # Run inference
        logging.info(f"Starting inference on {args.input_dir}")
        logging.info(f"Output will be saved to: {output_manager.output_dir}")
        
        results = pipeline.run_inference(
            input_dir=args.input_dir,
            file_pattern=args.file_pattern,
            process_z_stacks=args.process_z_stacks,
            save_overlays=not args.no_overlays,
            save_metadata=not args.no_metadata
        )
        
        # Print summary
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
        
    except Exception as e:
        logging.error(f"Inference failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
