#!/usr/bin/env python3
"""
Example script demonstrating how to use the inference pipeline.

This script shows how to set up and run inference on a test dataset
using the organized inference modules.
"""

import sys
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.inference.cellpose_predictor import CellposePredictor
from src.inference.output_manager import OutputManager
from src.inference.inference_pipeline import InferencePipeline


def example_basic_inference():
    """Example of basic inference setup and execution."""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    print("=== Basic Inference Example ===")
    
    # 1. Initialize the predictor
    predictor = CellposePredictor(
        model_type="cyto3",
        gpu=True,
        flow_threshold=0.4,
        min_size=30
    )
    
    # 2. Set up output management
    output_manager = OutputManager(
        base_output_dir="results",
        model_name="cyto3",
        dataset_name="test"
    )
    
    # 3. Create pipeline
    pipeline = InferencePipeline(predictor, output_manager)
    
    # 4. Validate setup
    validation = pipeline.validate_setup()
    if not validation['overall']:
        print("Pipeline validation failed!")
        return
    
    # 5. Run inference on test directory
    test_dir = "data/sample_plates_split/test"
    
    if Path(test_dir).exists():
        print(f"Running inference on: {test_dir}")
        
        results = pipeline.run_inference(
            input_dir=test_dir,
            file_pattern="*_BF.tif",
            save_overlays=True,
            save_metadata=True
        )
        
        print(f"\nResults:")
        print(f"  - Files processed: {len(results['processed_files'])}")
        print(f"  - Total cells detected: {results['total_cells']}")
        print(f"  - Output directory: {output_manager.output_dir}")
        
    else:
        print(f"Test directory not found: {test_dir}")
        print("Please update the path to your test data")


def example_config_based_inference():
    """Example of inference using configuration file."""
    
    print("\n=== Config-Based Inference Example ===")
    
    config_path = "config/inference_config.yaml"
    
    if Path(config_path).exists():
        pipeline = InferencePipeline.from_config(
            config_path=config_path,
            model_name="cyto3",
            output_dir="results",
            dataset_name="test_example"
        )
        
        # Get test directory from config
        test_dir = pipeline.config.get('paths', {}).get('test_data', 'data/sample_plates_split/test')
        
        if Path(test_dir).exists():
            print(f"Running inference with config on: {test_dir}")
            
            results = pipeline.run_inference(
                input_dir=test_dir,
                file_pattern="*_BF.tif"
            )
            
            print(f"\nResults:")
            print(f"  - Files processed: {len(results['processed_files'])}")
            print(f"  - Total cells detected: {results['total_cells']}")
            
        else:
            print(f"Test directory not found: {test_dir}")
            
    else:
        print(f"Config file not found: {config_path}")


def example_single_file_inference():
    """Example of running inference on a single file."""
    
    print("\n=== Single File Inference Example ===")
    
    # Initialize predictor
    predictor = CellposePredictor(model_type="cyto3", gpu=True)
    
    # Set up output
    output_manager = OutputManager(
        base_output_dir="results",
        model_name="cyto3_single",
        dataset_name="example"
    )
    
    # Create pipeline
    pipeline = InferencePipeline(predictor, output_manager)
    
    # Find a test file
    test_dir = Path("data/sample_plates_split/test")
    test_files = list(test_dir.glob("*_BF.tif")) if test_dir.exists() else []
    
    if test_files:
        test_file = test_files[0]
        print(f"Processing single file: {test_file}")
        
        result = pipeline.run_inference_single(test_file)
        
        print(f"Result:")
        print(f"  - Status: {result['status']}")
        print(f"  - Cells detected: {result.get('num_cells', 0)}")
        print(f"  - Image shape: {result.get('image_shape', 'Unknown')}")
        
    else:
        print("No test files found")


def example_custom_model_inference():
    """Example of using a custom trained model."""
    
    print("\n=== Custom Model Inference Example ===")
    
    # Check if custom model exists
    custom_model_path = "models/custom_cellpose_model"
    
    if Path(custom_model_path).exists():
        # Initialize with custom model
        predictor = CellposePredictor(model_type="cyto3", gpu=True)
        predictor.load_model(custom_model_path)
        
        output_manager = OutputManager(
            base_output_dir="results",
            model_name="custom_model",
            dataset_name="test"
        )
        
        pipeline = InferencePipeline(predictor, output_manager)
        
        print(f"Loaded custom model from: {custom_model_path}")
        print(f"Model info: {pipeline.get_model_info()}")
        
    else:
        print(f"Custom model not found at: {custom_model_path}")
        print("Using pretrained model instead")
        
        predictor = CellposePredictor(model_type="cyto3", gpu=True)
        print(f"Model info: {predictor.get_model_info()}")


if __name__ == "__main__":
    # Set environment variable for OpenMP conflict
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    print("Single Cell Analysis - Inference Examples")
    print("=" * 50)
    
    try:
        # Run examples
        example_basic_inference()
        example_config_based_inference()
        example_single_file_inference()
        example_custom_model_inference()
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have:")
        print("1. Installed the package in development mode: pip install -e .")
        print("2. Test data available in the expected directory")
        print("3. Cellpose properly installed")
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("Check the 'results/' directory for output files.")
