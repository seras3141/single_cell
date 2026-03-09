"""
Example scripts for training and evaluating 2D cell segmentation models.

This script demonstrates how to use the modular training and evaluation
system for both Cellpose and Omnipose models.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.base_trainer import TrainingConfig, EvaluationConfig
from models.cellpose_trainer import train_cellpose_2d, evaluate_cellpose_2d
from tmp.omnipose_trainer import train_omnipose_2d, evaluate_omnipose_2d


def example_cellpose_training():
    """Example of training a Cellpose model."""
    print("=== Cellpose Training Example ===")
    
    # Configure training
    config = TrainingConfig(
        train_dir="data/train",
        test_dir="data/test", 
        output_dir="models/cellpose_trained",
        model_type="cyto2",
        channels=[0, 0],  # Grayscale
        diameter=None,    # Auto-detect
        learning_rate=0.1,
        weight_decay=1e-4,
        n_epochs=50,
        batch_size=8,
        gpu=True,
        image_filter="_BF",
        mask_filter="_Cells",
        normalize=True
    )
    
    # Train model
    try:
        results = train_cellpose_2d(config)
        print(f"Training completed! Model saved to: {results.model_path}")
        print(f"Training time: {results.training_time:.2f} seconds")
        print(f"Final training loss: {results.train_losses[-1]:.6f}")
        return results.model_path
    except Exception as e:
        print(f"Training failed: {e}")
        return None


def example_cellpose_evaluation(model_path: str):
    """Example of evaluating a Cellpose model."""
    print("=== Cellpose Evaluation Example ===")
    
    # Configure evaluation
    config = EvaluationConfig(
        test_dir="data/test",
        model_path=model_path,
        output_dir="results/cellpose_predictions",
        model_type="cyto2",
        channels=[0, 0],
        diameter=None,
        flow_threshold=0.4,
        cellprob_threshold=0.0,
        min_size=30,
        gpu=True,
        image_filter="_BF",
        save_masks=True,
        save_outlines=True,
        save_flows=False
    )
    
    # Evaluate model
    try:
        results = evaluate_cellpose_2d(config)
        print(f"Evaluation completed! Processed {len(results.predictions)} images")
        print(f"Evaluation time: {results.evaluation_time:.2f} seconds")
        print("Metrics:")
        for key, value in results.metrics.items():
            print(f"  {key}: {value}")
        return results
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return None


def example_omnipose_training():
    """Example of training an Omnipose model."""
    print("=== Omnipose Training Example ===")
    
    # Configure training
    config = TrainingConfig(
        train_dir="data/train",
        test_dir="data/test",
        output_dir="models/omnipose_trained", 
        model_type="bact_phase_omni",  # Bacterial model for Omnipose
        channels=[0, 0],
        diameter=None,
        learning_rate=0.05,  # Lower learning rate for Omnipose
        weight_decay=1e-4,
        n_epochs=100,
        batch_size=4,  # Smaller batch size
        gpu=True,
        image_filter="_BF",
        mask_filter="_Cells",
        normalize=True
    )
    
    # Train model
    try:
        results = train_omnipose_2d(config)
        print(f"Training completed! Model saved to: {results.model_path}")
        print(f"Training time: {results.training_time:.2f} seconds")
        print(f"Final training loss: {results.train_losses[-1]:.6f}")
        return results.model_path
    except Exception as e:
        print(f"Training failed: {e}")
        return None


def example_omnipose_evaluation(model_path: str):
    """Example of evaluating an Omnipose model."""
    print("=== Omnipose Evaluation Example ===")
    
    # Configure evaluation
    config = EvaluationConfig(
        test_dir="data/test",
        model_path=model_path,
        output_dir="results/omnipose_predictions",
        model_type="bact_phase_omni",
        channels=[0, 0],
        diameter=None,
        flow_threshold=0.4,
        cellprob_threshold=0.0,
        min_size=15,  # Smaller minimum size for bacteria
        gpu=True,
        image_filter="_BF", 
        save_masks=True,
        save_outlines=True,
        save_flows=False
    )
    
    # Evaluate model
    try:
        results = evaluate_omnipose_2d(config)
        print(f"Evaluation completed! Processed {len(results.predictions)} images")
        print(f"Evaluation time: {results.evaluation_time:.2f} seconds")
        print("Metrics:")
        for key, value in results.metrics.items():
            print(f"  {key}: {value}")
        return results
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return None


def compare_models():
    """Example of comparing Cellpose and Omnipose models."""
    print("=== Model Comparison Example ===")
    
    # Use pretrained models for quick comparison
    cellpose_config = EvaluationConfig(
        test_dir="data/test",
        model_path="cyto2",  # Pretrained model
        output_dir="results/cellpose_comparison",
        model_type="cyto2",
        channels=[0, 0],
        gpu=True,
        image_filter="_BF"
    )
    
    omnipose_config = EvaluationConfig(
        test_dir="data/test", 
        model_path="bact_phase_omni",  # Pretrained model
        output_dir="results/omnipose_comparison",
        model_type="bact_phase_omni",
        channels=[0, 0],
        gpu=True,
        image_filter="_BF"
    )
    
    try:
        print("Evaluating Cellpose...")
        cellpose_results = evaluate_cellpose_2d(cellpose_config)
        
        print("Evaluating Omnipose...")
        omnipose_results = evaluate_omnipose_2d(omnipose_config)
        
        # Compare results
        print("\n=== Comparison Results ===")
        print(f"Cellpose - Images processed: {len(cellpose_results.predictions)}")
        print(f"Cellpose - Evaluation time: {cellpose_results.evaluation_time:.2f}s")
        print(f"Cellpose - Avg cells per image: {cellpose_results.metrics.get('avg_cells_per_image', 0):.1f}")
        
        print(f"Omnipose - Images processed: {len(omnipose_results.predictions)}")  
        print(f"Omnipose - Evaluation time: {omnipose_results.evaluation_time:.2f}s")
        print(f"Omnipose - Avg cells per image: {omnipose_results.metrics.get('avg_cells_per_image', 0):.1f}")
        
    except Exception as e:
        print(f"Comparison failed: {e}")


def main():
    """Run all examples."""
    print("Cell Segmentation Training and Evaluation Examples")
    print("=" * 50)
    
    # Ensure output directories exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Example 1: Train and evaluate Cellpose
    print("\n1. Training Cellpose model...")
    cellpose_model = example_cellpose_training()
    
    if cellpose_model:
        print("\n2. Evaluating Cellpose model...")
        example_cellpose_evaluation(cellpose_model)
    
    # Example 2: Train and evaluate Omnipose  
    print("\n3. Training Omnipose model...")
    omnipose_model = example_omnipose_training()
    
    if omnipose_model:
        print("\n4. Evaluating Omnipose model...")
        example_omnipose_evaluation(omnipose_model)
    
    # Example 3: Compare pretrained models
    print("\n5. Comparing pretrained models...")
    compare_models()
    
    print("\nAll examples completed!")


if __name__ == "__main__":
    main()
