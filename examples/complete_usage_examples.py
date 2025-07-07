"""
Complete usage examples for the refactored cell segmentation system.

This script demonstrates how to use the modular training and evaluation
system for both Cellpose and Omnipose models.
"""

import os
import sys
from pathlib import Path

# Add src to path if running as script
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.config_presets import (
    create_training_config_from_preset, 
    create_evaluation_config_from_preset,
    list_presets,
    get_preset_info
)
from models.cellpose_trainer import CellposeTrainer, CellposeEvaluator
from tmp.omnipose_trainer import OmniposeTrainer, OmniposeEvaluator


def example_cellpose_training():
    """Example of training a Cellpose model."""
    print("=== Cellpose Training Example ===")
    
    # Set your data paths
    train_dir = "path/to/your/training/data"
    test_dir = "path/to/your/test/data"
    output_dir = "path/to/save/models"
    
    # Create training configuration using preset
    config = create_training_config_from_preset(
        "mammalian_cells_cellpose",  # Preset for mammalian cells
        train_dir=train_dir,
        test_dir=test_dir,
        output_dir=output_dir,
        # Override any preset parameters as needed
        n_epochs=50,  # Reduce epochs for faster training
        learning_rate=0.05,
        batch_size=4
    )
    
    # Initialize trainer
    trainer = CellposeTrainer(config)
    
    # Start training (commented out for example)
    # trainer.train()
    
    print("✓ Cellpose trainer configured successfully")
    print(f"  - Model type: {config.model_type}")
    print(f"  - Training epochs: {config.n_epochs}")
    print(f"  - Learning rate: {config.learning_rate}")


def example_omnipose_training():
    """Example of training an Omnipose model."""
    print("\n=== Omnipose Training Example ===")
    
    # Set your data paths
    train_dir = "path/to/your/bacterial/training/data"
    test_dir = "path/to/your/bacterial/test/data"
    output_dir = "path/to/save/omnipose/models"
    
    # Create training configuration for bacteria
    config = create_training_config_from_preset(
        "bacteria_omnipose",
        train_dir=train_dir,
        test_dir=test_dir,
        output_dir=output_dir,
        # Omnipose-specific overrides
        n_epochs=100,
        learning_rate=0.01,
        batch_size=2  # Small batch size for Omnipose
    )
    
    # Initialize trainer
    trainer = OmniposeTrainer(config)
    
    # Start training (commented out for example)
    # trainer.train()
    
    print("✓ Omnipose trainer configured successfully")
    print(f"  - Model type: {config.model_type}")
    print(f"  - Training epochs: {config.n_epochs}")
    print(f"  - Learning rate: {config.learning_rate}")


def example_cellpose_evaluation():
    """Example of evaluating a Cellpose model."""
    print("\n=== Cellpose Evaluation Example ===")
    
    # Set your data paths
    test_dir = "path/to/your/test/data"
    model_path = "path/to/your/trained/model"  # or use "cyto2" for pretrained
    output_dir = "path/to/save/evaluation/results"
    
    # Create evaluation configuration
    config = create_evaluation_config_from_preset(
        "default",  # Use default evaluation settings
        test_dir=test_dir,
        model_path=model_path,  # Can be path to custom model or "cyto2", "nuclei", etc.
        output_dir=output_dir,
        model_type="cyto2"
    )
    
    # Initialize evaluator
    evaluator = CellposeEvaluator(config)
    
    # Run evaluation (commented out for example)
    # results = evaluator.evaluate()
    
    print("✓ Cellpose evaluator configured successfully")
    print(f"  - Model: {config.model_path}")
    print(f"  - Min size: {config.min_size}")
    print(f"  - Flow threshold: {config.flow_threshold}")


def example_omnipose_evaluation():
    """Example of evaluating an Omnipose model."""
    print("\n=== Omnipose Evaluation Example ===")
    
    # Set your data paths
    test_dir = "path/to/your/bacterial/test/data"
    model_path = "bact_phase_omni"  # Pretrained Omnipose model
    output_dir = "path/to/save/omnipose/evaluation/results"
    
    # Create evaluation configuration for bacteria
    config = create_evaluation_config_from_preset(
        "bacteria",  # Bacteria-specific evaluation settings
        test_dir=test_dir,
        model_path=model_path,
        output_dir=output_dir,
        model_type="bact_phase_omni"
    )
    
    # Initialize evaluator
    evaluator = OmniposeEvaluator(config)
    
    # Run evaluation (commented out for example)
    # results = evaluator.evaluate()
    
    print("✓ Omnipose evaluator configured successfully")
    print(f"  - Model: {config.model_path}")
    print(f"  - Min size: {config.min_size}")
    print(f"  - Flow threshold: {config.flow_threshold}")


def example_custom_configuration():
    """Example of creating custom configurations."""
    print("\n=== Custom Configuration Example ===")
    
    # Create completely custom training config
    from models.base_trainer import TrainingConfig
    
    custom_config = TrainingConfig(
        model_type="cyto2",
        train_dir="path/to/custom/training/data",
        test_dir="path/to/custom/test/data",
        output_dir="path/to/custom/models",
        channels=[0, 0],  # Grayscale
        learning_rate=0.1,
        weight_decay=1e-4,
        n_epochs=200,
        batch_size=8,
        min_train_masks=10,
        diameter=30,  # Custom diameter
        normalize=True,
        image_filter="_brightfield",  # Custom file naming
        mask_filter="_masks"
    )
    
    trainer = CellposeTrainer(custom_config)
    
    print("✓ Custom configuration created successfully")
    print(f"  - Custom diameter: {custom_config.diameter}")
    print(f"  - Custom file filters: {custom_config.image_filter}, {custom_config.mask_filter}")


def show_available_presets():
    """Show all available presets."""
    print("\n=== Available Presets ===")
    list_presets()
    
    # Show details for a specific preset
    print("\nDetailed info for 'bacteria_cellpose' preset:")
    info = get_preset_info("bacteria_cellpose", "training")
    for key, value in info.items():
        print(f"  {key}: {value}")


def main():
    """Run all examples."""
    print("Cell Segmentation System - Usage Examples")
    print("=" * 50)
    
    # Show available presets first
    show_available_presets()
    
    # Training examples
    example_cellpose_training()
    example_omnipose_training()
    
    # Evaluation examples
    example_cellpose_evaluation()
    example_omnipose_evaluation()
    
    # Custom configuration
    example_custom_configuration()
    
    print("\n" + "=" * 50)
    print("SUCCESS: All examples completed successfully!")
    print("\nTo actually run training or evaluation:")
    print("1. Replace the 'path/to/your/...' placeholders with real paths")
    print("2. Uncomment the trainer.train() or evaluator.evaluate() calls")
    print("3. Run the script")
    
    print("\nFor more detailed guidance, see:")
    print("- docs/TRAINING_GUIDE.md")
    print("- examples/training_examples.py")


if __name__ == "__main__":
    main()
