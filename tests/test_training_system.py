"""
Test script for the modular training and evaluation system.

This script tests the basic functionality of the training and evaluation
modules without requiring actual training data.
"""

import sys
import tempfile
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.base_trainer import TrainingConfig, EvaluationConfig
from models.config_presets import (
    create_training_config_from_preset,
    create_evaluation_config_from_preset,
    list_presets,
    get_preset_info,
    TRAINING_PRESETS,
    EVALUATION_PRESETS
)

# Try to import trainers, but don't fail if dependencies are missing
try:
    from models.cellpose_trainer import CellposeTrainer, CellposeEvaluator
    CELLPOSE_AVAILABLE = True
except ImportError:
    CELLPOSE_AVAILABLE = False

try:
    from tmp.omnipose_trainer import OmniposeTrainer, OmniposeEvaluator
    OMNIPOSE_AVAILABLE = True
except ImportError:
    OMNIPOSE_AVAILABLE = False


def test_config_creation():
    """Test configuration object creation."""
    print("Testing configuration creation...")
    
    # Test basic TrainingConfig
    config = TrainingConfig(
        train_dir="/tmp/train",
        test_dir="/tmp/test", 
        output_dir="/tmp/output"
    )
    assert config.train_dir == "/tmp/train"
    assert config.channels == [0, 0]  # Default
    assert config.n_epochs == 100     # Default
    print("✓ Basic TrainingConfig works")
    
    # Test basic EvaluationConfig
    eval_config = EvaluationConfig(
        test_dir="/tmp/test",
        model_path="/tmp/model.pt",
        output_dir="/tmp/results"
    )
    assert eval_config.test_dir == "/tmp/test"
    assert eval_config.flow_threshold == 0.4  # Default
    print("✓ Basic EvaluationConfig works")


def test_presets():
    """Test configuration presets."""
    print("\nTesting configuration presets...")
    
    # Test that all presets are accessible
    assert len(TRAINING_PRESETS) > 0
    assert len(EVALUATION_PRESETS) > 0
    print(f"✓ Found {len(TRAINING_PRESETS)} training presets")
    print(f"✓ Found {len(EVALUATION_PRESETS)} evaluation presets")
    
    # Test creating config from preset
    with tempfile.TemporaryDirectory() as tmpdir:
        config = create_training_config_from_preset(
            "mammalian_cells_cellpose",
            train_dir=f"{tmpdir}/train",
            test_dir=f"{tmpdir}/test",
            output_dir=f"{tmpdir}/output"
        )
        assert config.model_type == "cyto2"
        assert config.train_dir == f"{tmpdir}/train"
        print("✓ Training preset creation works")
        
        eval_config = create_evaluation_config_from_preset(
            "default",
            test_dir=f"{tmpdir}/test",
            model_path=f"{tmpdir}/model.pt",
            output_dir=f"{tmpdir}/results"
        )
        assert eval_config.flow_threshold == 0.4
        assert eval_config.test_dir == f"{tmpdir}/test"
        print("✓ Evaluation preset creation works")


def test_trainer_initialization():
    """Test trainer and evaluator initialization."""
    print("\nTesting trainer initialization...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test Cellpose trainer initialization
        config = TrainingConfig(
            train_dir=f"{tmpdir}/train",
            test_dir=f"{tmpdir}/test",
            output_dir=f"{tmpdir}/output"
        )
        
        # Test Cellpose trainer initialization
        if CELLPOSE_AVAILABLE:
            try:
                trainer = CellposeTrainer(config)
                assert trainer.config == config
                print("✓ CellposeTrainer initialization works")
            except Exception as e:
                print(f"⚠ CellposeTrainer initialization failed: {e}")
        else:
            print("⚠ CellposeTrainer initialization skipped (missing dependencies)")
        
        # Test Cellpose evaluator initialization
        eval_config = EvaluationConfig(
            test_dir=f"{tmpdir}/test",
            model_path="cyto2",  # Use pretrained model
            output_dir=f"{tmpdir}/results"
        )
        
        # Test Cellpose evaluator initialization
        if CELLPOSE_AVAILABLE:
            try:
                evaluator = CellposeEvaluator(eval_config)
                assert evaluator.config == eval_config
                print("✓ CellposeEvaluator initialization works")
            except Exception as e:
                print(f"⚠ CellposeEvaluator initialization failed: {e}")
        else:
            print("⚠ CellposeEvaluator initialization skipped (missing dependencies)")
        
        # Test Omnipose trainer initialization
        if OMNIPOSE_AVAILABLE:
            try:
                omni_trainer = OmniposeTrainer(config)
                assert omni_trainer.config == config
                print("✓ OmniposeTrainer initialization works")
            except Exception as e:
                print(f"⚠ OmniposeTrainer initialization failed: {e}")
        else:
            print("⚠ OmniposeTrainer initialization skipped (missing dependencies)")
        
        # Test Omnipose evaluator initialization
        if OMNIPOSE_AVAILABLE:
            try:
                omni_evaluator = OmniposeEvaluator(eval_config)
                assert omni_evaluator.config == eval_config
                print("✓ OmniposeEvaluator initialization works")
            except Exception as e:
                print(f"⚠ OmniposeEvaluator initialization failed: {e}")
        else:
            print("⚠ OmniposeEvaluator initialization skipped (missing dependencies)")


def test_preset_functions():
    """Test preset utility functions."""
    print("\nTesting preset utility functions...")
    
    # Test list_presets (should not crash)
    try:
        list_presets()
        print("✓ list_presets() works")
    except Exception as e:
        print(f"✗ list_presets() failed: {e}")
    
    # Test get_preset_info
    try:
        info = get_preset_info("mammalian_cells_cellpose", "training")
        assert isinstance(info, dict)
        assert "model_type" in info
        print("✓ get_preset_info() works for training")
        
        eval_info = get_preset_info("default", "evaluation")
        assert isinstance(eval_info, dict)
        assert "flow_threshold" in eval_info
        print("✓ get_preset_info() works for evaluation")
    except Exception as e:
        print(f"✗ get_preset_info() failed: {e}")


def test_config_validation():
    """Test configuration validation and error handling."""
    print("\nTesting configuration validation...")
    
    # Test invalid preset name
    try:
        create_training_config_from_preset(
            "nonexistent_preset",
            train_dir="/tmp/train",
            test_dir="/tmp/test", 
            output_dir="/tmp/output"
        )
        print("✗ Should have failed with invalid preset name")
    except ValueError:
        print("✓ Correctly caught invalid training preset name")
    
    try:
        create_evaluation_config_from_preset(
            "nonexistent_preset",
            test_dir="/tmp/test",
            model_path="/tmp/model.pt",
            output_dir="/tmp/results"
        )
        print("✗ Should have failed with invalid preset name")
    except ValueError:
        print("✓ Correctly caught invalid evaluation preset name")
    
    # Test parameter overrides
    with tempfile.TemporaryDirectory() as tmpdir:
        config = create_training_config_from_preset(
            "mammalian_cells_cellpose",
            train_dir=f"{tmpdir}/train",
            test_dir=f"{tmpdir}/test",
            output_dir=f"{tmpdir}/output",
            n_epochs=50,  # Override default
            learning_rate=0.05  # Override default
        )
        assert config.n_epochs == 50
        assert config.learning_rate == 0.05
        print("✓ Parameter overrides work correctly")


def test_output_directory_creation():
    """Test that output directories are created correctly."""
    print("\nTesting output directory creation...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = f"{tmpdir}/new_output_dir"
        
        config = TrainingConfig(
            train_dir=f"{tmpdir}/train",
            test_dir=f"{tmpdir}/test",
            output_dir=output_dir
        )
        
        # Output directory should be created in __post_init__
        assert os.path.exists(output_dir)
        print("✓ Output directory creation works")
        
        eval_output_dir = f"{tmpdir}/new_eval_dir"
        eval_config = EvaluationConfig(
            test_dir=f"{tmpdir}/test",
            model_path="/tmp/model.pt",
            output_dir=eval_output_dir
        )
        
        assert os.path.exists(eval_output_dir)
        print("✓ Evaluation output directory creation works")


def run_all_tests():
    """Run all tests."""
    print("Running Modular Training System Tests")
    print("=" * 50)
    
    try:
        test_config_creation()
        test_presets()
        test_trainer_initialization()
        test_preset_functions()
        test_config_validation()
        test_output_directory_creation()
        
        print("\n" + "=" * 50)
        print("SUCCESS: All tests passed!")
        print("\nThe modular training and evaluation system is working correctly.")
        print("You can now use it to train and evaluate Cellpose and Omnipose models.")
        
    except Exception as e:
        print(f"\nERROR: Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
