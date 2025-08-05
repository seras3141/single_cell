"""
Unit tests for the BasePredictor abstract base class and common predictor functionality.
"""

import pytest
import numpy as np
from abc import ABC
from pathlib import Path
from unittest.mock import MagicMock
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from src.inference.base_predictor import BasePredictor

class MockPredictor(BasePredictor):
    """Mock implementation of BasePredictor for testing."""
    
    def __init__(self, model_name="mock_model", **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        self.model = MagicMock()
        self._is_loaded = True
        
        # Store additional kwargs as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def load_model(self, model_path=None):
        """Mock model loading."""
        self._is_loaded = True
        return True
    
    def predict(self, image, **kwargs):
        """Mock prediction returning fake masks."""
        height, width = image.shape[:2]
        masks = np.random.randint(0, 100, (height, width), dtype=np.uint16)
        metadata = {
            'num_cells': np.max(masks),
            'image_shape': image.shape,
            'prediction_params': kwargs
        }
        return masks, metadata

    def predict_3d(self, z_stack, do_2d=True, **kwargs):
        """Mock 3D prediction."""
        if do_2d:
            results = []
            for i, slice_img in enumerate(z_stack):
                masks, metadata = self.predict(slice_img, **kwargs)
                metadata['slice_index'] = i
                results.append((masks, metadata))
            return results
        else:
            # 3D prediction
            masks = np.random.randint(0, 50, z_stack.shape, dtype=np.uint16)
            metadata = {
                'num_cells': np.max(masks),
                'stack_shape': z_stack.shape,
                'processing_mode': '3d'
            }
            return masks, metadata
    
    def get_model_info(self):
        """Mock model info."""
        return {
            'model_name': self.model_name,
            'status': 'loaded' if self._is_loaded else 'not_loaded',
            'gpu_enabled': getattr(self, 'gpu', False),
            'default_parameters': getattr(self, 'default_params', {})
        }

def test_abstract_class_cannot_be_instantiated():
    """Test that BasePredictor cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BasePredictor(model_name="test") # type: ignore

def test_mock_predictor_initialization():
    """Test that mock predictor initializes correctly."""
    predictor = MockPredictor(model_name="test_model")
    assert predictor.model_name == "test_model"
    assert predictor._is_loaded is True

def test_mock_predictor_with_kwargs():
    """Test mock predictor initialization with additional kwargs."""
    predictor = MockPredictor(
        model_name="test_model",
        gpu=True,
        flow_threshold=0.5
    )
    assert predictor.model_name == "test_model"
    assert hasattr(predictor, 'gpu')
    assert hasattr(predictor, 'flow_threshold')

def test_load_model():
    """Test model loading functionality."""
    predictor = MockPredictor()
    predictor._is_loaded = False
    
    result = predictor.load_model("fake/path")
    assert result is True
    assert predictor._is_loaded is True

def test_predict_single_image():
    """Test prediction on a single image."""
    predictor = MockPredictor()
    test_image = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
    
    masks, metadata = predictor.predict(test_image)
    
    assert masks.shape == test_image.shape
    assert masks.dtype == np.uint16
    assert 'num_cells' in metadata
    assert 'image_shape' in metadata
    assert metadata['image_shape'] == test_image.shape

def test_predict_with_parameters():
    """Test prediction with custom parameters."""
    predictor = MockPredictor()
    test_image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
    
    masks, metadata = predictor.predict(
        test_image,
        flow_threshold=0.3,
        min_size=20
    )
    
    assert 'prediction_params' in metadata
    assert metadata['prediction_params']['flow_threshold'] == 0.3
    assert metadata['prediction_params']['min_size'] == 20

def test_predict_3d_2d_processing():
    """Test 3D prediction with 2D slice processing."""
    predictor = MockPredictor()
    z_stack = np.random.randint(0, 255, (5, 256, 256), dtype=np.uint8)

    results = predictor.predict_3d(z_stack, do_2d=True)

    assert len(results) == 5  # 5 slices
    for i, (masks, metadata) in enumerate(results):
        assert masks.shape == (256, 256)
        assert metadata['slice_index'] == i
        assert 'num_cells' in metadata

def test_predict_3d_3d_processing():
    """Test 3D prediction with 3D processing."""
    predictor = MockPredictor()
    z_stack = np.random.randint(0, 255, (5, 256, 256), dtype=np.uint8)

    masks, metadata = predictor.predict_3d(z_stack, do_2d=False)
    
    assert masks.shape == z_stack.shape
    assert metadata['processing_mode'] == '3d'
    assert metadata['stack_shape'] == z_stack.shape

def test_get_model_info():
    """Test model info retrieval."""
    predictor = MockPredictor(model_name="test_model")
    
    info = predictor.get_model_info()
    
    assert info['model_name'] == "test_model"
    assert info['status'] == 'loaded'
    assert 'gpu_enabled' in info
    assert 'default_parameters' in info

def test_model_info_when_not_loaded():
    """Test model info when model is not loaded."""
    predictor = MockPredictor()
    predictor._is_loaded = False
    
    info = predictor.get_model_info()
    assert info['status'] == 'not_loaded'


if __name__ == "__main__":
    pytest.main([__file__])
