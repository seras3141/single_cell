"""
Unit tests for the CellposePredictor class.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
import tempfile
import shutil

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.inference.cellpose_predictor import CellposePredictor


class TestCellposePredictor:
    """Test cases for CellposePredictor."""
    
    @patch('src.inference.cellpose_predictor.models.CellposeModel')
    def test_initialization_with_defaults(self, mock_cellpose_model):
        """Test initialization with default parameters."""
        mock_model = MagicMock()
        mock_cellpose_model.return_value = mock_model
        
        predictor = CellposePredictor()
        
        assert predictor.model_name == "cellpose_cyto3"
        assert predictor.model_type == "cyto3"
        assert predictor.gpu is True
        mock_cellpose_model.assert_called_once()
    
    @patch('src.inference.cellpose_predictor.models.CellposeModel')
    def test_initialization_with_custom_params(self, mock_cellpose_model):
        """Test initialization with custom parameters."""
        mock_model = MagicMock()
        mock_cellpose_model.return_value = mock_model
        
        predictor = CellposePredictor(
            model_type="nuclei",
            gpu=False,
            flow_threshold=0.3,
            min_size=25
        )
        
        assert predictor.model_type == "nuclei"
        assert predictor.gpu is False
        assert predictor.flow_threshold == 0.3
        assert predictor.min_size == 25
    
    @patch('src.inference.cellpose_predictor.models.CellposeModel')
    def test_load_model_default(self, mock_cellpose_model):
        """Test loading default model."""
        mock_model = MagicMock()
        mock_cellpose_model.return_value = mock_model
        
        predictor = CellposePredictor()
        result = predictor.load_model()
        
        assert result is True
        mock_cellpose_model.assert_called()

    @patch('src.inference.cellpose_predictor.models.CellposeModel')
    def test_load_custom_model(self, mock_cellpose_model):
        """Test loading custom model from path."""
        mock_model = MagicMock()
        mock_cellpose_model.return_value = mock_model
        
        predictor = CellposePredictor()
        custom_path = "/path/to/custom/model"
        
        result = predictor.load_model(custom_path)
        
        assert result is True
        # Should reinitialize with custom model path
        assert mock_cellpose_model.call_count >= 1
    
    @patch('src.inference.cellpose_predictor.models.CellposeModel')
    def test_load_model_failure(self, mock_cellpose_model):
        """Test model loading failure handling."""
        mock_cellpose_model.side_effect = Exception("Model loading failed")
        
        predictor = CellposePredictor()
        result = predictor.load_model()
        
        assert result is False
    
    @patch('src.inference.cellpose_predictor.models.CellposeModel')
    def test_predict_single_image(self, mock_cellpose_model):
        """Test prediction on single image."""
        # Mock the Cellpose model
        mock_model = MagicMock()
        mock_masks = np.random.randint(0, 100, (256, 256), dtype=np.uint16)
        mock_flows = [
            np.random.random((256, 256, 3)),  # RGB flows
            np.random.random((2, 256, 256)),  # XY flows  
            np.random.random((256, 256))      # Probability
        ]
        mock_styles = np.random.random(64)
        
        mock_model.eval.return_value = (mock_masks, mock_flows, mock_styles)
        mock_cellpose_model.return_value = mock_model
        
        predictor = CellposePredictor()
        test_image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        
        masks, metadata = predictor.predict(test_image)
        
        assert masks.shape == test_image.shape
        assert 'num_cells' in metadata
        assert 'flows' in metadata
        assert 'styles' in metadata
        assert metadata['num_cells'] == np.max(mock_masks)
    
    @patch('src.inference.cellpose_predictor.models.CellposeModel')
    def test_predict_with_custom_params(self, mock_cellpose_model):
        """Test prediction with custom parameters."""
        mock_model = MagicMock()
        mock_masks = np.random.randint(0, 50, (128, 128), dtype=np.uint16)
        mock_flows = [np.random.random((128, 128, 3)), None, None]
        mock_styles = np.random.random(32)
        
        mock_model.eval.return_value = (mock_masks, mock_flows, mock_styles)
        mock_cellpose_model.return_value = mock_model
        
        predictor = CellposePredictor()
        test_image = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
        
        masks, metadata = predictor.predict(
            test_image,
            diameter=25,
            flow_threshold=0.2,
            min_size=15
        )
        
        # Check that eval was called with custom parameters
        mock_model.eval.assert_called_once()
        call_args = mock_model.eval.call_args
        assert 'diameter' in call_args[1] or call_args[0][1] == 25  # diameter parameter
        assert 'flow_threshold' in call_args[1] or any(arg == 0.2 for arg in call_args[0])
    
    @patch('src.inference.cellpose_predictor.models.CellposeModel')
    def test_predict_z_stack_2d(self, mock_cellpose_model):
        """Test Z-stack prediction with 2D processing."""
        mock_model = MagicMock()
        mock_cellpose_model.return_value = mock_model
        
        # Mock eval to return different masks for each slice
        def mock_eval(*args, **kwargs):
            image = args[0]
            if len(image.shape) == 2:
                masks = np.random.randint(0, 50, image.shape, dtype=np.uint16)
            else:
                masks = np.random.randint(0, 50, image.shape[:2], dtype=np.uint16)
            flows = [np.random.random((*masks.shape, 3)), None, None]
            styles = np.random.random(32)
            return masks, flows, styles
        
        mock_model.eval.side_effect = mock_eval
        
        predictor = CellposePredictor()
        z_stack = np.random.randint(0, 255, (3, 128, 128), dtype=np.uint8)
        
        masks, metadata = predictor.predict_z_stack(z_stack, process_2d=True)
        
        # Should return stacked masks and combined metadata
        assert masks.shape == (3, 128, 128)  # 3 slices stacked
        assert isinstance(metadata, dict)
        assert 'per_slice_metadata' in metadata
        assert len(metadata['per_slice_metadata']) == 3  # 3 slices
        assert metadata['processing_mode'] == '2d_per_slice'
        assert metadata['total_cells'] > 0
    
    @patch('src.inference.cellpose_predictor.models.CellposeModel')
    def test_get_model_info(self, mock_cellpose_model):
        """Test model info retrieval."""
        mock_model = MagicMock()
        mock_cellpose_model.return_value = mock_model
        
        predictor = CellposePredictor(
            model_type="cyto2",
            gpu=False,
            flow_threshold=0.5
        )
        
        info = predictor.get_model_info()
        
        assert info['model_name'] == "cellpose_cyto2"
        assert info['model_type'] == "cyto2"
        assert info['gpu_enabled'] is False
        assert 'default_parameters' in info
        assert info['default_parameters']['flow_threshold'] == 0.5
    
    @patch('src.inference.cellpose_predictor.models.CellposeModel')
    def test_channels_parameter_handling(self, mock_cellpose_model):
        """Test that channels parameter is handled correctly."""
        mock_model = MagicMock()
        mock_cellpose_model.return_value = mock_model
        
        predictor = CellposePredictor(channels=[1, 2])
        assert predictor.channels == [1, 2]
        
        info = predictor.get_model_info()
        assert info['default_parameters']['channels'] == [1, 2]
    
    @patch('src.inference.cellpose_predictor.models.CellposeModel')
    def test_gpu_detection(self, mock_cellpose_model):
        """Test GPU detection and usage."""
        mock_model = MagicMock()
        mock_cellpose_model.return_value = mock_model
        
        # Test GPU enabled
        predictor = CellposePredictor(gpu=True)
        assert predictor.gpu is True
        
        # Test GPU disabled
        predictor_no_gpu = CellposePredictor(gpu=False)
        assert predictor_no_gpu.gpu is False
    
    @patch('src.inference.cellpose_predictor.models.CellposeModel')
    def test_predict_empty_image(self, mock_cellpose_model):
        """Test prediction on empty/zero image."""
        mock_model = MagicMock()
        mock_model.eval.return_value = (
            np.zeros((64, 64), dtype=np.uint16),  # No cells found
            [np.zeros((64, 64, 3)), None, None],
            np.zeros(32)
        )
        mock_cellpose_model.return_value = mock_model
        
        predictor = CellposePredictor()
        empty_image = np.zeros((64, 64), dtype=np.uint8)
        
        masks, metadata = predictor.predict(empty_image)
        
        assert masks.shape == empty_image.shape
        assert metadata['num_cells'] == 0
    
    @patch('src.inference.cellpose_predictor.models.CellposeModel')
    def test_error_handling_in_predict(self, mock_cellpose_model):
        """Test error handling during prediction."""
        mock_model = MagicMock()
        mock_model.eval.side_effect = Exception("Prediction failed")
        mock_cellpose_model.return_value = mock_model
        
        predictor = CellposePredictor()
        test_image = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        
        with pytest.raises(Exception):
            predictor.predict(test_image)


if __name__ == "__main__":
    pytest.main([__file__])
