"""
Test fixtures and utilities for the inference test suite.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path


@pytest.fixture
def temp_directory():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    if Path(temp_dir).exists():
        shutil.rmtree(temp_dir)


@pytest.fixture
def sample_images(temp_directory):
    """Create sample test images."""
    import tifffile
    
    images_dir = temp_directory / "images"
    images_dir.mkdir()
    
    image_files = []
    
    # Create different types of test images
    test_cases = [
        ("test_001_BF.tif", (64, 64), np.uint8),
        ("test_002_BF.tif", (128, 128), np.uint16),
        ("test_003_BF.tif", (32, 32), np.uint8),
    ]
    
    for filename, shape, dtype in test_cases:
        file_path = images_dir / filename
        
        # Create image with some structure
        image = create_structured_image(shape, dtype)
        tifffile.imwrite(file_path, image)
        image_files.append(file_path)
    
    return images_dir, image_files


@pytest.fixture
def sample_z_stack(temp_directory):
    """Create a sample Z-stack image."""
    import tifffile
    
    z_stack_path = temp_directory / "test_stack.tif"
    
    # Create 5-slice Z-stack
    z_stack = np.random.randint(0, 255, (5, 64, 64), dtype=np.uint8)
    
    # Add some structure to each slice
    for z in range(5):
        z_stack[z] = create_structured_image((64, 64), np.uint8)
    
    tifffile.imwrite(z_stack_path, z_stack)
    return z_stack_path


def create_structured_image(shape, dtype):
    """Create an image with some structure (not just noise)."""
    if dtype == np.uint8:
        max_val = 255
    else:
        max_val = 65535
    
    image = np.zeros(shape, dtype=dtype)
    center_y, center_x = shape[0] // 2, shape[1] // 2
    
    # Add circular objects
    num_objects = np.random.randint(1, 4)
    for i in range(num_objects):
        # Random position near center
        cy = center_y + np.random.randint(-shape[0]//4, shape[0]//4)
        cx = center_x + np.random.randint(-shape[1]//4, shape[1]//4)
        radius = np.random.randint(5, 15)
        
        if 0 <= cy < shape[0] and 0 <= cx < shape[1]:
            y, x = np.ogrid[:shape[0], :shape[1]]
            mask = (y - cy)**2 + (x - cx)**2 <= radius**2
            image[mask] = max_val // 2 + np.random.randint(-max_val//10, max_val//10)
    
    # Add background noise
    noise = np.random.randint(0, max_val // 20, shape, dtype=dtype)
    image = np.clip(image + noise, 0, max_val).astype(dtype)
    
    return image


@pytest.fixture
def test_config(temp_directory):
    """Create a test configuration file."""
    import yaml
    
    config = {
        'segmentation': {
            'cellpose': {
                'model_type': 'cyto3',
                'gpu': True,
                'flow_threshold': 0.4,
                'cellprob_threshold': 0.0,
                'min_size': 30,
                'channels': [0, 0],
                'normalize': True,
                'invert': False
            }
        },
        'inference': {
            'file_patterns': ['*_BF.tif'],
            'output': {
                'save_overlays': True,
                'save_metadata': True
            },
            'processing': {
                'process_z_stacks': False
            }
        },
        'paths': {
            'test_data': 'data/test',
            'output_root': 'results'
        }
    }
    
    config_file = temp_directory / "test_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f)
    
    return config_file


@pytest.fixture
def mock_cellpose_model():
    """Create a mock Cellpose model for testing."""
    from unittest.mock import MagicMock
    
    mock_model = MagicMock()
    
    def mock_eval(image, **kwargs):
        height, width = image.shape[:2]
        
        # Create masks with a few objects
        masks = np.zeros((height, width), dtype=np.uint16)
        num_cells = np.random.randint(0, 5)
        
        for i in range(num_cells):
            cy = np.random.randint(10, height - 10)
            cx = np.random.randint(10, width - 10)
            y, x = np.ogrid[:height, :width]
            mask = (y - cy)**2 + (x - cx)**2 <= 8**2
            masks[mask] = i + 1
        
        flows = [
            np.random.random((height, width, 3)),
            np.random.random((2, height, width)),
            np.random.random((height, width))
        ]
        styles = np.random.random(64)
        
        return masks, flows, styles
    
    mock_model.eval.side_effect = mock_eval
    return mock_model
