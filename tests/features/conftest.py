"""
Test fixtures for feature extraction tests.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
import tifffile


@pytest.fixture
def temp_test_directory():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp(prefix="feature_test_")
    yield Path(temp_dir)
    if Path(temp_dir).exists():
        shutil.rmtree(temp_dir)


@pytest.fixture
def sample_image_mask_pair():
    """Create a simple image-mask pair for testing."""
    # Create a simple 32x32 image with some structure
    image = np.zeros((32, 32), dtype=np.uint8)
    
    # Add background noise
    image += np.random.randint(20, 50, (32, 32), dtype=np.uint8)
    
    # Add some brighter regions representing cells
    image[10:20, 10:20] += 100  # Cell region 1
    image[5:15, 20:30] += 80   # Cell region 2
    
    # Create corresponding labeled mask
    mask = np.zeros((32, 32), dtype=np.uint16)
    mask[10:20, 10:20] = 1  # Instance 1
    mask[5:15, 20:30] = 2   # Instance 2
    
    return image, mask


@pytest.fixture  
def realistic_segmentation_data():
    """Create realistic segmentation data with multiple cell-like instances."""
    np.random.seed(42)  # For reproducible tests
    
    # Create 64x64 image
    image = np.zeros((64, 64), dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint16)
    
    # Add background
    image += np.random.randint(30, 70, (64, 64), dtype=np.uint8)
    
    # Create cell-like instances
    cell_params = [
        {'center': (15, 15), 'radius': 6, 'intensity': 150, 'id': 1},
        {'center': (15, 45), 'radius': 8, 'intensity': 120, 'id': 2}, 
        {'center': (45, 15), 'radius': 5, 'intensity': 180, 'id': 3},
        {'center': (45, 45), 'radius': 7, 'intensity': 140, 'id': 4},
    ]
    
    for params in cell_params:
        cy, cx = params['center']
        radius = params['radius'] 
        intensity = params['intensity']
        cell_id = params['id']
        
        # Create circular cell region
        y, x = np.ogrid[:64, :64]
        cell_mask = (x - cx)**2 + (y - cy)**2 <= radius**2
        
        # Add to mask
        mask[cell_mask] = cell_id
        
        # Add to image with some variation
        cell_intensities = np.random.normal(intensity, 20, cell_mask.sum())
        cell_intensities = np.clip(cell_intensities, 0, 255)
        image[cell_mask] = cell_intensities.astype(np.uint8)
    
    return image, mask, cell_params


@pytest.fixture
def test_file_pairs(temp_test_directory):
    """Create test image-mask file pairs on disk."""
    input_dir = temp_test_directory / 'test_input'
    input_dir.mkdir(parents=True)
    
    file_pairs = []
    
    # Create multiple test file pairs with different naming conventions
    test_configs = [
        {'image_name': 'sample_001_BF.tif', 'mask_name': 'sample_001_Cells.tif'},
        {'image_name': 'sample_002_BF.tif', 'mask_name': 'sample_002_Cells.tif'},
        {'image_name': 'image_test1.tif', 'mask_name': 'mask_test1.tif'},
    ]
    
    np.random.seed(123)  # For consistent test data
    
    for i, config in enumerate(test_configs):
        # Create synthetic image
        image = np.random.randint(50, 200, (48, 48), dtype=np.uint8)
        
        # Create synthetic mask with 2-4 instances
        mask = np.zeros((48, 48), dtype=np.uint16)
        num_instances = np.random.randint(2, 5)
        
        for j in range(num_instances):
            # Random cell position and size
            cx, cy = np.random.randint(8, 40, 2)
            radius = np.random.randint(3, 8)
            
            # Create cell
            y, x = np.ogrid[:48, :48]
            cell_mask = (x - cx)**2 + (y - cy)**2 <= radius**2
            mask[cell_mask] = j + 1
        
        # Save files
        image_path = input_dir / config['image_name']
        mask_path = input_dir / config['mask_name']
        
        tifffile.imwrite(image_path, image)
        tifffile.imwrite(mask_path, mask)
        
        file_pairs.append({
            'image_path': image_path,
            'mask_path': mask_path,
            'expected_instances': num_instances
        })
    
    return input_dir, file_pairs


@pytest.fixture
def feature_extraction_config(temp_test_directory):
    """Create a test configuration for feature extraction."""
    config = {
        'paths': {
            'input_dir': str(temp_test_directory / 'input'),
            'output_dir': str(temp_test_directory / 'output')
        },
        'file_patterns': {
            'images': ['*_BF.tif', 'image_*.tif', '*.tif'],
            'masks': ['*_Cells.tif', 'mask_*.tif', '*_mask.tif']
        },
        'feature_extraction': {
            'n_jobs': 1,  # Single-threaded for deterministic testing
            'batch_size': 5,
            'preprocessing': {
                'normalize_intensity': False,
                'clip_percentiles': None
            },
            'processing': {
                'validation': {
                    'min_instances_per_image': 1,
                    'max_instances_per_image': 500,
                    'check_image_mask_dimensions': True
                }
            }
        },
        'output': {
            'save_individual_files': True,
            'save_combined_file': True,
            'include_metadata': True,
            'metadata_fields': [
                'image_filename', 'mask_filename', 'processing_timestamp',
                'feature_extraction_version', 'dataset_name'
            ],
            'combined_filename': 'all_features.csv',
            'individual_format': '{image_name}_features.csv',
            'create_subdirs': False
        },
        'logging': {
            'level': 'DEBUG',
            'filename': 'test_feature_extraction.log'
        }
    }
    return config


@pytest.fixture
def edge_case_data():
    """Create edge case data for testing robustness."""
    test_cases = {}
    
    # Single pixel instances
    mask_single = np.zeros((10, 10), dtype=np.uint16)
    mask_single[2, 2] = 1
    mask_single[5, 5] = 2  
    mask_single[8, 8] = 3
    image_single = np.random.randint(50, 200, (10, 10), dtype=np.uint8)
    test_cases['single_pixels'] = (image_single, mask_single)
    
    # Very elongated instance
    mask_elongated = np.zeros((20, 60), dtype=np.uint16)
    mask_elongated[9:11, 5:55] = 1  # Very thin rectangle
    image_elongated = np.random.randint(50, 200, (20, 60), dtype=np.uint8)
    test_cases['elongated'] = (image_elongated, mask_elongated)
    
    # Edge-touching instances
    mask_edge = np.zeros((20, 20), dtype=np.uint16)
    mask_edge[0:5, 0:5] = 1    # Top-left corner
    mask_edge[15:20, 15:20] = 2  # Bottom-right corner
    mask_edge[0:5, 15:20] = 3   # Top-right corner
    image_edge = np.random.randint(50, 200, (20, 20), dtype=np.uint8)
    test_cases['edge_touching'] = (image_edge, mask_edge)
    
    # Large instance
    mask_large = np.zeros((100, 100), dtype=np.uint16)
    y, x = np.ogrid[:100, :100]
    large_circle = (x - 50)**2 + (y - 50)**2 <= 40**2
    mask_large[large_circle] = 1
    image_large = np.random.randint(50, 200, (100, 100), dtype=np.uint8)
    test_cases['large_instance'] = (image_large, mask_large)
    
    return test_cases


@pytest.fixture
def performance_test_data():
    """Create larger dataset for performance testing."""
    np.random.seed(999)  # Fixed seed for reproducible performance tests
    
    # Create multiple images with many instances each
    datasets = []
    
    for img_idx in range(5):  # 5 images
        image = np.random.randint(40, 220, (256, 256), dtype=np.uint8)
        mask = np.zeros((256, 256), dtype=np.uint16)
        
        # Add many small instances
        instance_id = 1
        for _ in range(50):  # 50 instances per image
            # Random position and size
            cx = np.random.randint(15, 241)
            cy = np.random.randint(15, 241)
            radius = np.random.randint(3, 12)
            
            # Create instance
            y, x = np.ogrid[:256, :256]
            instance_mask = (x - cx)**2 + (y - cy)**2 <= radius**2
            
            # Check for overlap and add if minimal
            if np.sum(mask[instance_mask] > 0) < instance_mask.sum() * 0.1:
                mask[instance_mask] = instance_id
                instance_id += 1
        
        datasets.append({
            'image': image,
            'mask': mask,
            'num_instances': instance_id - 1
        })
    
    return datasets
