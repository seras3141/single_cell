"""
Integration tests for the complete inference pipeline.
"""

import pytest
import numpy as np
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import tifffile
import yaml
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from src.inference import InferencePipeline, CellposePredictor, OutputManager
from src.inference.output_manager import load_labels

@pytest.fixture
def temp_dirs():
    temp_dir = tempfile.mkdtemp()
    test_dir = Path(temp_dir)
    input_dir = test_dir / "input"
    output_dir = test_dir / "output"
    config_dir = test_dir / "config"
    input_dir.mkdir()
    config_dir.mkdir()
    yield input_dir, output_dir, config_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def test_files(temp_dirs):
    input_dir, _, _ = temp_dirs
    test_files = []
    test_cases = [
        ("sample_001_BF.tif", (128, 128), np.uint8),
        ("sample_002_BF.tif", (256, 256), np.uint16),
        ("sample_003_BF.tif", (64, 64), np.uint8),
    ]
    for filename, shape, dtype in test_cases:
        file_path = input_dir / filename
        max_val = 255 if dtype == np.uint8 else 65535
        image = np.zeros(shape, dtype=dtype)
        center_y, center_x = shape[0] // 2, shape[1] // 2
        for i in range(3):
            cy = center_y + (i - 1) * 30
            cx = center_x + (i - 1) * 30
            if 0 <= cy < shape[0] and 0 <= cx < shape[1]:
                y, x = np.ogrid[:shape[0], :shape[1]]
                mask = (y - cy)**2 + (x - cx)**2 <= 15**2
                image[mask] = max_val // 2
        noise = np.random.randint(0, max_val // 10, shape, dtype=dtype)
        image = np.clip(image + noise, 0, max_val).astype(dtype)
        tifffile.imwrite(file_path, image)
        test_files.append(file_path)
    return test_files

@pytest.fixture
def test_config(temp_dirs):
    _, _, config_dir = temp_dirs
    config = {
        'segmentation': {
            'cellpose': {
                'model_type': 'cyto3',
                'gpu': True,
                'flow_threshold': 0.4,
                'cellprob_threshold': 0.0,
                'min_size': 30,
                'channels': [0, 0]
            },
            'inference': {
                'save_overlays': True,
                'save_metadata': True,
                'process_z_stacks': False,
                'file_pattern': '*_BF.tif'
            }
        },
    }
    config_file = config_dir / "test_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f)
    return config_file

@patch('src.inference.cellpose_predictor.models.CellposeModel')
def test_complete_pipeline_with_mocked_cellpose(mock_cellpose_model, temp_dirs, test_files, test_config):
    input_dir, output_dir, _ = temp_dirs
    mock_model = MagicMock()
    def mock_eval(image, diameter=None, channels=None, **kwargs):
        height, width = image.shape[:2]
        masks = np.zeros((height, width), dtype=np.uint16)
        num_cells = np.random.randint(1, 5)
        for i in range(num_cells):
            cy = np.random.randint(20, height - 20)
            cx = np.random.randint(20, width - 20)
            y, x = np.ogrid[:height, :width]
            mask = (y - cy)**2 + (x - cx)**2 <= 10**2
            masks[mask] = i + 1
        flows = [
            np.random.random((height, width, 3)),
            np.random.random((2, height, width)),
            np.random.random((height, width))
        ]
        styles = np.random.random(64)
        return masks, flows, styles
    mock_model.eval.side_effect = mock_eval
    mock_cellpose_model.return_value = mock_model
    pipeline = InferencePipeline.from_config(
        config_path=test_config,
        model_name="cyto3",
        output_dir=output_dir,
        dataset_name="test_integration"
    )
    results = pipeline.run_inference(
        input_dir=input_dir,
        file_pattern="*_BF.tif"
    )
    assert 'processed_files' in results
    assert len(results['processed_files']) == len(test_files)

def test_output_single_file_validation(temp_dirs, test_files):
    from tests.inference.test_base_predictor import MockPredictor
    predictor = MockPredictor(model_name="test_model")
    output_manager = OutputManager(
        base_output_dir=temp_dirs[1],
        model_name="test_model",
        dataset_name="validation_test"
    )
    pipeline = InferencePipeline(predictor, output_manager)
    test_file = test_files[0]
    result = pipeline.run_inference_single(
        test_file,
        save_overlays=False,
        save_metadata=True
    )
    assert result['status'] == 'success'
    expected_base = temp_dirs[1] / "test_model" / "validation_test"
    # OutputManager defaults to zarr format
    mask_file = expected_base / "masks" / test_file.name.replace("_BF.tif", "_masks.zarr")
    assert mask_file.exists(), "Mask file does not exist : {}".format(mask_file)

    masks = load_labels(mask_file)
    assert masks.dtype in (np.dtype("uint8"), np.dtype("uint16"), np.dtype("uint32"))
    assert masks.shape[0] > 0 and masks.shape[1] > 0

    metadata_file = expected_base / "metadata" / test_file.name.replace("_BF.tif", "_metadata.json")
    assert metadata_file.exists()
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    required_fields = ['num_cells', 'image_shape', 'input_file', 'saved_at']
    for field in required_fields:
        assert field in metadata

def test_output_batch_summary(temp_dirs, test_files):
    from tests.inference.test_base_predictor import MockPredictor
    predictor = MockPredictor(model_name="test_model")
    input_dir, output_dir, _ = temp_dirs

    output_manager = OutputManager(
        base_output_dir=output_dir,
        model_name="test_model",
        dataset_name="validation_test"
    )
    pipeline = InferencePipeline(predictor, output_manager)

    result = pipeline.run_inference(
        input_dir,
        file_pattern="*_BF.tif",
        save_overlays=False,
        save_metadata=True
    )

    expected_base = output_dir / "test_model" / "validation_test"
    summary_file = expected_base / "run_summary.json"
    assert summary_file.exists()
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    assert 'summary' in summary
    assert summary['summary']['total_files_processed'] == 3
    assert 'processed_files' in summary
    assert len(summary['processed_files']) == 3

def test_different_file_patterns(temp_dirs, test_files):
    import tifffile
    additional_files = [
        "data_001.tif",
        "experiment_BF_001.tiff",
        "sample.png"
    ]
    for filename in additional_files:
        file_path = temp_dirs[0] / filename
        test_image = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
        if filename.endswith(('.tif', '.tiff')):
            tifffile.imwrite(file_path, test_image)
        else:
            file_path.touch()
    from tests.inference.test_base_predictor import MockPredictor
    predictor = MockPredictor()
    output_manager = OutputManager(
        base_output_dir=temp_dirs[1],
        model_name="test_patterns",
        dataset_name="pattern_test"
    )
    pipeline = InferencePipeline(predictor, output_manager)
    test_cases = [
        ("*_BF.tif", 3),
        ("*.tif", 4),
        ("*.tiff", 1),
        ("data_*.tif", 1),
        ("*.png", 0),
    ]
    for pattern, expected_count in test_cases[:4]:
        results = pipeline.run_inference(
            input_dir=temp_dirs[0],
            file_pattern=pattern
        )
        assert results['total_files'] == expected_count

def test_memory_efficiency_large_batch(temp_dirs, test_files):
    import tifffile
    additional_files = []
    for i in range(10):
        filename = f"batch_test_{i:03d}_BF.tif"
        file_path = temp_dirs[0] / filename
        test_image = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
        tifffile.imwrite(file_path, test_image)
        additional_files.append(file_path)
    from tests.inference.test_base_predictor import MockPredictor
    predictor = MockPredictor()
    output_manager = OutputManager(
        base_output_dir=temp_dirs[1],
        model_name="batch_test",
        dataset_name="memory_test"
    )
    pipeline = InferencePipeline(predictor, output_manager)
    progress_calls = []
    def track_progress(current, total, file_path):
        progress_calls.append((current, total))
    results = pipeline.run_inference(
        input_dir=temp_dirs[0],
        file_pattern="*_BF.tif",
        progress_callback=track_progress
    )
    assert results['total_files'] == 13
    assert len(results['processed_files']) == 13
    assert len(progress_calls) == 13
    for i, (current, total) in enumerate(progress_calls):
        assert current == i + 1
        assert total == 13


if __name__ == "__main__":
    pytest.main([__file__])
