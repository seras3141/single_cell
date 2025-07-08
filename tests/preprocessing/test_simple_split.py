"""
Simple test for dataset split functionality.
"""
import os
import re
import tempfile
import shutil
from pathlib import Path
import pytest
from src.preprocessing.dataset_split import train_test_split_directory, get_image_from_pattern, get_mask_from_pattern
from src.utils.file_utils import DefaultFileHandler, BF_IF_FileHandler

@pytest.fixture
def mock_data_dir():
    temp_dir = tempfile.mkdtemp()
    data_dir = Path(temp_dir) / "mock_data" / "Plate 2126"
    output_dir = Path(temp_dir) / "output"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    mock_files = [
        "t1_A01_s1_w1_z1.tif",
        "t1_A01_s1_w1_z2.tif",
        "Cells_R1-C1-F1-Z1-T1.tif",
        "Cells_R1-C1-F1-Z2-T1.tif",
        "t1_B02_s1_w1_z1.tif",
        "t1_B02_s1_w1_z2.tif",
        "Cells_R2-C2-F1-Z1-T1.tif",
        "Cells_R2-C2-F1-Z2-T1.tif",
        "t1_C03_s1_w1_z1.tif",
        "t1_C03_s1_w1_z2.tif",
        "Cells_R3-C3-F1-Z1-T1.tif",
        "Cells_R3-C3-F1-Z2-T1.tif"
    ]
    for filename in mock_files:
        mock_file = data_dir / filename
        mock_file.write_text("mock image data")
    yield data_dir, output_dir
    shutil.rmtree(temp_dir, ignore_errors=True)

def test_split_keeps_groups_together(mock_data_dir):
    data_dir, output_dir = mock_data_dir
    print(f"Data directory: {data_dir}")
    result = train_test_split_directory(
        str(data_dir),
        str(output_dir),
        test_size=0.33,
        random_state=42,
        image_pattern="t1_*_s1_w1_z1.tif",
        mask_pattern="Cells_*-Z1-T1.tif",
        file_handler=BF_IF_FileHandler()
    )
    assert len(result['train_images']) > 0, "Should have training images"
    assert len(result['test_images']) > 0, "Should have test images"
    train_wells = set()
    test_wells = set()
    for img in result['train_images']:
        filename = Path(img).name
        match = re.search(r't1_([A-Z]\d+)_s1_w1_z1', filename)
        if match:
            train_wells.add(match.group(1))
    for img in result['test_images']:
        filename = Path(img).name
        match = re.search(r't1_([A-Z]\d+)_s1_w1_z1', filename)
        if match:
            test_wells.add(match.group(1))
    overlap = train_wells & test_wells
    assert len(overlap) == 0, f"Wells should not appear in both train and test splits. Found in both: {overlap}"
    assert len(train_wells) > 0, "Should have at least one training well"
    assert len(test_wells) > 0, "Should have at least one test well"

def test_get_image_from_no_pattern(mock_data_dir):
    data_dir, output_dir = mock_data_dir
    with pytest.raises(ValueError) as context:
        get_image_from_pattern(
            str(data_dir),
            pattern="nonexistent_*.tif",
        )
    assert "No images found matching pattern" in str(context.value)

def test_get_mask_from_no_pattern(mock_data_dir):
    data_dir, output_dir = mock_data_dir
    with pytest.raises(ValueError) as context:
        get_mask_from_pattern(
            str(data_dir),
            pattern="nonexistent_*.tif",
        )
    assert "No masks found matching pattern" in str(context.value)


def test_split_with_no_matching_files(tmp_path):
    # Create images and masks that cannot be paired by the handler
    data_dir = tmp_path / "mock_data" / "Plate 2126"
    output_dir = tmp_path / "output"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    # Only images, no masks
    (data_dir / "t1_A01_s1_w1_z1.tif").write_text("mock image data")
    (data_dir / "t1_B02_s1_w1_z1.tif").write_text("mock image data")
    # Only masks, no images
    (data_dir / "Cells_R9-C9-F1-Z1-T1.tif").write_text("mock mask data")
    (data_dir / "Cells_R8-C8-F1-Z1-T1.tif").write_text("mock mask data")
    with pytest.raises(ValueError) as context:
        train_test_split_directory(
            str(data_dir),
            str(output_dir),
            test_size=0.33,
            random_state=42,
            image_pattern="t1_*_s1_w1_z1.tif",
            mask_pattern="Cells_*-Z1-T1.tif",
            file_handler=BF_IF_FileHandler(),
        )
    assert "No matching image-mask groups found" in str(context.value)

def test_split_proportions(mock_data_dir):
    data_dir, output_dir = mock_data_dir
    result = train_test_split_directory(
        str(data_dir),
        str(output_dir),
        test_size=0.33,
        random_state=42,
        image_pattern="t1_*_s1_w1_z1.tif",
        mask_pattern="Cells_*-Z1-T1.tif",
        file_handler=BF_IF_FileHandler(),
    )
    total_images = len(result['train_images']) + len(result['test_images'])
    test_proportion = len(result['test_images']) / total_images
    assert abs(test_proportion - 0.33) < 0.1, f"Test proportion {test_proportion:.2f} should be close to 0.33"
    assert len(result['train_images']) == len(result['train_masks']), "Number of training images should match training masks"
    assert len(result['test_images']) == len(result['test_masks']), "Number of test images should match test masks"
