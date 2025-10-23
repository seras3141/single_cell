"""
Test dataset splitting functionality.

This module tests the dataset splitting functionality in the preprocessing module,
ensuring that images from the same key (e.g., J03) are kept together in either
the training or test split.
"""

import re
import tempfile
import shutil
from pathlib import Path
import pytest

from src.preprocessing.dataset_split import (
    split_dataset,
    train_test_split_directory,
    copy_file,
    copy_without_split,
    copy_with_split
)
from src.utils.file_utils import BF_IF_FileHandler, BF_FileHandler

@pytest.fixture(scope="module")
def mock_data_dirs():
    mock_data_dir = tempfile.mkdtemp()
    data_dir = Path(mock_data_dir) / "Plate 2126 Test Data"
    image_dir = data_dir
    mask_dir = data_dir / "masks"
    data_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    mock_image_files = [
        # J03 w1,2 z1-3
        "t1_J03_s1_w1_z1.tif",
        "t1_J03_s1_w1_z2.tif",
        "t1_J03_s1_w1_z3.tif",
        "t1_J03_s1_w2_z1.tif",
        "t1_J03_s1_w2_z2.tif",
        "t1_J03_s1_w2_z3.tif",
        # J04 w1,2 z1-3
        "t1_J04_s1_w1_z1.tif",
        "t1_J04_s1_w1_z2.tif",
        "t1_J04_s1_w1_z3.tif",
        "t1_J04_s1_w2_z1.tif",
        "t1_J04_s1_w2_z2.tif",
        "t1_J04_s1_w2_z3.tif",
        # L11 w1,2 z1-2
        "t1_L11_s1_w1_z1.tif",
        "t1_L11_s1_w1_z2.tif",
        "t1_L11_s1_w2_z1.tif",
        "t1_L11_s1_w2_z2.tif",
    ]
    mock_mask_files = [
        "Cells_R10-C3-F0-Z0-T0.tif",
        "Cells_R10-C3-F0-Z1-T0.tif",
        "Cells_R10-C3-F0-Z2-T0.tif",
        "Cells_R10-C4-F0-Z0-T0.tif",
        "Cells_R10-C4-F0-Z1-T0.tif",
        "Cells_R10-C4-F0-Z2-T0.tif",
        "Cells_R12-C11-F0-Z0-T0.tif",
        "Cells_R12-C11-F0-Z1-T0.tif",
    ]
    for filename in mock_image_files:
        (image_dir / filename).touch()
    for filename in mock_mask_files:
        (mask_dir / filename).touch()
    yield {
        "mock_data_dir": mock_data_dir,
        "data_dir": data_dir,
        "image_dir": image_dir,
        "mask_dir": mask_dir,
        "mock_image_files": mock_image_files,
        "mock_mask_files": mock_mask_files
    }
    shutil.rmtree(mock_data_dir, ignore_errors=True)


@pytest.fixture
def temp_output_dir():
    temp_dir = tempfile.mkdtemp()
    output_dir = Path(temp_dir) / "output"
    output_dir.mkdir(exist_ok=True)
    yield output_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_split_dataset(mock_data_dirs, temp_output_dir):

    image_dir = mock_data_dirs["image_dir"]
    mask_dir = mock_data_dirs["mask_dir"]
    mock_image_files = mock_data_dirs["mock_image_files"]
    mock_mask_files = mock_data_dirs["mock_mask_files"]
    image_files = [str(image_dir / f) for f in mock_image_files if 'w1' in f]
    mask_files = [str(mask_dir / f) for f in mock_mask_files]
    
    # Test without output directory
    train_images, train_masks, test_images, test_masks = split_dataset(
        image_files, mask_files, test_size=0.33, random_state=42,
        file_handler=BF_FileHandler()
    )
    assert len(train_images) > 0
    assert len(test_images) > 0
    assert len(train_masks) > 0
    assert len(test_masks) > 0
    train_groups = set()
    test_groups = set()
    for img in train_images:
        match = re.search(r'_([A-Za-z]+\d+)_', Path(img).name)
        if match:
            train_groups.add(match.group(1))
    for img in test_images:
        match = re.search(r'_([A-Za-z]+\d+)_', Path(img).name)
        if match:
            test_groups.add(match.group(1))
    assert len(train_groups & test_groups) == 0, f"Groups should not appear in both train and test splits. Found: {train_groups & test_groups}"

    # Test with output directory
    train_images, train_masks, test_images, test_masks = split_dataset(
        image_files, mask_files, test_size=0.33, random_state=42,
        file_handler=BF_IF_FileHandler(), output_dir=temp_output_dir
    )
    assert len(train_images) > 0
    assert len(test_images) > 0
    assert len(train_masks) > 0
    assert len(test_masks) > 0
    
    # Check if files were copied correctly
    train_dir = temp_output_dir / 'train'
    test_dir = temp_output_dir / 'test'
    assert train_dir.exists()
    assert test_dir.exists()
    assert any(train_dir.glob('*.tif'))
    assert any(test_dir.glob('*.tif'))
    
    # Test no-split case (test_size = 0)
    all_images, all_masks, no_images, no_masks = split_dataset(
        image_files, mask_files, test_size=0, random_state=42,
        file_handler=BF_IF_FileHandler(), output_dir=temp_output_dir / 'no_split'
    )
    assert len(all_images) == len(image_files)
    assert len(all_masks) == len(mask_files)
    assert len(no_images) == 0
    assert len(no_masks) == 0
    assert (temp_output_dir / 'no_split').exists()
    assert any((temp_output_dir / 'no_split').glob('*.tif'))


def test_split_dataset_train_only(mock_data_dirs, temp_output_dir):
    """Test dataset splitting with edge cases (test_size = 0 or 1)"""
    image_dir = mock_data_dirs["image_dir"]
    mask_dir = mock_data_dirs["mask_dir"]
    mock_image_files = [f for f in mock_data_dirs["mock_image_files"] if "_w1_" in f]
    mock_mask_files = [f for f in mock_data_dirs["mock_mask_files"] if "_w1_" in f]
    image_files = [str(image_dir / f) for f in mock_image_files]
    mask_files = [str(mask_dir / f) for f in mock_mask_files]

    train_images, train_masks, test_images, test_masks = split_dataset(
        image_files, mask_files, test_size=0, random_state=42,
        file_handler=BF_FileHandler(), output_dir=temp_output_dir / 'all_train'
    )

    assert len(train_images) == len(image_files), "All images should be in train set"
    assert len(test_images) == 0, "Test set should be empty"
    assert len(train_masks) == len(mask_files), "All masks should be in train set"
    assert len(test_masks) == 0, "Test mask set should be empty"

    assert (temp_output_dir / 'all_train').exists(), "Output directory should be created"
    assert any((temp_output_dir / 'all_train').glob('*.tif')), "Files should be copied"
    assert len(list((temp_output_dir / 'all_train').glob('*.tif'))) == len(image_files) + len(mask_files), "All files should be copied"
    assert not (temp_output_dir / 'all_train' / 'train').exists(), "No train subdirectory should be created"
    assert not (temp_output_dir / 'all_train' / 'test').exists(), "No test subdirectory should be created"

def test_split_dataset_test_only(mock_data_dirs, temp_output_dir):
    """Test dataset splitting with edge cases (test_size = 0 or 1)"""
    image_dir = mock_data_dirs["image_dir"]
    mask_dir = mock_data_dirs["mask_dir"]
    mock_image_files = [f for f in mock_data_dirs["mock_image_files"] if "_w1_" in f]
    mock_mask_files = [f for f in mock_data_dirs["mock_mask_files"] if "_w1_" in f]
    image_files = [str(image_dir / f) for f in mock_image_files]
    mask_files = [str(mask_dir / f) for f in mock_mask_files]

    # Test with output directory for test_size = 1
    train_images, train_masks, test_images, test_masks = split_dataset(
        image_files, mask_files, test_size=1, random_state=42,
        file_handler=BF_FileHandler(), output_dir=temp_output_dir / 'all_test'
    )

    assert len(train_images) == 0, "Train set should be empty"
    assert len(test_images) == len(image_files), "All images should be in test set"
    assert len(train_masks) == 0, "Train mask set should be empty"
    assert len(test_masks) == len(mask_files), "All masks should be in test set"

    assert (temp_output_dir / 'all_test').exists(), "Output directory should be created"
    assert any((temp_output_dir / 'all_test').glob('*.tif')), "Files should be copied"
    assert len(list((temp_output_dir / 'all_test').glob('*.tif'))) == len(image_files) + len(mask_files), "All files should be copied"
    assert not (temp_output_dir / 'all_test' / 'train').exists(), "No train subdirectory should be created"
    assert not (temp_output_dir / 'all_test' / 'test').exists(), "No test subdirectory should be created"

    # # Verify that groups are kept together
    # test_groups = set()
    # for img in test_images:
    #     match = re.search(r'_([A-Za-z]+\d+)_', Path(img).name)
    #     if match:
    #         test_groups.add(match.group(1))
    # all_groups = set()
    # for img in image_files:
    #     match = re.search(r'_([A-Za-z]+\d+)_', Path(img).name)
    #     if match:
    #         all_groups.add(match.group(1))
    # assert len(test_groups) == len(all_groups), "All groups should be in test set"


def test_train_test_split_directory(mock_data_dirs, temp_output_dir):
    data_dir = mock_data_dirs["data_dir"]
    output_dir = temp_output_dir

    result = train_test_split_directory(
        str(data_dir),
        str(output_dir),
        test_size=0.33,
        random_state=42,
        # image_pattern="t1_*_w1_*.tif",
        # mask_pattern="Cells_*.tif",
        file_handler=BF_FileHandler()
    )
    assert len(result["train_images"]) > 0
    assert len(result["test_images"]) > 0
    train_groups = set()
    test_groups = set()
    for img in result['train_images']:
        filename = Path(img).name
        match = re.search(r'_([A-Za-z]+\d+)_', filename)
        if match:
            train_groups.add(match.group(1))
    for img in result['test_images']:
        filename = Path(img).name
        match = re.search(r'_([A-Za-z]+\d+)_', filename)
        if match:
            test_groups.add(match.group(1))
    assert len(train_groups & test_groups) == 0, "Groups should not appear in both train and test splits"

def test_copy_functions(mock_data_dirs, temp_output_dir):
    # Test copy_file
    src_file = mock_data_dirs["image_dir"] / mock_data_dirs["mock_image_files"][0]
    dest_file = temp_output_dir / "test_copy.tif"
    copy_file(src_file, dest_file)
    assert dest_file.exists()
    
    # Test copy_without_split
    image_tuples = [
        (str(src_file), "renamed_image.tif"),
        (str(src_file), "renamed_image2.tif")
    ]
    mask_tuples = [
        (str(mock_data_dirs["mask_dir"] / mock_data_dirs["mock_mask_files"][0]), "renamed_mask.tif"),
    ]
    output_dir = temp_output_dir / "no_split"
    copy_without_split(image_tuples, mask_tuples, output_dir)
    assert (output_dir / "renamed_image.tif").exists()
    assert (output_dir / "renamed_image2.tif").exists()
    assert (output_dir / "renamed_mask.tif").exists()
    
    # Test copy_with_split
    train_image_tuples = [(str(src_file), "train_image.tif")]
    train_mask_tuples = [(str(mock_data_dirs["mask_dir"] / mock_data_dirs["mock_mask_files"][0]), "train_mask.tif")]
    test_image_tuples = [(str(src_file), "test_image.tif")]
    test_mask_tuples = [(str(mock_data_dirs["mask_dir"] / mock_data_dirs["mock_mask_files"][1]), "test_mask.tif")]
    
    split_output_dir = temp_output_dir / "split"
    copy_with_split(
        train_image_tuples, train_mask_tuples,
        test_image_tuples, test_mask_tuples,
        split_output_dir
    )
    assert (split_output_dir / "train" / "train_image.tif").exists()
    assert (split_output_dir / "train" / "train_mask.tif").exists()
    assert (split_output_dir / "test" / "test_image.tif").exists()
    assert (split_output_dir / "test" / "test_mask.tif").exists()
