"""
Simple test for dataset split functionality.
"""
import tempfile
import shutil
from pathlib import Path
import pytest
from src.preprocessing.dataset_split import train_test_split_directory
from src.utils.file_utils import BF_IF_FileHandler


def _unique_ids_from_paths(file_paths: list[str], file_handler: BF_IF_FileHandler) -> set[str]:
    """Extract unique grouping IDs for split-integrity checks."""
    return {
        file_handler.extract_unique_id(Path(path).name)
        for path in file_paths
        if path
    }

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
    file_handler = BF_IF_FileHandler()
    result = train_test_split_directory(
        str(data_dir),
        str(output_dir),
        test_size=0.33,
        random_state=42,
        file_handler=file_handler,
    )
    assert "train_files" in result
    assert "test_files" in result
    assert "BF" in result["train_files"]
    assert "BF" in result["test_files"]
    assert "image" in result["train_files"]
    assert "image" in result["test_files"]
    assert "mask" in result["train_files"]
    assert "mask" in result["test_files"]

    train_total = sum(len(paths) for paths in result["train_files"].values())
    test_total = sum(len(paths) for paths in result["test_files"].values())
    assert train_total > 0, "Should have training files"
    assert test_total > 0, "Should have test files"

    for key in ("BF", "image", "mask"):
        train_groups = _unique_ids_from_paths(result["train_files"][key], file_handler)
        test_groups = _unique_ids_from_paths(result["test_files"][key], file_handler)
        overlap = train_groups & test_groups
        assert len(overlap) == 0, f"Groups should not overlap for {key}: {overlap}"


def test_split_with_unpaired_groups_is_supported(tmp_path):
    """Unpaired image/mask groups are currently supported and should not raise."""
    data_dir = tmp_path / "mock_data" / "Plate 2126"
    output_dir = tmp_path / "output"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    # Create image groups that do not correspond to mask groups.
    (data_dir / "t1_A01_s1_w1_z1.tif").write_text("mock image data")
    (data_dir / "t1_B02_s1_w1_z1.tif").write_text("mock image data")
    (data_dir / "Cells_R9-C9-F1-Z1-T1.tif").write_text("mock mask data")
    (data_dir / "Cells_R8-C8-F1-Z1-T1.tif").write_text("mock mask data")

    result = train_test_split_directory(
        str(data_dir),
        str(output_dir),
        test_size=0.33,
        random_state=42,
        file_handler=BF_IF_FileHandler(),
    )

    assert "train_files" in result
    assert "test_files" in result
    assert "BF" in result["train_files"]
    assert "mask" in result["train_files"]

    # All input files should appear in exactly one split for each file type.
    total_bf = len(result["train_files"]["BF"]) + len(result["test_files"]["BF"])
    total_mask = len(result["train_files"]["mask"]) + len(result["test_files"]["mask"])
    assert total_bf == 2
    assert total_mask == 2

def test_split_count_conservation(mock_data_dir):
    data_dir, output_dir = mock_data_dir
    result = train_test_split_directory(
        str(data_dir),
        str(output_dir),
        test_size=0.33,
        random_state=42,
        file_handler=BF_IF_FileHandler(),
    )

    train_bf = result["train_files"]["BF"]
    test_bf = result["test_files"]["BF"]
    train_image = result["train_files"]["image"]
    test_image = result["test_files"]["image"]
    train_mask = result["train_files"]["mask"]
    test_mask = result["test_files"]["mask"]

    # Fixture has 3 wells x 2 z-slices = 6 files per type.
    assert len(train_bf) + len(test_bf) == 6
    assert len(train_image) + len(test_image) == 6
    assert len(train_mask) + len(test_mask) == 6

    train_total = len(train_bf) + len(train_image) + len(train_mask)
    test_total = len(test_bf) + len(test_image) + len(test_mask)
    assert train_total > 0
    assert test_total > 0
