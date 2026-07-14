"""Tests for dataset splitting and copy utilities."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Iterator, TypedDict, cast

import pytest

from src.preprocessing.dataset_split import (
    copy_with_split_dict,
    get_groups_from_filenames,
    split_dataset_dict,
    split_dataset_list,
    train_test_split_directory,
)
from src.utils.file_utils import BF_IF_FileHandler

SplitMap = dict[str, list[str]]
SplitResult = tuple[SplitMap, SplitMap]


class MockDataDirs(TypedDict):
    """Container for temporary mock dataset paths and filenames."""

    mock_data_dir: Path
    data_dir: Path
    image_dir: Path
    mask_dir: Path
    mock_image_files: list[str]
    mock_mask_files: list[str]


def _group_ids_from_paths(file_paths: list[str], file_handler: BF_IF_FileHandler) -> set[str]:
    """Extract grouping IDs used for split integrity assertions."""
    return {
        file_handler.extract_unique_id(Path(path).name)
        for path in file_paths
        if path
    }


@pytest.fixture(scope="module")
def mock_data_dirs() -> Iterator[MockDataDirs]:
    """Create temporary mock image and mask dataset."""
    temp_root = Path(tempfile.mkdtemp())
    data_dir = temp_root / "Plate 2126 Test Data"
    image_dir = data_dir
    mask_dir = data_dir / "masks"

    data_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    mock_image_files = [
        "t1_J03_s1_w1_z1.tif",
        "t1_J03_s1_w1_z2.tif",
        "t1_J03_s1_w1_z3.tif",
        "t1_J03_s1_w2_z1.tif",
        "t1_J03_s1_w2_z2.tif",
        "t1_J03_s1_w2_z3.tif",
        "t1_J04_s1_w1_z1.tif",
        "t1_J04_s1_w1_z2.tif",
        "t1_J04_s1_w1_z3.tif",
        "t1_J04_s1_w2_z1.tif",
        "t1_J04_s1_w2_z2.tif",
        "t1_J04_s1_w2_z3.tif",
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
        "mock_data_dir": temp_root,
        "data_dir": data_dir,
        "image_dir": image_dir,
        "mask_dir": mask_dir,
        "mock_image_files": mock_image_files,
        "mock_mask_files": mock_mask_files,
    }

    shutil.rmtree(temp_root, ignore_errors=True)


@pytest.fixture
def temp_output_dir() -> Iterator[Path]:
    """Create temporary output directory."""
    temp_dir = Path(tempfile.mkdtemp())
    output_dir = temp_dir / "output"
    output_dir.mkdir(exist_ok=True)
    yield output_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_split_dataset_dict_group_integrity_and_copy(
    mock_data_dirs: MockDataDirs, temp_output_dir: Path
) -> None:
    """Split dictionary input and ensure group integrity between train and test."""
    image_files = [
        str(mock_data_dirs["image_dir"] / f)
        for f in mock_data_dirs["mock_image_files"]
        if "_w1_" in f
    ]
    mask_files = [str(mock_data_dirs["mask_dir"] / f) for f in mock_data_dirs["mock_mask_files"]]
    split_dir = temp_output_dir / "dict_split"
    file_handler = BF_IF_FileHandler()

    train_files, test_files = cast(
        SplitResult,
        split_dataset_dict(
            file_map={"image": image_files, "mask": mask_files},
            test_size=0.33,
            random_state=42,
            file_handler=file_handler,
            output_dir=split_dir,
        ),
    )

    assert len(train_files["image"]) > 0
    assert len(test_files["image"]) > 0
    assert len(train_files["mask"]) > 0
    assert len(test_files["mask"]) > 0

    train_groups = _group_ids_from_paths(train_files["image"], file_handler)
    test_groups = _group_ids_from_paths(test_files["image"], file_handler)
    assert train_groups
    assert test_groups
    assert train_groups.isdisjoint(test_groups)

    assert (split_dir / "train").exists()
    assert (split_dir / "test").exists()
    assert any((split_dir / "train").rglob("*.tif"))
    assert any((split_dir / "test").rglob("*.tif"))


def test_split_dataset_dict_no_split_all_train(
    mock_data_dirs: MockDataDirs, temp_output_dir: Path
) -> None:
    """Test test_size=0 behavior where all data should go to train output."""
    image_files = [
        str(mock_data_dirs["image_dir"] / f)
        for f in mock_data_dirs["mock_image_files"]
        if "_w1_" in f
    ]
    mask_files = [str(mock_data_dirs["mask_dir"] / f) for f in mock_data_dirs["mock_mask_files"]]
    output_dir = temp_output_dir / "dict_no_split"

    train_files, test_files = cast(
        SplitResult,
        split_dataset_dict(
            file_map={"image": image_files, "mask": mask_files},
            test_size=0,
            random_state=42,
            file_handler=BF_IF_FileHandler(),
            output_dir=output_dir,
        ),
    )

    assert len(train_files["image"]) == len(image_files)
    assert len(train_files["mask"]) == len(mask_files)
    assert test_files == {}

    copied_files = list(output_dir.rglob("*.tif"))
    assert len(copied_files) == len(image_files) + len(mask_files)
    assert not (output_dir / "train").exists()
    assert not (output_dir / "test").exists()


def test_split_dataset_dict_all_test(
    mock_data_dirs: MockDataDirs, temp_output_dir: Path
) -> None:
    """Test test_size=1 behavior where all data should go to test output."""
    image_files = [
        str(mock_data_dirs["image_dir"] / f)
        for f in mock_data_dirs["mock_image_files"]
        if "_w1_" in f
    ]
    mask_files = [str(mock_data_dirs["mask_dir"] / f) for f in mock_data_dirs["mock_mask_files"]]
    output_dir = temp_output_dir / "dict_all_test"

    train_files, test_files = cast(
        SplitResult,
        split_dataset_dict(
            file_map={"image": image_files, "mask": mask_files},
            test_size=1,
            random_state=42,
            file_handler=BF_IF_FileHandler(),
            output_dir=output_dir,
        ),
    )

    assert train_files == {}
    assert len(test_files["image"]) == len(image_files)
    assert len(test_files["mask"]) == len(mask_files)

    copied_files = list(output_dir.rglob("*.tif"))
    assert len(copied_files) == len(image_files) + len(mask_files)
    assert not (output_dir / "train").exists()
    assert not (output_dir / "test").exists()


def test_split_dataset_list(
    mock_data_dirs: MockDataDirs, temp_output_dir: Path
) -> None:
    """Test list-based API to ensure parity with dictionary-based splitting."""
    image_files = [
        str(mock_data_dirs["image_dir"] / f)
        for f in mock_data_dirs["mock_image_files"]
        if "_w1_" in f
    ]
    mask_files = [str(mock_data_dirs["mask_dir"] / f) for f in mock_data_dirs["mock_mask_files"]]

    file_handler = BF_IF_FileHandler()
    train_files, test_files = cast(
        SplitResult,
        split_dataset_list(
            file_list=image_files + mask_files,
            test_size=0.33,
            random_state=42,
            file_handler=file_handler,
            output_dir=temp_output_dir / "list_split",
        ),
    )

    assert len(train_files["image"]) > 0
    assert len(test_files["image"]) > 0
    assert len(train_files["mask"]) > 0
    assert len(test_files["mask"]) > 0

    train_groups = _group_ids_from_paths(train_files["image"], file_handler)
    test_groups = _group_ids_from_paths(test_files["image"], file_handler)
    assert train_groups.isdisjoint(test_groups)


def test_train_test_split_directory(
    mock_data_dirs: MockDataDirs, temp_output_dir: Path
) -> None:
    """Test directory-based splitting and persisted split metadata."""
    result = train_test_split_directory(
        data_dir=str(mock_data_dirs["data_dir"]),
        output_dir=str(temp_output_dir),
        test_size=0.33,
        random_state=42,
        file_handler=BF_IF_FileHandler(),
    )

    assert "train_files" in result
    assert "test_files" in result
    assert "train_images" not in result
    assert "test_images" not in result
    assert "image" in result["train_files"]
    assert "image" in result["test_files"]
    assert len(result["train_files"]["image"]) > 0
    assert len(result["test_files"]["image"]) > 0
    assert (temp_output_dir / "dataset_split.json").exists()


def test_copy_with_split_dict(mock_data_dirs: MockDataDirs, temp_output_dir: Path) -> None:
    """Test copy_with_split_dict copies files into train/test folders."""
    src_file = mock_data_dirs["image_dir"] / mock_data_dirs["mock_image_files"][0]
    mask_file = mock_data_dirs["mask_dir"] / mock_data_dirs["mock_mask_files"][0]

    train_file_tuple = {
        "images": [(str(src_file), "train_image.tif")],
        "masks": [(str(mask_file), "train_mask.tif")],
    }
    test_file_tuple = {
        "images": [(str(src_file), "test_image.tif")],
        "masks": [(str(mask_file), "test_mask.tif")],
    }

    split_output_dir = temp_output_dir / "split_dict"
    copy_with_split_dict(train_file_tuple, test_file_tuple, split_output_dir)

    assert (split_output_dir / "train" / "train_image.tif").exists()
    assert (split_output_dir / "train" / "train_mask.tif").exists()
    assert (split_output_dir / "test" / "test_image.tif").exists()
    assert (split_output_dir / "test" / "test_mask.tif").exists()


def test_copy_with_split_dict_filter_keys(
    mock_data_dirs: MockDataDirs, temp_output_dir: Path
) -> None:
    """Test copy_with_split_dict with file key filtering."""
    src_file = mock_data_dirs["image_dir"] / mock_data_dirs["mock_image_files"][0]
    mask_file = mock_data_dirs["mask_dir"] / mock_data_dirs["mock_mask_files"][0]

    train_file_tuple = {
        "images": [(str(src_file), "train_image.tif")],
        "masks": [(str(mask_file), "train_mask.tif")],
    }
    test_file_tuple = {
        "images": [(str(src_file), "test_image.tif")],
        "masks": [(str(mask_file), "test_mask.tif")],
    }

    split_output_dir = temp_output_dir / "split_dict_filter"
    copy_with_split_dict(
        train_file_tuple,
        test_file_tuple,
        split_output_dir,
        filter_file_keys=["images"],
    )

    assert (split_output_dir / "train" / "train_image.tif").exists()
    assert not (split_output_dir / "train" / "train_mask.tif").exists()
    assert (split_output_dir / "test" / "test_image.tif").exists()
    assert not (split_output_dir / "test" / "test_mask.tif").exists()


# ─── get_groups_from_filenames ────────────────────────────────────────────────

def test_get_groups_groups_by_unique_id() -> None:
    """Files with the same plate/well/time end up in the same group."""
    handler = BF_IF_FileHandler()
    tuples = [
        ("orig/t1_A01_s1_w1_z1.tif", "p2126_A01_t1_z1_w1.tif"),
        ("orig/t1_A01_s1_w1_z2.tif", "p2126_A01_t1_z2_w1.tif"),
        ("orig/t1_B02_s1_w1_z1.tif", "p2126_B02_t1_z1_w1.tif"),
    ]
    groups = get_groups_from_filenames(tuples, handler)

    assert "p2126_A01_t1" in groups
    assert len(groups["p2126_A01_t1"]) == 2
    assert "p2126_B02_t1" in groups
    assert len(groups["p2126_B02_t1"]) == 1


def test_get_groups_empty_input_returns_empty_dict() -> None:
    groups = get_groups_from_filenames([], BF_IF_FileHandler())
    assert groups == {}


def test_get_groups_preserves_original_src_path() -> None:
    handler = BF_IF_FileHandler()
    tuples = [("my/original/path.tif", "p2126_A01_t1_z1_w1.tif")]
    groups = get_groups_from_filenames(tuples, handler)
    src, _ = list(groups.values())[0][0]
    assert src == "my/original/path.tif"


# ─── split_dataset_dict / split_dataset_list — error cases ───────────────────

def test_split_dataset_dict_none_handler_raises() -> None:
    with pytest.raises(ValueError, match="file handler"):
        split_dataset_dict(
            file_map={"image": ["a.tif"]},
            file_handler=None,
        )


def test_split_dataset_list_none_handler_raises() -> None:
    with pytest.raises(ValueError, match="file handler"):
        split_dataset_list(file_list=["a.tif"], file_handler=None)


def test_split_dataset_dict_empty_map_all_train() -> None:
    """Empty file_map with test_size=0 should return two empty dicts without error."""
    train, test = split_dataset_dict({}, test_size=0, file_handler=BF_IF_FileHandler())
    assert train == {}
    assert test == {}


# ─── train_test_split_directory — JSON content ───────────────────────────────

def test_train_test_split_json_content_matches_result(
    mock_data_dirs: MockDataDirs, temp_output_dir: Path
) -> None:
    """The persisted JSON should be byte-for-byte equivalent to the returned dict."""
    out = temp_output_dir / "json_content"
    result = train_test_split_directory(
        data_dir=str(mock_data_dirs["data_dir"]),
        output_dir=str(out),
        test_size=0.33,
        random_state=42,
        file_handler=BF_IF_FileHandler(),
    )
    with open(out / "dataset_split.json") as f:
        loaded = json.load(f)

    assert loaded["train_files"] == result["train_files"]
    assert loaded["test_files"] == result["test_files"]


def test_train_test_split_json_image_lists_non_empty(
    mock_data_dirs: MockDataDirs, temp_output_dir: Path
) -> None:
    """JSON should record non-empty image lists for both splits."""
    out = temp_output_dir / "json_keys"
    train_test_split_directory(
        data_dir=str(mock_data_dirs["data_dir"]),
        output_dir=str(out),
        test_size=0.33,
        random_state=42,
        file_handler=BF_IF_FileHandler(),
    )
    with open(out / "dataset_split.json") as f:
        loaded = json.load(f)

    assert "image" in loaded["train_files"]
    assert "image" in loaded["test_files"]
    assert len(loaded["train_files"]["image"]) > 0
    assert len(loaded["test_files"]["image"]) > 0
