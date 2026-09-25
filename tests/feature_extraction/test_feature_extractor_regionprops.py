"""Tests for ``src/feature_extraction/feature_extractor_regionprops.py``."""

import numpy as np
import pytest
import tifffile

from src.feature_extraction.feature_extractor_regionprops import (
    extract_regionprops_features,
    get_region_properties,
)

COLUMNS = {
    "cell_id",
    "area",
    "eccentricity",
    "mean_intensity",
    "max_intensity",
    "min_intensity",
}


def _mask2d() -> np.ndarray:
    mask = np.zeros((20, 20), dtype=np.uint16)
    mask[1:5, 1:5] = 1  # 16 px
    mask[10:16, 10:13] = 2  # 18 px
    return mask


def test_2d_columns_ids_and_area():
    mask = _mask2d()
    df = get_region_properties(mask, intensity_image=np.full(mask.shape, 7.0))
    assert set(df.columns) == COLUMNS
    assert list(df["cell_id"]) == [1, 2]
    assert list(df["area"]) == [16, 18]
    assert list(df["mean_intensity"]) == [7.0, 7.0]


def test_3d_gives_per_z_rows_with_z_stack():
    stack = np.stack([_mask2d(), np.zeros((20, 20), np.uint16), _mask2d()])
    df = get_region_properties(stack, intensity_image=np.ones(stack.shape))
    assert set(df.columns) == COLUMNS | {"z_stack"}
    assert list(df["z_stack"]) == [0, 0, 2, 2]


def test_3d_without_intensity_image_raises_clear_skimage_error():
    # mean/max/min_intensity need an intensity image; skimage says so.
    with pytest.raises(AttributeError):
        get_region_properties(np.stack([_mask2d()] * 2))


@pytest.mark.parametrize("shape", [(20,), (2, 2, 20, 20)])
def test_other_dimensionalities_raise(shape):
    with pytest.raises(ValueError, match="2D or 3D"):
        get_region_properties(np.zeros(shape, dtype=np.uint16))


def test_empty_mask_gives_empty_frame():
    mask = np.zeros((20, 20), dtype=np.uint16)
    df = get_region_properties(mask, intensity_image=np.ones(mask.shape))
    assert df.empty
    assert "cell_id" in df.columns


def test_extract_from_paths(tmp_path):
    bf, seg, out = tmp_path / "bf.tif", tmp_path / "seg.tif", tmp_path / "f.csv"
    tifffile.imwrite(bf, np.full((20, 20), 3, dtype=np.uint16))
    tifffile.imwrite(seg, _mask2d())
    df = extract_regionprops_features(bf, seg, output_csv_path=out)
    assert list(df["cell_id"]) == [1, 2]
    assert out.exists()
