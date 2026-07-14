"""Tests for the cell_id-only metrics loader (no label/label_id aliases).

``cell_activity_labeler`` is a standalone sub-package whose ``__init__`` chain
uses top-level absolute imports and is not installed in the main ``.venv``.
``loaders.py`` depends only on pandas/pathlib, so it is loaded directly from its
file to avoid triggering the package import chain.
"""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

_LOADERS_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "cell_activity_labeler"
    / "io"
    / "loaders.py"
)
_spec = importlib.util.spec_from_file_location("_cal_loaders", _LOADERS_PATH)
loaders = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(loaders)


def _base_frame(id_column):
    return pd.DataFrame(
        {
            "image_path": ["a/foo_mCherry.tif", "a/foo_mCherry.tif"],
            id_column: [1, 2],
            "mean_intensity": [10.0, 20.0],
        }
    )


def test_cell_id_frame_normalizes():
    df = loaders.normalize_metrics_dataframe(_base_frame("cell_id"))
    assert "cell_id" in df.columns
    assert df["cell_id"].tolist() == [1, 2]
    # derived identity columns still filled from image_path
    assert "image" in df.columns
    assert "ID" in df.columns


@pytest.mark.parametrize("legacy_column", ["label_id", "label"])
def test_legacy_id_columns_rejected(legacy_column):
    df = _base_frame(legacy_column)
    with pytest.raises(ValueError, match="cell_id"):
        loaders.validate_metrics_input_dataframe(df)
    # error also points at the migration script
    with pytest.raises(ValueError, match="migrate_label_id_to_cell_id"):
        loaders.normalize_metrics_dataframe(df)


def test_missing_image_column_reported():
    df = pd.DataFrame({"cell_id": [1], "mean_intensity": [1.0]})
    with pytest.raises(ValueError, match="image"):
        loaders.validate_metrics_input_dataframe(df)
