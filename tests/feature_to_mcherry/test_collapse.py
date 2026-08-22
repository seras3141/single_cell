"""Tests for the shared per-cell observation unit."""

import pandas as pd
import pytest

from src.feature_to_mcherry.data.collapse import (
    CELL_OBSERVATION_KEY,
    collapse_slices_to_cells,
)


def _slices() -> pd.DataFrame:
    """Two cells in one well at one timepoint, with unequal z-slice spans.

    Cell 1 spans 3 slices, cell 2 spans 1. Raw-row statistics would weight cell 1 three
    times as heavily as cell 2; that is the defect this module exists to prevent.
    """
    return pd.DataFrame(
        {
            "sample_id": ["A", "A", "A", "A"],
            "timepoint": ["1", "1", "1", "1"],
            "z_index": ["5", "6", "7", "5"],
            "cell_id": ["1", "1", "1", "2"],
            "value": [10.0, 20.0, 30.0, 100.0],
            "other": [1.0, 1.0, 4.0, 7.0],
        }
    )


def test_collapses_to_one_row_per_cell() -> None:
    out = collapse_slices_to_cells(_slices(), ["value"])
    assert len(out) == 2
    assert list(out.columns) == CELL_OBSERVATION_KEY + ["value"]
    by_cell = dict(zip(out["cell_id"], out["value"]))
    assert by_cell["1"] == pytest.approx(20.0)  # median(10, 20, 30)
    assert by_cell["2"] == pytest.approx(100.0)  # single slice, unchanged


def test_collapses_several_columns_at_once() -> None:
    out = collapse_slices_to_cells(_slices(), ["value", "other"])
    assert list(out.columns) == CELL_OBSERVATION_KEY + ["value", "other"]
    row = out[out["cell_id"] == "1"].iloc[0]
    assert row["value"] == pytest.approx(20.0)
    assert row["other"] == pytest.approx(1.0)  # median(1, 1, 4)


def test_z_index_is_dropped_not_carried_through() -> None:
    """Keeping a slice-level column on a per-cell frame invites the same confusion."""
    out = collapse_slices_to_cells(_slices(), ["value"])
    assert "z_index" not in out.columns


def test_same_cell_id_in_different_wells_stays_separate() -> None:
    """cell_id is only unique within a well, so the well must be part of the key."""
    frame = pd.DataFrame(
        {
            "sample_id": ["A", "B"],
            "timepoint": ["1", "1"],
            "cell_id": ["1", "1"],
            "value": [10.0, 90.0],
        }
    )
    out = collapse_slices_to_cells(frame, ["value"])
    assert len(out) == 2
    assert sorted(out["value"]) == [10.0, 90.0]


def test_same_cell_at_different_timepoints_stays_separate() -> None:
    frame = pd.DataFrame(
        {
            "sample_id": ["A", "A"],
            "timepoint": ["1", "11"],
            "cell_id": ["1", "1"],
            "value": [10.0, 90.0],
        }
    )
    assert len(collapse_slices_to_cells(frame, ["value"])) == 2


def test_already_per_cell_input_is_left_alone() -> None:
    frame = pd.DataFrame(
        {
            "sample_id": ["A", "A"],
            "timepoint": ["1", "11"],
            "cell_id": ["1", "2"],
            "value": [1.0, 2.0],
        }
    )
    out = collapse_slices_to_cells(frame, ["value"])
    assert len(out) == 2
    assert sorted(out["value"]) == [1.0, 2.0]


def test_nan_slices_are_ignored_rather_than_poisoning_the_cell() -> None:
    """A missing measurement on one slice must not discard the whole cell."""
    frame = pd.DataFrame(
        {
            "sample_id": ["A", "A", "A"],
            "timepoint": ["1", "1", "1"],
            "cell_id": ["1", "1", "1"],
            "value": [10.0, float("nan"), 30.0],
        }
    )
    out = collapse_slices_to_cells(frame, ["value"])
    assert len(out) == 1
    assert out.iloc[0]["value"] == pytest.approx(20.0)  # median(10, 30)


def test_missing_cell_id_raises_rather_than_falling_back() -> None:
    frame = pd.DataFrame({"sample_id": ["A"], "timepoint": ["1"], "value": [1.0]})
    with pytest.raises(ValueError, match="cell_id"):
        collapse_slices_to_cells(frame, ["value"])


def test_missing_value_column_raises() -> None:
    with pytest.raises(ValueError, match="absent_feature"):
        collapse_slices_to_cells(_slices(), ["absent_feature"])


def test_empty_value_columns_raises() -> None:
    with pytest.raises(ValueError, match="nothing to collapse"):
        collapse_slices_to_cells(_slices(), [])


def test_dataset_design_wrapper_delegates_to_the_same_definition() -> None:
    """Step 7's single-column wrapper must not be a second implementation."""
    from src.feature_to_mcherry.dataset_design import (
        collapse_slices_to_cells as wrapper,
    )

    frame = _slices()
    assert wrapper(frame, "value").equals(collapse_slices_to_cells(frame, ["value"]))


def test_mixed_key_dtypes_still_collapse_to_one_cell() -> None:
    """ "1" and 1 describe the same cell and must not group as two.

    Found by adversarial review. Without the str cast the frame below returns two rows
    -- silently leaving a slice-inflated observation behind, which is the exact defect
    this module exists to prevent. `contract.normalize_cell_key` guards the join path
    for the same reason.
    """
    frame = pd.DataFrame(
        {
            "sample_id": ["A", "A"],
            "timepoint": ["1", 1],  # same timepoint, different dtype
            "cell_id": ["7", 7],
            "value": [1.0, 9.0],
        }
    )
    out = collapse_slices_to_cells(frame, ["value"])
    assert len(out) == 1
    assert out.iloc[0]["value"] == pytest.approx(5.0)  # median(1, 9)


def test_all_nan_cell_yields_nan_not_a_dropped_row() -> None:
    """A cell whose every slice is NaN stays as one row with a NaN value.

    Dropping it would silently change the denominator of any downstream count.
    """
    frame = pd.DataFrame(
        {
            "sample_id": ["A", "A", "A"],
            "timepoint": ["1", "1", "1"],
            "cell_id": ["1", "1", "2"],
            "value": [float("nan"), float("nan"), 5.0],
        }
    )
    out = collapse_slices_to_cells(frame, ["value"])
    assert len(out) == 2
    assert pd.isna(out[out["cell_id"] == "1"].iloc[0]["value"])
