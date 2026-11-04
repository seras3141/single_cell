"""Tests for feature_to_mcherry.evaluation.well_timepoint."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.feature_to_mcherry.evaluation.well_timepoint import (
    first_low_count_timepoints,
    well_timepoint_metrics,
)


def _metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["A", "A", "A", "A", "B", "B"],
            "timepoint": ["2", "2", "10", "10", "2", "2"],
            "cell_id": ["1", "1", "2", "3", "4", "5"],
        }
    )


def test_metrics_match_hand_computed_values() -> None:
    y = np.array([[1.0], [3.0], [2.0], [6.0], [5.0], [5.0]])
    pred = np.array([[2.0], [2.0], [2.0], [4.0], [4.0], [6.0]])

    table = well_timepoint_metrics(_metadata(), y, pred, ["t"], "ridge")

    cell = table[(table["sample_id"] == "A") & (table["timepoint"] == "2")].iloc[0]
    assert cell["n_rows"] == 2
    assert cell["n_cells"] == 1
    assert cell["mae"] == 1.0
    assert math.isclose(cell["r2"], 1 - 2.0 / 2.0)
    other = table[(table["sample_id"] == "A") & (table["timepoint"] == "10")].iloc[0]
    assert math.isclose(other["mae"], 1.0)
    assert math.isclose(other["r2"], 1 - 4.0 / 8.0)


def test_r2_is_nan_for_a_constant_target_or_a_single_cell() -> None:
    y = np.array([[1.0], [3.0], [2.0], [6.0], [5.0], [5.0]])

    table = well_timepoint_metrics(_metadata(), y, y + 1, ["t"], "ridge")

    constant = table[(table["sample_id"] == "B")].iloc[0]
    assert math.isnan(constant["r2"])
    assert constant["mae"] == 1.0
    single = well_timepoint_metrics(
        _metadata().iloc[:1], y[:1], y[:1], ["t"], "ridge"
    ).iloc[0]
    assert math.isnan(single["r2"])


def test_timepoints_sort_numerically_and_models_are_labelled() -> None:
    y = np.arange(12, dtype=float).reshape(6, 2)

    table = well_timepoint_metrics(_metadata(), y, y, ["p75", "p90"], "linear_quantile")

    assert set(table["model"]) == {"linear_quantile"}
    assert set(table["target"]) == {"p75", "p90"}
    first_well = table[(table["sample_id"] == "A") & (table["target"] == "p75")]
    assert first_well["timepoint"].tolist() == ["2", "10"]


def _targets(counts: dict) -> pd.DataFrame:
    rows = []
    for well, per_timepoint in counts.items():
        for timepoint, n_cells in per_timepoint.items():
            rows += [
                {"sample_id": well, "timepoint": int(timepoint), "cell_id": i}
                for i in range(n_cells)
            ]
    return pd.DataFrame(rows)


def test_first_low_count_timepoint_uses_the_relative_drop_and_the_floor() -> None:
    targets = _targets(
        {
            "N11": {"1": 100, "2": 100, "10": 100, "11": 100},
            "W1": {"1": 100, "2": 90, "10": 8, "11": 5},
            "W2": {"1": 60, "2": 55, "10": 50, "11": 29},
            "W3": {"1": 60, "2": 58, "10": 59, "11": 57},
        }
    )

    first = first_low_count_timepoints(targets, "N11").set_index("sample_id")

    assert first.loc["W1", "first_low_count_timepoint"] == "10"
    assert first.loc["W2", "first_low_count_timepoint"] == "11"
    assert pd.isna(first.loc["W3", "first_low_count_timepoint"])
    assert pd.isna(first.loc["N11", "first_low_count_timepoint"])


def test_a_degraded_dmso_reference_does_not_mark_the_other_wells() -> None:
    targets = _targets(
        {
            "N11": {"1": 100, "2": 100, "10": 5},
            "W1": {"1": 60, "2": 58, "10": 59},
        }
    )

    first = first_low_count_timepoints(targets, "N11").set_index("sample_id")

    assert first.loc["N11", "first_low_count_timepoint"] == "10"
    assert pd.isna(first.loc["W1", "first_low_count_timepoint"])


def test_empty_input_gives_an_empty_table_with_the_columns() -> None:
    table = well_timepoint_metrics(
        _metadata().iloc[0:0], np.empty((0, 1)), np.empty((0, 1)), ["t"], "ridge"
    )

    assert table.empty
    assert {"n_rows", "n_cells", "r2", "mae"} <= set(table.columns)


def test_low_count_timepoints_are_strings_like_the_metrics_labels() -> None:
    targets = _targets(
        {"N11": {"1": 100, "2": 100, "10": 100}, "W1": {"1": 100, "2": 5, "10": 5}}
    )

    first = first_low_count_timepoints(targets, "N11").set_index("sample_id")

    assert first.loc["W1", "first_low_count_timepoint"] == "2"
    assert first["first_low_count_timepoint"].map(type).tolist() == [type(None), str]
