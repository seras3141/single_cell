import numpy as np
import pandas as pd
import pytest

from src.feature_to_mcherry.config import FeatureToMcherryConfig
from src.feature_to_mcherry.evaluation.nested_r2 import nested_r2, per_fold_r2
from src.feature_to_mcherry.evaluation.run_decomposition import (
    decompose_oof,
    rebuild_targets_and_metadata,
)


def test_a_timepoint_offset_counts_in_pooled_but_not_within_timepoint():
    well = np.array(["A"] * 4 + ["B"] * 4)
    timepoint = np.array(["1", "1", "2", "2"] * 2)
    truth = np.array([0.0, 1.0, 10.0, 11.0] * 2)
    pred = np.array([0.5, 0.5, 10.5, 10.5] * 2)
    levels = nested_r2(well, timepoint, truth, pred)
    assert levels["pooled"] > 0.99
    assert levels["within_timepoint"] < levels["pooled"]
    assert np.isclose(levels["within_timepoint"], 0.0)


def test_a_level_with_no_variance_is_nan():
    well = np.array(["A", "A", "B", "B"])
    timepoint = np.array(["1", "2", "1", "2"])
    truth = np.array([0.0, 5.0, 0.0, 5.0])
    levels = nested_r2(well, timepoint, truth, truth + 1.0)
    assert np.isnan(levels["within_timepoint"])
    assert np.isnan(levels["within_well_timepoint"])
    assert np.isfinite(levels["pooled"])


def test_per_fold_r2_scores_each_fold_on_its_own_wells():
    well = np.array(["A", "A", "B", "B"])
    truth = np.array([0.0, 2.0, 0.0, 2.0])
    pred = np.array([0.0, 2.0, 1.0, 1.0])
    table = per_fold_r2(well, truth, pred, {"A": 0, "B": 1})
    assert table.set_index("fold")["r2"].round(6).to_dict() == {0: 1.0, 1: 0.0}


def _run_tables():
    wells = ["A", "A", "B", "B"]
    metadata = pd.DataFrame({"sample_id": wells, "timepoint": ["1", "2", "1", "2"]})
    oof = pd.DataFrame(
        {
            "group_id": wells * 2,
            "target_name": ["t1"] * 4 + ["t2"] * 4,
            "y_true": [0.0, 2.0, 0.0, 2.0] * 2,
            "y_pred": [0.0, 2.0, 1.0, 1.0] * 2,
        }
    )
    folds = pd.DataFrame({"group_id": ["A", "B"], "fold": [0, 1]})
    y = np.column_stack([[0.0, 2.0, 0.0, 2.0]] * 2)
    return oof, folds, y, metadata


def test_decompose_oof_gives_fold_and_nested_rows_per_target():
    spread, nested = decompose_oof(*_run_tables())
    assert sorted(spread["target"].unique()) == ["t1", "t2"]
    assert len(spread) == 4
    assert set(nested["level"]) == {
        "pooled",
        "within_timepoint",
        "within_well_timepoint",
    }


def test_decompose_oof_rejects_a_different_row_count():
    oof, folds, y, metadata = _run_tables()
    with pytest.raises(ValueError, match="oof rows"):
        decompose_oof(oof, folds, y[:3], metadata.iloc[:3])


def test_decompose_oof_rejects_wells_in_a_different_order():
    oof, folds, y, metadata = _run_tables()
    metadata["sample_id"] = ["B", "B", "A", "A"]
    with pytest.raises(ValueError, match="do not match"):
        decompose_oof(oof, folds, y, metadata)


def test_decompose_oof_rejects_true_values_that_differ_from_the_rebuild():
    oof, folds, y, metadata = _run_tables()
    with pytest.raises(ValueError, match="y_true differs"):
        decompose_oof(oof, folds, y[::-1].copy(), metadata)


def test_a_well_without_a_fold_is_an_error():
    well = np.array(["A", "A", "C", "C"])
    truth = np.array([0.0, 2.0, 0.0, 2.0])
    with pytest.raises(ValueError, match="without a fold"):
        per_fold_r2(well, truth, truth, {"A": 0})


def test_a_run_grouped_by_timepoint_cannot_be_decomposed():
    config = FeatureToMcherryConfig(
        feature_csv="unused", target_csv="unused", group_by="timepoint"
    )
    with pytest.raises(ValueError, match="grouped by sample_id"):
        rebuild_targets_and_metadata(config)
