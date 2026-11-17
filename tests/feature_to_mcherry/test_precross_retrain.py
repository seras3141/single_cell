import numpy as np
import pandas as pd
import pytest

from src.feature_to_mcherry.evaluation.run_matrix import drop_wells
from src.feature_to_mcherry.precross_retrain import (
    pre_window_mask,
    retrain_culture,
    run_culture,
    top_dose_wells,
    window_cutoffs,
)


def _summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "experiment": ["X"] * 5 + ["Y"],
            "well": ["A1", "A2", "A3", "A4", "N11", "A1"],
            "dose_rank": [1.0, 2.0, 1.0, 3.0, np.nan, 1.0],
            "is_dmso": [False, False, False, False, True, False],
            "t_cross_peak": [5.0, 3.0, np.nan, np.nan, 4.0, 9.0],
        }
    )


def test_top_dose_wells_are_non_dmso_rank_one_of_that_culture():
    assert top_dose_wells(_summary(), "X") == ["A1", "A3"]


def test_cutoff_is_the_earlier_of_dmso_and_own_and_a_never_crossing_well_uses_dmso():
    cutoffs = window_cutoffs(_summary(), "X", "N11")
    assert cutoffs["A1"] == 4.0
    assert cutoffs["A2"] == 3.0
    assert cutoffs["A4"] == 4.0
    assert cutoffs["N11"] == 4.0


def test_pre_window_mask_keeps_rows_up_to_and_including_the_cutoff():
    metadata = pd.DataFrame(
        {
            "sample_id": ["A1"] * 4 + ["B"] * 2,
            "timepoint": ["1", "3", "4", "5", "1", "99"],
        }
    )
    mask = pre_window_mask(metadata, {"A1": 4.0, "B": None})
    assert mask.tolist() == [True, True, True, False, True, True]


def test_run_culture_reports_all_runs_and_flags_small_folds():
    rng = np.random.default_rng(0)
    wells = np.repeat(["A", "B", "C", "D"], 60)
    timepoint = np.tile(np.repeat(["1", "2", "3"], 20), 4)
    metadata = pd.DataFrame(
        {
            "sample_id": wells,
            "timepoint": timepoint,
            "cell_id": np.tile(np.arange(20), 12),
        }
    )
    X = rng.normal(size=(240, 3))
    y = np.column_stack([X[:, 0] + rng.normal(0, 0.1, 240), X[:, 1]])
    cutoffs = {"A": 2.0, "B": 2.0, "C": 2.0, "D": None}

    tables = run_culture(
        "X", X, y, metadata, ["percentile_75", "percentile_90"], cutoffs
    )

    pooled = tables["pooled"]
    assert set(pooled["run"]) == {
        "A_pre_only",
        "B_whole_window",
        "C_B_scored_pre",
        "D_B_scored_post",
    }
    assert set(pooled["model"]) == {"ridge", "lightgbm"}
    a_rows = pooled[(pooled.run == "A_pre_only") & (pooled.model == "ridge")]
    assert (a_rows["n_rows"] == 3 * 40 + 60).all()
    assert (a_rows["n_timepoints"] == 3).all()
    assert (a_rows["n_wells"] == 4).all()
    assert set(tables["per_well"]["run"]) == {
        "A_pre_only",
        "B_whole_window",
        "C_B_scored_pre",
    }
    assert pooled[pooled.run == "C_B_scored_pre"]["well_r2_mean"].notna().all()
    d_rows = pooled[pooled.run == "D_B_scored_post"]
    assert (d_rows["n_wells"] == 3).all()
    flagged = tables["folds"][tables["folds"]["warning"] != ""]
    assert set(flagged["run"]) == {"A_pre_only"}
    assert sorted(flagged["held_out_well"]) == ["A", "B", "C"]


def test_a_dmso_well_missing_from_the_summary_is_a_clear_error():
    with pytest.raises(ValueError, match="DMSO well 'M11'"):
        window_cutoffs(_summary(), "X", "M11")


def test_a_well_listed_twice_in_the_summary_is_an_error():
    summary = pd.concat([_summary(), _summary().iloc[[1]]], ignore_index=True)
    with pytest.raises(ValueError, match="more than once"):
        window_cutoffs(summary, "X", "N11")


def test_a_well_without_a_cutoff_is_an_error_not_an_unbounded_window():
    metadata = pd.DataFrame({"sample_id": ["A1", "E7"], "timepoint": ["1", "2"]})
    with pytest.raises(ValueError, match="E7"):
        pre_window_mask(metadata, {"A1": 4.0})


def test_run_culture_needs_two_wells_with_pre_window_rows():
    metadata = pd.DataFrame(
        {
            "sample_id": ["A", "A", "B", "B"],
            "timepoint": ["1", "2", "5", "6"],
            "cell_id": [1, 2, 3, 4],
        }
    )
    X, y = np.zeros((4, 2)), np.zeros((4, 1))
    with pytest.raises(ValueError, match="at least 2"):
        run_culture("X", X, y, metadata, ["percentile_90"], {"A": 2.0, "B": 2.0})


def test_drop_wells_removes_rows_and_rejects_unknown_wells():
    features = pd.DataFrame({"sample_id": ["A", "B", "C"], "f": [1, 2, 3]})
    targets = pd.DataFrame({"sample_id": ["A", "B", "C"], "t": [1, 2, 3]})
    kept_features, kept_targets = drop_wells(features, targets, ["B"])
    assert kept_features["sample_id"].tolist() == ["A", "C"]
    assert kept_targets["sample_id"].tolist() == ["A", "C"]
    with pytest.raises(ValueError, match="E7"):
        drop_wells(features, targets, ["E7"])


def test_retraining_refuses_normalised_targets(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        "feature_to_mcherry:\n"
        "  feature_csv: unused\n"
        "  target_csv: unused\n"
        "  normalize_to_dmso: true\n"
        "  dmso_well: N11\n"
    )
    with pytest.raises(ValueError, match="raw targets only"):
        retrain_culture(
            "X", tmp_path, "N11", config, tmp_path / "summary.csv", tmp_path / "out"
        )
