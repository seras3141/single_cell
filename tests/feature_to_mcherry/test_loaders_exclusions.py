"""Excluded-stack drop in the feature_to_mcherry loaders."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.feature_to_mcherry.data.loaders import (
    load_features,
    load_features_from_directory,
    load_targets,
)
from src.utils.data_exclusions import DataExclusions, ExcludedStack

EXP = "HD1883 MF5V1 0-72h 20-03-26"
REGISTRY = DataExclusions(excluded_stacks=(ExcludedStack(EXP, "H09", 201, "shifted"),))


def _targets(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "sample_id": ["H09", "H09", "H10"],
            "timepoint": [201, 211, 201],
            "z_index": [4, 4, 4],
            "cell_id": [1, 1, 1],
            "percentile_75": [1.0, 2.0, 3.0],
            "percentile_90": [1.0, 2.0, 3.0],
            "percentile_95": [1.0, 2.0, 3.0],
        }
    ).to_csv(path, index=False)
    return path


def _features(path: Path, sample: str, timepoint: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "cell_id": [1, 2],
            "area": [10.0, 12.0],
            "sample_id": [sample] * 2,
            "timepoint": [timepoint] * 2,
            "z_index": [4, 4],
        }
    ).to_csv(path, index=False)
    return path


def test_targets_drop_only_the_excluded_stack(tmp_path):
    csv = _targets(tmp_path / EXP / "mcherry_metrics" / "instance_metrics.csv")
    df = load_targets(csv, exclusions=REGISTRY)
    assert list(zip(df["sample_id"], df["timepoint"])) == [("H09", 211), ("H10", 201)]


def test_other_experiment_untouched(tmp_path):
    csv = _targets(tmp_path / "HD1509 MF5V1 0-72h 23-02-26" / "instance_metrics.csv")
    assert len(load_targets(csv, exclusions=REGISTRY)) == 3


def test_empty_registry_disables_drop(tmp_path):
    csv = _targets(tmp_path / EXP / "instance_metrics.csv")
    assert len(load_targets(csv, exclusions=DataExclusions.empty())) == 3


def test_features_string_timepoint_dropped(tmp_path):
    csv = _features(tmp_path / EXP / "split" / "a.csv", "H09", "201")
    assert load_features(csv, id_column="cell_id", exclusions=REGISTRY).empty


def test_features_from_directory(tmp_path):
    split = tmp_path / EXP / "features_incarta" / "split_data"
    _features(split / "pMF5V1_H09_t201_z4_BF_features.csv", "H09", "201")
    _features(split / "pMF5V1_H09_t211_z4_BF_features.csv", "H09", "211")
    df = load_features_from_directory(split, id_column="cell_id", exclusions=REGISTRY)
    assert set(df["timepoint"].astype(str)) == {"211"}


def test_default_registry_drops_nothing_today(tmp_path):
    csv = _targets(tmp_path / EXP / "instance_metrics.csv")
    assert len(load_targets(csv)) == 3  # tracked registry has no entries


def test_float_timepoint_column_still_matches(tmp_path):
    csv = tmp_path / EXP / "instance_metrics.csv"
    csv.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "sample_id": ["H09", "H09"],
            "timepoint": [201.0, None],  # a blank makes pandas read float
            "z_index": [4, 4],
            "cell_id": [1, 2],
            "percentile_75": [1.0, 1.0],
            "percentile_90": [1.0, 1.0],
            "percentile_95": [1.0, 1.0],
        }
    ).to_csv(csv, index=False)
    df = load_targets(csv, exclusions=REGISTRY)
    assert list(df["cell_id"]) == [2]
