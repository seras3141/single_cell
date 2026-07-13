"""Tests for feature_to_mcherry.informativeness.noise_ceiling.

Uses the real MF5v1 plate layout (config/MF5v1_plate_layout.json) so the
well->condition mapping is exercised against real drug/dose structure, with
synthetic per-well target values crafted to control between/within-condition
variance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.feature_to_mcherry.informativeness.noise_ceiling import (
    INSUFFICIENT_REPLICATES_REASON,
    NO_LAYOUT_REASON,
    compute_noise_ceiling,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
REAL_LAYOUT_PATH = REPO_ROOT / "config" / "MF5v1_plate_layout.json"

TARGET_COLUMN = "percentile_75"

# Doxorubicin rows C (replicate 1) and D (replicate 2); quadrant-1 column offsets
# 2 and 3 give conc_1 (1.0 uM) and conc_2 (0.1 uM) respectively (see the layout json).


def _targets_df(well_values: dict) -> pd.DataFrame:
    rows = [
        {"sample_id": well, TARGET_COLUMN: value} for well, value in well_values.items()
    ]
    return pd.DataFrame(rows)


def test_compute_noise_ceiling_high_when_replicates_agree_and_conditions_differ() -> (
    None
):
    # Condition A (conc_1): C02, D02 agree closely. Condition B (conc_2): C03, D03
    # agree closely but at a very different level -> high between-, low within-variance.
    targets_df = _targets_df({"C02": 10.0, "D02": 10.2, "C03": 50.0, "D03": 50.3})

    result = compute_noise_ceiling(
        targets_df, [TARGET_COLUMN], str(REAL_LAYOUT_PATH), sample_id_column="sample_id"
    )

    row = result.iloc[0]
    assert row["n_conditions"] == 2
    assert row["ceiling"] > 0.8
    assert row["reason"] == ""


def test_compute_noise_ceiling_low_when_replicates_disagree_within_condition() -> None:
    # Both conditions share the same mean (30) but replicates disagree wildly within
    # each condition -> ~zero between-variance, large within-variance -> low/negative
    # ICC.
    targets_df = _targets_df({"C02": 10.0, "D02": 50.0, "C03": 15.0, "D03": 45.0})

    result = compute_noise_ceiling(
        targets_df, [TARGET_COLUMN], str(REAL_LAYOUT_PATH), sample_id_column="sample_id"
    )

    row = result.iloc[0]
    assert row["ceiling"] < 0.2


def test_compute_noise_ceiling_returns_nan_reason_when_layout_missing() -> None:
    targets_df = _targets_df({"C02": 10.0, "D02": 10.2})

    result = compute_noise_ceiling(
        targets_df, [TARGET_COLUMN], None, sample_id_column="sample_id"
    )

    row = result.iloc[0]
    assert np.isnan(row["ceiling"])
    assert row["reason"] == NO_LAYOUT_REASON


def test_compute_noise_ceiling_returns_nan_reason_when_layout_file_absent() -> None:
    targets_df = _targets_df({"C02": 10.0, "D02": 10.2})

    result = compute_noise_ceiling(
        targets_df,
        [TARGET_COLUMN],
        "/nonexistent/plate_layout.json",
        sample_id_column="sample_id",
    )

    row = result.iloc[0]
    assert np.isnan(row["ceiling"])
    assert row["reason"] == NO_LAYOUT_REASON


def test_compute_noise_ceiling_returns_nan_reason_when_no_replicate_structure() -> None:
    # Each well is a distinct condition (different concentrations) -> no condition has
    # >=2 replicate wells.
    targets_df = _targets_df({"C02": 10.0, "C03": 20.0, "C04": 30.0})

    result = compute_noise_ceiling(
        targets_df, [TARGET_COLUMN], str(REAL_LAYOUT_PATH), sample_id_column="sample_id"
    )

    row = result.iloc[0]
    assert np.isnan(row["ceiling"])
    assert row["reason"] == INSUFFICIENT_REPLICATES_REASON
    assert row["n_conditions"] == 0
