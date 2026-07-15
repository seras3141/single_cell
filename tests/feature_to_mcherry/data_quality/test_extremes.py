"""Tests for feature_to_mcherry.data_quality.extremes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.feature_to_mcherry.data_quality.extremes import (
    REPORT_COLUMNS,
    compute_extreme_value_report,
)


def _synthetic_tables(seed: int = 0):
    rng = np.random.default_rng(seed)
    n_per_well = 200
    wells = ["A01", "A02", "A03"]

    rows_features = []
    rows_targets = []
    cell_id = 0
    for well in wells:
        values = rng.normal(0, 1, size=n_per_well)
        # Inject a cluster of extreme values only in A01. Values are varied (not a
        # repeated constant) so none of them ties exactly with the quantile
        # boundary computed below (a tie would sit exactly at q_hi and fail the
        # strict "> q_hi" extreme test).
        if well == "A01":
            values[:10] = rng.uniform(50, 150, size=10)
        for i in range(n_per_well):
            rows_features.append(
                {
                    "sample_id": well,
                    "timepoint": 1 if i % 2 == 0 else 11,
                    "z_index": 1,
                    "cell_id": cell_id,
                    "feat1": values[i],
                }
            )
            rows_targets.append(
                {
                    "sample_id": well,
                    "timepoint": 1 if i % 2 == 0 else 11,
                    "z_index": 1,
                    "cell_id": cell_id,
                    "percentile_75": rng.normal(50, 5),
                }
            )
            cell_id += 1

    return pd.DataFrame(rows_features), pd.DataFrame(rows_targets)


def test_compute_extreme_value_report_flags_injected_cluster_by_well() -> None:
    features_df, targets_df = _synthetic_tables()

    report = compute_extreme_value_report(
        features_df,
        targets_df,
        source_label="TestSource",
        quantile_lo=0.01,
        quantile_hi=0.99,
    )

    assert list(report.columns) == REPORT_COLUMNS
    assert (report["source"] == "TestSource").all()

    by_well = report[
        (report["value_column"] == "feat1") & (report["group_type"] == "sample_id")
    ].sort_values("enrichment", ascending=False)
    assert by_well.iloc[0]["group_value"] == "A01"
    assert by_well.iloc[0]["enrichment"] > by_well.iloc[1]["enrichment"]


def test_compute_extreme_value_report_includes_target_columns() -> None:
    features_df, targets_df = _synthetic_tables()

    report = compute_extreme_value_report(features_df, targets_df, source_label="S")

    assert "percentile_75" in set(report["value_column"])


def test_compute_extreme_value_report_covers_all_group_types() -> None:
    features_df, targets_df = _synthetic_tables()

    report = compute_extreme_value_report(features_df, targets_df, source_label="S")

    assert set(report["group_type"]) == {"sample_id", "z_index", "timepoint"}


def test_compute_extreme_value_report_skips_all_nan_column() -> None:
    features_df, targets_df = _synthetic_tables()
    features_df["broken"] = np.nan

    report = compute_extreme_value_report(features_df, targets_df, source_label="S")

    assert "broken" not in set(report["value_column"])


def test_compute_extreme_value_report_enrichment_hand_computed() -> None:
    # 4 cells, 2 wells; well "B" has both extreme values.
    features_df = pd.DataFrame(
        {
            "sample_id": ["A", "A", "B", "B"],
            "timepoint": [1, 1, 1, 1],
            "z_index": [1, 1, 1, 1],
            "cell_id": [1, 2, 3, 4],
            "feat1": [0.0, 0.1, 100.0, 100.0],
        }
    )
    targets_df = pd.DataFrame(
        {
            "sample_id": ["A", "A", "B", "B"],
            "timepoint": [1, 1, 1, 1],
            "z_index": [1, 1, 1, 1],
            "cell_id": [1, 2, 3, 4],
            "percentile_75": [50.0, 51.0, 52.0, 53.0],
        }
    )

    report = compute_extreme_value_report(
        features_df, targets_df, source_label="S", quantile_lo=0.0, quantile_hi=0.5
    )

    row_b = report[
        (report["value_column"] == "feat1")
        & (report["group_type"] == "sample_id")
        & (report["group_value"] == "B")
    ].iloc[0]
    assert row_b["n"] == 2
    assert row_b["n_extreme"] == 2
    assert row_b["extreme_rate"] == 1.0
