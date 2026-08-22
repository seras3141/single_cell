"""Unit tests for :mod:`src.feature_to_mcherry.pre_collapse`.

Synthetic frames only -- no real data files, per the repo's testing convention.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import pytest

from src.feature_to_mcherry.pre_collapse import (
    VERDICT_BORDERLINE,
    VERDICT_ESTIMABLE,
    VERDICT_NOT_ESTIMABLE,
    build_estimability_table,
    classify_estimability,
    select_informative_features,
    summarise_selection,
    top_features,
    truncate_to_pre_collapse,
)


# ---------------------------------------------------------------------------------
# classify_estimability
# ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n,expected",
    [
        (20, VERDICT_ESTIMABLE),
        (10, VERDICT_ESTIMABLE),
        (9, VERDICT_BORDERLINE),
        (6, VERDICT_BORDERLINE),
        (5, VERDICT_BORDERLINE),
        (4, VERDICT_NOT_ESTIMABLE),
        (3, VERDICT_NOT_ESTIMABLE),
        (2, VERDICT_NOT_ESTIMABLE),
        (0, VERDICT_NOT_ESTIMABLE),
    ],
)
def test_verdict_boundaries(n: int, expected: str) -> None:
    assert classify_estimability(n) == expected


def test_the_real_window_widths_get_the_plans_verdicts() -> None:
    """The observed windows must land on the verdicts the plan states."""
    observed = {"HD1509": 20, "HD1883": 6, "Ew2-1": 2, "Ew2-2": 3, "SA110": 3}
    verdicts = {k: classify_estimability(v) for k, v in observed.items()}
    assert verdicts["HD1509"] == VERDICT_ESTIMABLE
    assert verdicts["HD1883"] == VERDICT_BORDERLINE
    assert verdicts["Ew2-1"] == VERDICT_NOT_ESTIMABLE
    assert verdicts["Ew2-2"] == VERDICT_NOT_ESTIMABLE
    assert verdicts["SA110"] == VERDICT_NOT_ESTIMABLE


def test_unknown_window_is_not_estimable_rather_than_unlimited() -> None:
    """An unknown window is not evidence of a wide one."""
    assert classify_estimability(None) == VERDICT_NOT_ESTIMABLE
    assert classify_estimability(float("nan")) == VERDICT_NOT_ESTIMABLE


# ---------------------------------------------------------------------------------
# build_estimability_table
# ---------------------------------------------------------------------------------


def _summary(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    base = {
        "t_cross_peak": 11.0,
        "end_pct_of_peak": 5.0,
        "collapse_shape": "sharp step",
        "n_timepoints_pre_cross": 2,
    }
    return pd.DataFrame([{**base, **r} for r in rows])


def test_estimability_table_uses_the_dmso_row() -> None:
    summary = _summary(
        [
            {
                "experiment": "E",
                "well": "E07",
                "is_dmso": False,
                "n_timepoints_pre_cross": 30,
            },
            {
                "experiment": "E",
                "well": "M11",
                "is_dmso": True,
                "n_timepoints_pre_cross": 2,
            },
        ]
    )
    table = build_estimability_table(summary)
    assert len(table) == 1
    assert table.iloc[0]["dmso_well"] == "M11"
    # The drug well's wide window must not be what gets classified.
    assert table.iloc[0]["n_timepoints_pre_cross"] == 2
    assert table.iloc[0]["verdict"] == VERDICT_NOT_ESTIMABLE


def test_missing_dmso_row_raises() -> None:
    summary = _summary([{"experiment": "E", "well": "E07", "is_dmso": False}])
    with pytest.raises(ValueError, match="no DMSO well"):
        build_estimability_table(summary)


def test_multiple_dmso_rows_raise() -> None:
    summary = _summary(
        [
            {"experiment": "E", "well": "M11", "is_dmso": True},
            {"experiment": "E", "well": "N11", "is_dmso": True},
        ]
    )
    with pytest.raises(ValueError, match="expected exactly one"):
        build_estimability_table(summary)


# ---------------------------------------------------------------------------------
# select_informative_features
# ---------------------------------------------------------------------------------


def _univariate(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    base = {"feature": "area", "target": "percentile_90", "scope": "pooled", "rho": 0.5}
    return pd.DataFrame([{**base, **r} for r in rows])


def test_only_pooled_rows_are_selected() -> None:
    """per_group rows outnumber pooled ~9:1; mixing them would inflate the selection."""
    uni = _univariate(
        [
            {"feature": "area", "scope": "pooled", "rho": 0.6},
            {"feature": "perimeter", "scope": "per_group", "rho": 0.9},
            {"feature": "kurtosis", "scope": "per_group", "rho": 0.8},
        ]
    )
    selected = select_informative_features(uni)
    assert list(selected["feature"]) == ["area"]


def test_threshold_is_on_absolute_rho() -> None:
    uni = _univariate(
        [
            {"feature": "a", "rho": -0.9},
            {"feature": "b", "rho": 0.1},
            {"feature": "c", "rho": -0.1},
        ]
    )
    selected = select_informative_features(uni, rho_threshold=0.3)
    assert list(selected["feature"]) == ["a"]


def test_selection_is_sorted_strongest_first() -> None:
    uni = _univariate(
        [
            {"feature": "weak", "rho": 0.35},
            {"feature": "strong", "rho": -0.8},
            {"feature": "mid", "rho": 0.6},
        ]
    )
    selected = select_informative_features(uni)
    assert list(selected["feature"]) == ["strong", "mid", "weak"]


def test_absent_pooled_scope_raises_rather_than_returning_empty() -> None:
    """An empty result here means the input is not the expected univariate output."""
    uni = _univariate([{"scope": "per_group"}])
    with pytest.raises(ValueError, match="no rows with scope"):
        select_informative_features(uni)


def test_missing_columns_raise() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        select_informative_features(pd.DataFrame({"feature": ["a"]}))


def test_empty_selection_is_returned_not_raised() -> None:
    """A real-but-empty selection is the caller's decision to act on."""
    uni = _univariate([{"rho": 0.01}])
    assert select_informative_features(uni).empty


# ---------------------------------------------------------------------------------
# top_features
# ---------------------------------------------------------------------------------


def test_top_features_deduplicates_across_targets() -> None:
    """One feature selected via three targets is one panel, not three."""
    selected = pd.DataFrame(
        {
            "feature": ["area", "area", "area", "perimeter"],
            "target": ["p75", "p90", "p95", "p90"],
            "abs_rho": [0.9, 0.8, 0.7, 0.6],
        }
    )
    assert top_features(selected, limit=6) == ["area", "perimeter"]


def test_top_features_respects_the_limit() -> None:
    selected = pd.DataFrame(
        {"feature": [f"f{i}" for i in range(10)], "abs_rho": range(10)}
    )
    assert len(top_features(selected, limit=3)) == 3


# ---------------------------------------------------------------------------------
# truncate_to_pre_collapse
# ---------------------------------------------------------------------------------


def _metadata(pairs: List[tuple]) -> pd.DataFrame:
    return pd.DataFrame(
        {"sample_id": [p[0] for p in pairs], "timepoint": [p[1] for p in pairs]}
    )


def test_truncation_keeps_rows_at_or_before_t_cross() -> None:
    metadata = _metadata([("E07", 1), ("E07", 11), ("E07", 21), ("E07", 31)])
    summary = _summary(
        [{"experiment": "E", "well": "E07", "is_dmso": False, "t_cross_peak": 21.0}]
    )
    keep = truncate_to_pre_collapse(metadata, summary, "E")
    assert list(keep) == [True, True, True, False]


def test_a_well_that_never_crosses_keeps_every_timepoint() -> None:
    """Null t_cross means 'no collapse', not 'no data'."""
    metadata = _metadata([("E07", 1), ("E07", 351)])
    summary = _summary(
        [{"experiment": "E", "well": "E07", "is_dmso": False, "t_cross_peak": None}]
    )
    assert truncate_to_pre_collapse(metadata, summary, "E").all()


def test_a_well_absent_from_the_summary_keeps_its_rows_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Silently dropping unknown wells would understate the surviving data."""
    metadata = _metadata([("E07", 1), ("ZZ99", 1)])
    summary = _summary(
        [{"experiment": "E", "well": "E07", "is_dmso": False, "t_cross_peak": 21.0}]
    )
    with caplog.at_level("WARNING"):
        keep = truncate_to_pre_collapse(metadata, summary, "E")
    assert keep.all()
    assert "ZZ99" in caplog.text


def test_truncation_is_per_well_not_global() -> None:
    metadata = _metadata([("A01", 21), ("B02", 21)])
    summary = _summary(
        [
            {"experiment": "E", "well": "A01", "is_dmso": False, "t_cross_peak": 11.0},
            {"experiment": "E", "well": "B02", "is_dmso": False, "t_cross_peak": 31.0},
        ]
    )
    assert list(truncate_to_pre_collapse(metadata, summary, "E")) == [False, True]


def test_truncation_handles_string_timepoints() -> None:
    """``build_matrix_with_metadata`` normalises CELL_KEY to str, so the mask must
    compare numerically rather than lexically -- "101" sorts before "21" as text."""
    metadata = pd.DataFrame(
        {
            "sample_id": ["E07"] * 4,
            "timepoint": ["1", "21", "101", "351"],  # str, as the real metadata is
        }
    )
    summary = _summary(
        [{"experiment": "E", "well": "E07", "is_dmso": False, "t_cross_peak": 101.0}]
    )
    keep = truncate_to_pre_collapse(metadata, summary, "E")
    # Lexical comparison would wrongly drop "21" (> "101" as text) and keep "351".
    assert list(keep) == [True, True, True, False]


def test_unparseable_timepoint_is_dropped_not_silently_kept() -> None:
    """A timepoint that cannot be read as a number fails the <= test, so it is
    excluded rather than passed through as if it were in-window."""
    metadata = pd.DataFrame(
        {"sample_id": ["E07", "E07"], "timepoint": ["1", "not-a-number"]}
    )
    summary = _summary(
        [{"experiment": "E", "well": "E07", "is_dmso": False, "t_cross_peak": 101.0}]
    )
    assert list(truncate_to_pre_collapse(metadata, summary, "E")) == [True, False]


def test_unknown_experiment_raises() -> None:
    metadata = _metadata([("E07", 1)])
    summary = _summary([{"experiment": "E", "well": "E07", "is_dmso": False}])
    with pytest.raises(ValueError, match="no rows in the Step 1 summary"):
        truncate_to_pre_collapse(metadata, summary, "OTHER")


# ---------------------------------------------------------------------------------
# summarise_selection
# ---------------------------------------------------------------------------------


def test_within_window_trend_detects_a_real_trend() -> None:
    from src.feature_to_mcherry.pre_collapse import within_window_trend

    metadata = pd.DataFrame(
        {
            "sample_id": ["A"] * 6,
            "timepoint": ["1", "11", "21", "31", "41", "51"],
            "cell_id": ["1"] * 6,
        }
    )
    features = pd.DataFrame({"rising": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    out = within_window_trend(metadata, features, ["rising"], dmso_t_cross=51)
    row = out.iloc[0]
    assert row["n_within"] == 6
    assert row["rho_within"] == pytest.approx(1.0)


def test_within_window_trend_reports_a_flat_feature_as_near_zero() -> None:
    """The load-bearing case: a feature that does not move before collapse.

    A constant feature has no defined rank correlation, so this must come back NaN
    rather than a spurious value or a RuntimeWarning.
    """
    from src.feature_to_mcherry.pre_collapse import within_window_trend

    metadata = pd.DataFrame(
        {
            "sample_id": ["A"] * 6,
            "timepoint": ["1", "11", "21", "31", "41", "51"],
            "cell_id": ["1"] * 6,
        }
    )
    features = pd.DataFrame({"flat": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0]})
    out = within_window_trend(metadata, features, ["flat"], dmso_t_cross=51)
    assert pd.isna(out.iloc[0]["rho_within"])
    # Still reports the data it did see, so a flat feature is not mistaken for absent.
    assert out.iloc[0]["n_within"] == 6
    assert out.iloc[0]["median_within"] == pytest.approx(5.0)


def test_within_window_trend_returns_plain_floats_not_a_scipy_result() -> None:
    """Guards the scipy-version trap: ``result.statistic`` only exists from scipy 1.9,
    but pyproject allows ``scipy>=1.7.0``. Tuple-unpacking works across all of them."""
    from src.feature_to_mcherry.pre_collapse import within_window_trend

    metadata = pd.DataFrame(
        {
            "sample_id": ["A"] * 4,
            "timepoint": ["1", "11", "21", "31"],
            "cell_id": ["1"] * 4,
        }
    )
    features = pd.DataFrame({"f": [1.0, 2.0, 3.0, 4.0]})
    row = within_window_trend(metadata, features, ["f"], dmso_t_cross=31).iloc[0]
    assert isinstance(row["rho_within"], float)
    assert isinstance(row["p_within"], float)


def test_within_window_trend_handles_too_few_points_without_raising() -> None:
    """Two in-window points cannot support a rank correlation."""
    from src.feature_to_mcherry.pre_collapse import within_window_trend

    metadata = pd.DataFrame(
        {"sample_id": ["A", "A"], "timepoint": ["1", "11"], "cell_id": ["1", "1"]}
    )
    features = pd.DataFrame({"f": [1.0, 2.0]})
    row = within_window_trend(metadata, features, ["f"], dmso_t_cross=11).iloc[0]
    assert row["n_within"] == 2
    assert pd.isna(row["rho_within"])


def test_within_window_trend_splits_at_the_dmso_crossing() -> None:
    """Post-window rows are reported separately, not folded into the trend."""
    from src.feature_to_mcherry.pre_collapse import within_window_trend

    metadata = pd.DataFrame(
        {
            "sample_id": ["A"] * 4,
            "timepoint": ["1", "11", "101", "201"],
            "cell_id": ["1"] * 4,
        }
    )
    features = pd.DataFrame({"f": [10.0, 10.0, 50.0, 50.0]})
    out = within_window_trend(metadata, features, ["f"], dmso_t_cross=11)
    row = out.iloc[0]
    assert row["n_within"] == 2
    assert row["n_post"] == 2
    assert row["median_within"] == pytest.approx(10.0)
    assert row["median_post"] == pytest.approx(50.0)
    assert row["pct_change_post_vs_within"] == pytest.approx(400.0)


def test_within_window_trend_without_a_crossing_uses_every_row() -> None:
    from src.feature_to_mcherry.pre_collapse import within_window_trend

    metadata = pd.DataFrame(
        {
            "sample_id": ["A"] * 3,
            "timepoint": ["1", "11", "21"],
            "cell_id": ["1"] * 3,
        }
    )
    features = pd.DataFrame({"f": [1.0, 2.0, 3.0]})
    out = within_window_trend(metadata, features, ["f"], dmso_t_cross=None)
    assert out.iloc[0]["n_within"] == 3
    assert out.iloc[0]["n_post"] == 0


def test_selection_summary_reports_counts_and_the_strongest_pair() -> None:
    selected = pd.DataFrame(
        {
            "feature": ["area", "perimeter"],
            "target": ["percentile_90", "percentile_75"],
            "abs_rho": [0.67, 0.4],
        }
    )
    out = summarise_selection({"SA110": selected}, 0.3)
    row = out.iloc[0]
    assert row["n_pairs_selected"] == 2
    assert row["n_distinct_features"] == 2
    assert row["max_abs_rho"] == pytest.approx(0.67)
    assert row["strongest_pair"] == "area ~ percentile_90"


def test_selection_summary_handles_an_empty_selection() -> None:
    out = summarise_selection(
        {"X": pd.DataFrame(columns=["feature", "target", "abs_rho"])}, 0.3
    )
    row = out.iloc[0]
    assert row["n_pairs_selected"] == 0
    assert row["strongest_pair"] is None


def test_within_window_trend_counts_cells_not_slices() -> None:
    """Two cells spanning 3 and 1 z-slices are 2 observations, not 4."""
    from src.feature_to_mcherry.pre_collapse import within_window_trend

    metadata = pd.DataFrame(
        {
            "sample_id": ["A"] * 4,
            "timepoint": ["1"] * 4,
            "z_index": ["5", "6", "7", "5"],
            "cell_id": ["1", "1", "1", "2"],
        }
    )
    features = pd.DataFrame({"f": [10.0, 20.0, 30.0, 100.0]})
    row = within_window_trend(metadata, features, ["f"], dmso_t_cross=None).iloc[0]
    assert row["n_within"] == 2
    # median over cells {median(10, 20, 30) = 20, 100} = median(20, 100) = 60
    assert row["median_within"] == pytest.approx(60.0)


def test_within_window_trend_is_not_skewed_by_slice_span() -> None:
    """The load-bearing regression test for the slice-weighting defect.

    Two cells at each of two timepoints. At t=1 the low-valued cell spans 3 z-slices;
    at t=11 the high-valued one does. On raw slice rows the median is pulled toward
    whichever cell spans more z, manufacturing a time trend; per cell there is none.

    Raw rows   t=1: [1, 1, 1, 9] -> median 1    t=11: [1, 9, 9, 9] -> median 9
    Per cell   t=1: [1, 9]       -> median 5    t=11: [1, 9]       -> median 5
    """
    from src.feature_to_mcherry.pre_collapse import within_window_trend

    metadata = pd.DataFrame(
        {
            "sample_id": ["A"] * 8,
            "timepoint": ["1"] * 4 + ["11"] * 4,
            "z_index": ["5", "6", "7", "5", "5", "5", "6", "7"],
            "cell_id": ["1", "1", "1", "2", "1", "2", "2", "2"],
        }
    )
    features = pd.DataFrame({"f": [1.0, 1.0, 1.0, 9.0, 1.0, 9.0, 9.0, 9.0]})
    row = within_window_trend(metadata, features, ["f"], dmso_t_cross=None).iloc[0]
    assert row["n_within"] == 4  # 2 cells x 2 timepoints
    # Both timepoints now hold the same pair {1, 9}, so no time trend survives.
    assert row["rho_within"] == pytest.approx(0.0)


def test_within_window_trend_requires_the_observation_unit() -> None:
    """No silent fall-back to slice rows: metadata without cell_id must fail."""
    from src.feature_to_mcherry.pre_collapse import within_window_trend

    metadata = pd.DataFrame({"sample_id": ["A", "A"], "timepoint": ["1", "11"]})
    features = pd.DataFrame({"f": [1.0, 2.0]})
    with pytest.raises(ValueError, match="cell_id"):
        within_window_trend(metadata, features, ["f"], dmso_t_cross=None)
