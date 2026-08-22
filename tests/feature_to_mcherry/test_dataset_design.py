"""Unit tests for :mod:`src.feature_to_mcherry.dataset_design`.

Synthetic frames only -- no real data files, per the repo's testing convention.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from src.feature_to_mcherry.dataset_design import (
    LABEL_LARGE,
    LABEL_MODERATE,
    LABEL_NO_LARGE,
    LABEL_NOT_ESTIMABLE,
    WINDOW_FULL,
    WINDOW_PRE,
    build_cutoff_table,
    build_effect_table,
    classify_shift,
    cliffs_delta,
    density_vs_collapse,
    timepoint_matched_delta,
)

# ---------------------------------------------------------------------------------
# cliffs_delta
# ---------------------------------------------------------------------------------


def test_complete_separation_gives_plus_one() -> None:
    assert cliffs_delta([10, 11, 12], [1, 2, 3]) == pytest.approx(1.0)


def test_complete_separation_the_other_way_gives_minus_one() -> None:
    assert cliffs_delta([1, 2, 3], [10, 11, 12]) == pytest.approx(-1.0)


def test_identical_distributions_give_zero() -> None:
    assert cliffs_delta([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(0.0)


def test_all_ties_give_zero_not_one() -> None:
    """Ties must count as half, not as wins -- that is what rank-averaging buys."""
    assert cliffs_delta([5, 5, 5], [5, 5, 5]) == pytest.approx(0.0)


def test_half_overlap_is_signed_and_bounded() -> None:
    delta = cliffs_delta([2, 3], [1, 4])
    assert -1.0 <= delta <= 1.0
    assert delta == pytest.approx(0.0)


def test_matches_the_brute_force_definition_on_random_data() -> None:
    """Guards the rank-based shortcut against the O(n*m) definition it replaces."""
    rng = np.random.default_rng(0)
    for _ in range(5):
        a = rng.normal(size=40)
        b = rng.normal(loc=0.4, size=55)
        greater = sum(1 for x in a for y in b if x > y)
        less = sum(1 for x in a for y in b if x < y)
        brute = (greater - less) / (len(a) * len(b))
        assert cliffs_delta(a, b) == pytest.approx(brute, abs=1e-9)


def test_brute_force_agreement_holds_with_heavy_ties() -> None:
    """Ties are where a naive rank shortcut would drift from the definition."""
    a = [1, 1, 2, 2, 3]
    b = [2, 2, 3, 3, 1]
    greater = sum(1 for x in a for y in b if x > y)
    less = sum(1 for x in a for y in b if x < y)
    brute = (greater - less) / (len(a) * len(b))
    assert cliffs_delta(a, b) == pytest.approx(brute, abs=1e-12)


def test_empty_side_gives_nan() -> None:
    assert pd.isna(cliffs_delta([], [1, 2]))
    assert pd.isna(cliffs_delta([1, 2], []))


def test_non_finite_values_are_dropped_not_propagated() -> None:
    clean = cliffs_delta([1, 2, 3], [10, 11, 12])
    withnan = cliffs_delta([1, 2, 3, np.nan], [10, 11, 12, np.inf])
    assert withnan == pytest.approx(clean)


# ---------------------------------------------------------------------------------
# classify_shift
# ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "delta,expected",
    [
        (0.9, LABEL_LARGE),
        (-0.9, LABEL_LARGE),
        (0.474, LABEL_LARGE),
        (0.40, LABEL_MODERATE),
        (-0.35, LABEL_MODERATE),
        (0.33, LABEL_MODERATE),
        (0.20, LABEL_NO_LARGE),
        (0.0, LABEL_NO_LARGE),
    ],
)
def test_bands_are_symmetric_in_sign_and_fixed_at_the_boundaries(
    delta: float, expected: str
) -> None:
    assert classify_shift(delta) == expected


def test_not_estimable_wins_over_any_delta() -> None:
    assert classify_shift(0.99, estimable=False) == LABEL_NOT_ESTIMABLE


def test_nan_delta_is_not_estimable() -> None:
    assert classify_shift(float("nan")) == LABEL_NOT_ESTIMABLE
    assert classify_shift(None) == LABEL_NOT_ESTIMABLE


def test_the_no_large_label_does_not_claim_indistinguishability() -> None:
    """A single well cannot support 'not distinguishable'; the wording must not
    imply it."""
    assert "not distinguishable" not in LABEL_NO_LARGE
    assert "this imaged well" in LABEL_NO_LARGE


# ---------------------------------------------------------------------------------
# timepoint_matched_delta
# ---------------------------------------------------------------------------------


def _targets(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _well_rows(well: str, ti: int, values: List[float]) -> List[Dict[str, Any]]:
    """One row per *cell* (unique cell_id), so slice-collapsing is a no-op here."""
    return [
        {
            "sample_id": well,
            "timepoint": str(ti),
            "z_index": 0,
            "cell_id": f"{well}-{ti}-{i}",
            "percentile_90": v,
        }
        for i, v in enumerate(values)
    ]


def _slice_rows(
    well: str, ti: int, per_cell: List[List[float]]
) -> List[Dict[str, Any]]:
    """Several z-slices per cell: ``per_cell[i]`` is cell i's values across slices."""
    rows: List[Dict[str, Any]] = []
    for i, slices in enumerate(per_cell):
        for z, v in enumerate(slices):
            rows.append(
                {
                    "sample_id": well,
                    "timepoint": str(ti),
                    "z_index": z,
                    "cell_id": f"{well}-{ti}-{i}",
                    "percentile_90": v,
                }
            )
    return rows


def test_matching_happens_within_timepoint() -> None:
    """A per-timepoint effect must not be diluted by the other timepoint's cells."""
    rows: List[Dict[str, Any]] = []
    rows += _well_rows("E07", 1, [10.0] * 12)
    rows += _well_rows("M11", 1, [1.0] * 12)
    rows += _well_rows("E07", 11, [1.0] * 12)
    rows += _well_rows("M11", 11, [10.0] * 12)
    result = timepoint_matched_delta(_targets(rows), "E07", "M11", "percentile_90")
    # +1 at ti=1 and -1 at ti=11 -> median of [1, -1] == 0, and both are counted.
    assert result["n_timepoints_matched"] == 2
    assert result["delta"] == pytest.approx(0.0)


def test_timepoints_below_the_cell_floor_are_skipped() -> None:
    rows = _well_rows("E07", 1, [10.0] * 3) + _well_rows("M11", 1, [1.0] * 3)
    rows += _well_rows("E07", 11, [10.0] * 12) + _well_rows("M11", 11, [1.0] * 12)
    result = timepoint_matched_delta(_targets(rows), "E07", "M11", "percentile_90")
    assert result["n_timepoints_matched"] == 1  # ti=1 had only 3 cells per side


def test_max_ti_truncates_to_the_window() -> None:
    rows: List[Dict[str, Any]] = []
    for ti in (1, 11, 101, 201):
        rows += _well_rows("E07", ti, [10.0] * 12)
        rows += _well_rows("M11", ti, [1.0] * 12)
    full = timepoint_matched_delta(_targets(rows), "E07", "M11", "percentile_90")
    windowed = timepoint_matched_delta(
        _targets(rows), "E07", "M11", "percentile_90", max_ti=11
    )
    assert full["n_timepoints_matched"] == 4
    assert windowed["n_timepoints_matched"] == 2


def test_string_timepoints_are_compared_numerically() -> None:
    """Timepoints arrive as strings; '101' must not sort before '21' lexically."""
    rows = _well_rows("E07", 21, [10.0] * 12) + _well_rows("M11", 21, [1.0] * 12)
    rows += _well_rows("E07", 101, [10.0] * 12) + _well_rows("M11", 101, [1.0] * 12)
    windowed = timepoint_matched_delta(
        _targets(rows), "E07", "M11", "percentile_90", max_ti=21
    )
    assert windowed["n_timepoints_matched"] == 1


def test_no_shared_timepoints_gives_nan_not_zero() -> None:
    """Zero would read as 'no effect'; NaN correctly reads as 'not measured'."""
    rows = _well_rows("E07", 1, [10.0] * 12) + _well_rows("M11", 11, [1.0] * 12)
    result = timepoint_matched_delta(_targets(rows), "E07", "M11", "percentile_90")
    assert result["n_timepoints_matched"] == 0
    assert pd.isna(result["delta"])


def test_slice_rows_are_collapsed_to_one_value_per_cell() -> None:
    """instance_metrics.csv is per (cell, z_slice): ~5.4 rows per cell in real data."""
    from src.feature_to_mcherry.dataset_design import collapse_slices_to_cells

    rows = _slice_rows("E07", 1, [[10.0, 12.0, 14.0], [20.0, 22.0]])
    collapsed = collapse_slices_to_cells(_targets(rows), "percentile_90")
    assert len(rows) == 5
    assert len(collapsed) == 2
    # medians: [10,12,14] -> 12.0 ; [20,22] -> 21.0 (mean of the two middle values)
    assert sorted(collapsed["percentile_90"]) == [12.0, 21.0]


def test_missing_cell_id_raises_rather_than_using_slice_rows() -> None:
    """Falling back to raw rows would silently reinstate slice-weighted comparisons."""
    from src.feature_to_mcherry.dataset_design import collapse_slices_to_cells

    frame = pd.DataFrame(
        {"sample_id": ["E07"], "timepoint": ["1"], "percentile_90": [1.0]}
    )
    with pytest.raises(ValueError, match="cannot establish the observation unit"):
        collapse_slices_to_cells(frame, "percentile_90")


def test_slice_weighting_and_cell_weighting_can_disagree_on_the_band() -> None:
    """The reason this matters: unequal slices per cell can move the reported band.

    The drug well has one cell represented by many slices and several by one, so
    slice-weighting over-counts that cell. Collapsing first removes the imbalance.
    """
    from src.feature_to_mcherry.dataset_design import timepoint_matched_delta

    # Drug: 1 cell at 100 spread over 9 slices, plus 11 cells at 1.0 (one slice each).
    drug = _slice_rows("E07", 1, [[100.0] * 9] + [[1.0]] * 11)
    # DMSO: 12 cells at 2.0, one slice each.
    dmso = _slice_rows("M11", 1, [[2.0]] * 12)
    result = timepoint_matched_delta(
        _targets(drug + dmso), "E07", "M11", "percentile_90"
    )
    # Counts are cells, not the 20 drug slice rows.
    assert result["n_cells_drug"] == 12
    assert result["n_cells_dmso"] == 12
    assert result["n_slice_rows_in"] == 32
    # 1 of 12 cells above DMSO, 11 below -> strongly negative once cell-weighted.
    assert result["delta"] < -0.5


def test_counts_report_cells_not_slice_rows() -> None:
    from src.feature_to_mcherry.dataset_design import timepoint_matched_delta

    drug = _slice_rows("E07", 1, [[10.0, 11.0, 12.0]] * 12)  # 12 cells, 36 rows
    dmso = _slice_rows("M11", 1, [[1.0, 2.0]] * 12)  # 12 cells, 24 rows
    result = timepoint_matched_delta(
        _targets(drug + dmso), "E07", "M11", "percentile_90"
    )
    assert result["n_cells_drug"] == 12
    assert result["n_cells_dmso"] == 12
    assert result["n_slice_rows_in"] == 60


# ---------------------------------------------------------------------------------
# build_effect_table
# ---------------------------------------------------------------------------------


def _summary(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    base = {
        "experiment": "E",
        "drug": "Navitoclax",
        "dose_rank": 1.0,
        "concentration_uM": 75.0,
        "is_dmso": False,
        "peak_n_cells": 1000.0,
        "t_cross_peak": 11.0,
        "n_timepoints_pre_cross": 2,
        "end_pct_of_peak": 5.0,
        "collapse_shape": "sharp step",
    }
    return pd.DataFrame([{**base, **r} for r in rows])


def test_effect_table_emits_both_windows_per_drug_well() -> None:
    rows: List[Dict[str, Any]] = []
    for ti in (1, 11, 101, 201):
        rows += _well_rows("E07", ti, [10.0] * 12)
        rows += _well_rows("M11", ti, [1.0] * 12)
    summary = _summary(
        [
            {"well": "E07"},
            {"well": "M11", "is_dmso": True, "t_cross_peak": 101.0, "drug": "DMSO"},
        ]
    )
    table = build_effect_table(_targets(rows), summary, "E", "M11", "percentile_90")
    assert set(table["window"]) == {WINDOW_PRE, WINDOW_FULL}
    assert len(table) == 2  # one drug well x two windows; DMSO excluded
    assert (table["n_wells"] == 1).all()


def test_the_dmso_well_is_not_compared_against_itself() -> None:
    rows = _well_rows("E07", 1, [10.0] * 12) + _well_rows("M11", 1, [1.0] * 12)
    summary = _summary([{"well": "E07"}, {"well": "M11", "is_dmso": True}])
    table = build_effect_table(_targets(rows), summary, "E", "M11", "percentile_90")
    assert "M11" not in set(table["well"])


def test_direction_is_reported_separately_from_magnitude() -> None:
    """mCherry's sign is MoA/culture-specific: an unsigned magnitude would mislead."""
    rows: List[Dict[str, Any]] = []
    for ti in (1, 11, 21):
        rows += _well_rows("E07", ti, [1.0] * 12)  # drug BELOW dmso
        rows += _well_rows("M11", ti, [10.0] * 12)
    summary = _summary([{"well": "E07"}, {"well": "M11", "is_dmso": True}])
    table = build_effect_table(_targets(rows), summary, "E", "M11", "percentile_90")
    full = table[table["window"] == WINDOW_FULL].iloc[0]
    assert full["cliffs_delta"] < 0
    assert full["direction"] == "suppresses"


def test_too_few_matched_timepoints_is_labelled_not_estimable() -> None:
    """The narrow-window cultures must not receive a confident verdict."""
    rows = _well_rows("E07", 1, [10.0] * 12) + _well_rows("M11", 1, [1.0] * 12)
    summary = _summary(
        [{"well": "E07"}, {"well": "M11", "is_dmso": True, "t_cross_peak": 1.0}]
    )
    table = build_effect_table(_targets(rows), summary, "E", "M11", "percentile_90")
    pre = table[table["window"] == WINDOW_PRE].iloc[0]
    assert pre["n_timepoints_matched"] == 1
    assert pre["label"] == LABEL_NOT_ESTIMABLE


def test_missing_dmso_row_raises_rather_than_mislabelling_the_window() -> None:
    rows = _well_rows("E07", 1, [10.0] * 12)
    summary = _summary([{"well": "E07"}])
    with pytest.raises(ValueError, match="absent from the summary"):
        build_effect_table(_targets(rows), summary, "E", "M11", "percentile_90")


def test_unknown_experiment_raises() -> None:
    summary = _summary([{"well": "M11", "is_dmso": True}])
    with pytest.raises(ValueError, match="no rows in the Step 1 summary"):
        build_effect_table(pd.DataFrame(), summary, "OTHER", "M11", "percentile_90")


def test_a_three_timepoint_culture_window_is_not_estimable_in_step7_too() -> None:
    """Step 5' and Step 7 must agree on whether a culture's window is interpretable.

    Ew2-2 and SA110 have 3-timepoint pre-confluence windows, which Step 5' classifies as
    not estimable. A separate, looser threshold here previously handed those wells
    confident labels, so the deliverable contradicted itself between sections.
    """
    rows: List[Dict[str, Any]] = []
    for ti in (1, 11, 21, 101, 201):
        rows += _well_rows("E07", ti, [10.0] * 12)
        rows += _well_rows("M11", ti, [1.0] * 12)
    summary = _summary(
        [
            {"well": "E07", "n_timepoints_pre_cross": 3, "t_cross_peak": None},
            {
                "well": "M11",
                "is_dmso": True,
                "t_cross_peak": 21.0,
                "n_timepoints_pre_cross": 3,
            },
        ]
    )
    table = build_effect_table(_targets(rows), summary, "E", "M11", "percentile_90")
    pre = table[table["window"] == WINDOW_PRE].iloc[0]
    # Three matched timepoints clears the per-well floor; the culture window does not.
    assert pre["n_timepoints_matched"] == 3
    assert pre["label"] == LABEL_NOT_ESTIMABLE


def test_a_wide_culture_window_is_still_estimable() -> None:
    """The stricter rule must not swallow cultures that genuinely are interpretable."""
    rows: List[Dict[str, Any]] = []
    for ti in (1, 11, 21, 31, 41):
        rows += _well_rows("E07", ti, [10.0] * 12)
        rows += _well_rows("M11", ti, [1.0] * 12)
    summary = _summary(
        [
            {
                "well": "E07",
                "n_timepoints_pre_cross": 20,
                "t_cross_peak": None,
            },
            {
                "well": "M11",
                "is_dmso": True,
                "t_cross_peak": 41.0,
                "n_timepoints_pre_cross": 20,
            },
        ]
    )
    table = build_effect_table(_targets(rows), summary, "E", "M11", "percentile_90")
    pre = table[table["window"] == WINDOW_PRE].iloc[0]
    assert pre["label"] == LABEL_LARGE
    assert pre["cliffs_delta"] == pytest.approx(1.0)


def test_not_estimable_rows_carry_no_delta_or_direction() -> None:
    """A number printed beside 'not estimable' gets read as the effect anyway."""
    rows = _well_rows("E07", 1, [10.0] * 12) + _well_rows("M11", 1, [1.0] * 12)
    summary = _summary(
        [
            {"well": "E07", "n_timepoints_pre_cross": 2},
            {
                "well": "M11",
                "is_dmso": True,
                "t_cross_peak": 1.0,
                "n_timepoints_pre_cross": 2,
            },
        ]
    )
    table = build_effect_table(_targets(rows), summary, "E", "M11", "percentile_90")
    pre = table[table["window"] == WINDOW_PRE].iloc[0]
    assert pre["label"] == LABEL_NOT_ESTIMABLE
    assert pd.isna(pre["cliffs_delta"])
    assert pre["direction"] is None


def test_the_full_timecourse_row_survives_a_narrow_culture_window() -> None:
    """The narrow-window rule applies to the pre-confluence row only.

    The full-timecourse readout is the one T4 warned against truncating, so a narrow
    pre-confluence window must not suppress it.
    """
    rows: List[Dict[str, Any]] = []
    for ti in (1, 11, 21, 101, 201):
        rows += _well_rows("E07", ti, [10.0] * 12)
        rows += _well_rows("M11", ti, [1.0] * 12)
    summary = _summary(
        [
            {"well": "E07", "n_timepoints_pre_cross": 2},
            {
                "well": "M11",
                "is_dmso": True,
                "t_cross_peak": 11.0,
                "n_timepoints_pre_cross": 2,
            },
        ]
    )
    table = build_effect_table(_targets(rows), summary, "E", "M11", "percentile_90")
    full = table[table["window"] == WINDOW_FULL].iloc[0]
    assert full["label"] == LABEL_LARGE
    assert full["cliffs_delta"] == pytest.approx(1.0)


def test_window_verdict_is_reported_on_every_row() -> None:
    rows = _well_rows("E07", 1, [10.0] * 12) + _well_rows("M11", 1, [1.0] * 12)
    summary = _summary([{"well": "E07"}, {"well": "M11", "is_dmso": True}])
    table = build_effect_table(_targets(rows), summary, "E", "M11", "percentile_90")
    assert table["window_verdict"].notna().all()


def test_a_zero_delta_has_no_direction() -> None:
    """Zero means interchangeable -- neither elevation nor suppression."""
    rows: List[Dict[str, Any]] = []
    for ti in (1, 11, 21, 31, 41):
        rows += _well_rows("E07", ti, [5.0] * 12)
        rows += _well_rows("M11", ti, [5.0] * 12)  # identical -> delta 0
    summary = _summary(
        [
            {
                "well": "E07",
                "n_timepoints_pre_cross": 20,
                "t_cross_peak": None,
            },
            {
                "well": "M11",
                "is_dmso": True,
                "t_cross_peak": 41.0,
                "n_timepoints_pre_cross": 20,
            },
        ]
    )
    table = build_effect_table(_targets(rows), summary, "E", "M11", "percentile_90")
    pre = table[table["window"] == WINDOW_PRE].iloc[0]
    assert pre["cliffs_delta"] == pytest.approx(0.0)
    assert pre["direction"] is None
    assert pre["label"] == LABEL_NO_LARGE


# ---------------------------------------------------------------------------------
# build_cutoff_table
# ---------------------------------------------------------------------------------


def test_cutoff_table_uses_the_dmso_row_and_states_the_denominator() -> None:
    summary = _summary(
        [
            {"well": "E07", "n_timepoints_pre_cross": 30},
            {"well": "M11", "is_dmso": True, "n_timepoints_pre_cross": 2},
        ]
    )
    table = build_cutoff_table(summary, gate_threshold=0.10)
    assert len(table) == 1
    row = table.iloc[0]
    assert row["dmso_well"] == "M11"
    assert row["n_timepoints_pre_cross"] == 2  # not the drug well's 30
    assert row["denominator"] == "peak"
    assert row["verdict_cutoff_fraction"] == 0.5
    assert row["gate_cutoff_fraction"] == 0.10


def test_cutoff_table_allows_omitting_the_gate_threshold() -> None:
    summary = _summary([{"well": "M11", "is_dmso": True}])
    row = build_cutoff_table(summary, gate_threshold=None).iloc[0]
    assert row["gate_cutoff_fraction"] is None


def test_cutoff_table_raises_without_a_dmso_row() -> None:
    with pytest.raises(ValueError, match="no DMSO well"):
        build_cutoff_table(_summary([{"well": "E07"}]))


# ---------------------------------------------------------------------------------
# density_vs_collapse
# ---------------------------------------------------------------------------------


def test_censored_wells_are_excluded_and_counted_not_dropped_silently() -> None:
    summary = _summary(
        [
            {"well": "A01", "peak_n_cells": 100.0, "t_cross_peak": 11.0},
            {"well": "A02", "peak_n_cells": 200.0, "t_cross_peak": 21.0},
            {"well": "A03", "peak_n_cells": 300.0, "t_cross_peak": 31.0},
            {"well": "A04", "peak_n_cells": 400.0, "t_cross_peak": None},
        ]
    )
    pooled = density_vs_collapse(summary).iloc[0]
    assert pooled["n_wells"] == 4
    assert pooled["n_used"] == 3
    assert pooled["n_censored_excluded"] == 1


def test_perfect_monotonic_relationship_recovers_rho_one() -> None:
    summary = _summary(
        [
            {
                "well": f"A{i:02d}",
                "peak_n_cells": float(i),
                "t_cross_peak": float(i * 10),
            }
            for i in range(1, 6)
        ]
    )
    assert density_vs_collapse(summary).iloc[0]["rho"] == pytest.approx(1.0)


def test_per_culture_rows_accompany_the_pooled_row() -> None:
    rows = [
        {"experiment": "A", "well": "W1", "peak_n_cells": 1.0, "t_cross_peak": 11.0},
        {"experiment": "A", "well": "W2", "peak_n_cells": 2.0, "t_cross_peak": 21.0},
        {"experiment": "B", "well": "W3", "peak_n_cells": 3.0, "t_cross_peak": 31.0},
    ]
    out = density_vs_collapse(_summary(rows))
    assert list(out["scope"]) == ["pooled (all cultures)", "A", "B"]


def test_too_few_crossing_wells_gives_nan_rho_not_a_crash() -> None:
    summary = _summary(
        [
            {"well": "A01", "peak_n_cells": 100.0, "t_cross_peak": 11.0},
            {"well": "A02", "peak_n_cells": 200.0, "t_cross_peak": None},
        ]
    )
    assert pd.isna(density_vs_collapse(summary).iloc[0]["rho"])


# ---------------------------------------------------------------------------------
# the pre-confluence window is bounded on BOTH sides
# ---------------------------------------------------------------------------------


def test_pre_window_cuts_at_the_drug_wells_own_collapse() -> None:
    """A drug well that collapses before its reference must bound its own window.

    This is the defect the report review caught: with only the DMSO cut, timepoints
    after the DRUG well had already aggregated were included in a window described as
    "where per-cell features are valid". On real data this moved HD1509 E07 from
    delta -0.249 to -0.037.
    """
    rows: List[Dict[str, Any]] = []
    for ti in (1, 11, 21, 31, 41):
        rows += _well_rows("E07", ti, [10.0] * 12)
        rows += _well_rows("M11", ti, [1.0] * 12)
    summary = _summary(
        [
            # drug well collapses at 21, its reference not until 41
            {"well": "E07", "n_timepoints_pre_cross": 20, "t_cross_peak": 21.0},
            {
                "well": "M11",
                "is_dmso": True,
                "t_cross_peak": 41.0,
                "n_timepoints_pre_cross": 20,
            },
        ]
    )
    table = build_effect_table(_targets(rows), summary, "E", "M11", "percentile_90")
    pre = table[table["window"] == WINDOW_PRE].iloc[0]
    assert pre["window_max_ti"] == 21.0  # the earlier bound, not the DMSO 41
    assert pre["n_timepoints_matched"] == 3  # ti 1, 11, 21 -- not 31 or 41


def test_pre_window_falls_back_to_the_reference_when_the_well_never_collapses() -> None:
    """A well that never crosses imposes no bound, so the reference governs."""
    rows: List[Dict[str, Any]] = []
    for ti in (1, 11, 21, 31, 41):
        rows += _well_rows("E07", ti, [10.0] * 12)
        rows += _well_rows("M11", ti, [1.0] * 12)
    summary = _summary(
        [
            {"well": "E07", "n_timepoints_pre_cross": 20, "t_cross_peak": None},
            {
                "well": "M11",
                "is_dmso": True,
                "t_cross_peak": 21.0,
                "n_timepoints_pre_cross": 20,
            },
        ]
    )
    table = build_effect_table(_targets(rows), summary, "E", "M11", "percentile_90")
    pre = table[table["window"] == WINDOW_PRE].iloc[0]
    assert pre["window_max_ti"] == 21.0
    assert pre["n_timepoints_matched"] == 3


def test_pre_window_is_untruncated_when_neither_side_collapses() -> None:
    """Neither bound exists -> no truncation, rather than a NaN bound emptying it."""
    rows: List[Dict[str, Any]] = []
    for ti in (1, 11, 21):
        rows += _well_rows("E07", ti, [10.0] * 12)
        rows += _well_rows("M11", ti, [1.0] * 12)
    summary = _summary(
        [
            {"well": "E07", "n_timepoints_pre_cross": 20, "t_cross_peak": None},
            {
                "well": "M11",
                "is_dmso": True,
                "t_cross_peak": None,
                "n_timepoints_pre_cross": 20,
            },
        ]
    )
    table = build_effect_table(_targets(rows), summary, "E", "M11", "percentile_90")
    pre = table[table["window"] == WINDOW_PRE].iloc[0]
    assert pd.isna(pre["window_max_ti"])  # no bound -> no truncation
    assert pre["n_timepoints_matched"] == 3


def test_the_full_window_ignores_the_drug_wells_own_collapse() -> None:
    """Only the pre-confluence row is bounded; the response readout is not."""
    rows: List[Dict[str, Any]] = []
    for ti in (1, 11, 21, 31, 41):
        rows += _well_rows("E07", ti, [10.0] * 12)
        rows += _well_rows("M11", ti, [1.0] * 12)
    summary = _summary(
        [
            {"well": "E07", "n_timepoints_pre_cross": 20, "t_cross_peak": 11.0},
            {
                "well": "M11",
                "is_dmso": True,
                "t_cross_peak": 41.0,
                "n_timepoints_pre_cross": 20,
            },
        ]
    )
    table = build_effect_table(_targets(rows), summary, "E", "M11", "percentile_90")
    full = table[table["window"] == WINDOW_FULL].iloc[0]
    assert pd.isna(full["window_max_ti"])  # the full window is never truncated
    assert full["n_timepoints_matched"] == 5


def test_a_nan_max_ti_raises_rather_than_emptying_the_window() -> None:
    """NaN would make every <= comparison False and read as 'not estimable'."""
    rows = _well_rows("E07", 1, [10.0] * 12) + _well_rows("M11", 1, [1.0] * 12)
    with pytest.raises(ValueError, match="max_ti is NaN"):
        timepoint_matched_delta(
            _targets(rows), "E07", "M11", "percentile_90", max_ti=float("nan")
        )


def test_the_cell_floor_counts_finite_targets_not_rows() -> None:
    """A timepoint padded with NaN targets must not clear the per-side floor.

    Found by adversarial review. 12 drug cells of which only 2 carry a finite target
    previously passed a row-count floor of 10 and produced a confident delta computed
    from those 2. The CLI path is protected only incidentally, because load_targets
    drops NaN target rows.
    """
    nan = float("nan")
    rows = _well_rows("E07", 1, [10.0, 10.0] + [nan] * 10)
    rows += _well_rows("M11", 1, [1.0] * 12)
    result = timepoint_matched_delta(
        _targets(rows), "E07", "M11", "percentile_90", min_cells_per_side=10
    )
    assert result["n_timepoints_matched"] == 0
    assert pd.isna(result["delta"])


def test_the_cell_floor_still_admits_a_fully_finite_timepoint() -> None:
    """The finite floor must not reject legitimate data."""
    rows = _well_rows("E07", 1, [10.0] * 12) + _well_rows("M11", 1, [1.0] * 12)
    result = timepoint_matched_delta(
        _targets(rows), "E07", "M11", "percentile_90", min_cells_per_side=10
    )
    assert result["n_timepoints_matched"] == 1
    assert result["delta"] == pytest.approx(1.0)
