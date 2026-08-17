"""Unit tests for :mod:`src.dataset_analysis.collapse_summary`.

Synthetic trajectories only -- no real data files, per the repo's testing convention.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd
import pytest

from src.dataset_analysis.collapse_summary import (
    SUMMARY_COLUMNS,
    assert_well_composition,
    compute_collapse_metrics,
    dmso_reference_table,
    summarize_all_experiments,
    summarize_experiment,
)

LAYOUT_PATH = Path(__file__).resolve().parents[2] / "config" / "MF5v1_plate_layout.json"


def _trajectory(counts: Sequence[float], *, step: int = 10) -> pd.DataFrame:
    """Build a trajectory with frame indices 1, 11, 21, ... like the real data."""
    return pd.DataFrame(
        {
            "ti": [1 + step * i for i in range(len(counts))],
            "n_cells": list(counts),
        }
    )


# --------------------------------------------------------------------------------------
# compute_collapse_metrics
# --------------------------------------------------------------------------------------


def test_sharp_step_collapse_is_labelled_and_located() -> None:
    # Peak at the first sample, crossing 50% immediately after -> sharp step.
    metrics = compute_collapse_metrics(_trajectory([100, 90, 20, 18, 16, 15]))
    assert metrics["peak_n_cells"] == 100.0
    assert metrics["t_cross_peak"] == 21  # first ti with n_cells < 50
    assert metrics["collapse_shape"] == "sharp step"
    assert metrics["distinct_collapses"] is True


def test_gradual_collapse_when_crossing_lags_the_peak() -> None:
    # Peak early, but the crossing only arrives many samples later -> gradual.
    counts = [100, 95, 90, 85, 80, 70, 60, 40, 30, 25, 20, 15]
    metrics = compute_collapse_metrics(_trajectory(counts))
    assert metrics["t_cross_peak"] == 71  # index 7 -> ti 71
    assert metrics["collapse_shape"] == "gradual"


def test_no_material_collapse_leaves_t_cross_none() -> None:
    metrics = compute_collapse_metrics(_trajectory([100, 98, 96, 95, 94, 93]))
    assert metrics["t_cross_peak"] is None
    assert metrics["collapse_shape"] == "no material collapse"
    assert metrics["distinct_collapses"] is False


def test_n_timepoints_pre_cross_counts_samples_up_to_and_including_crossing() -> None:
    metrics = compute_collapse_metrics(_trajectory([100, 90, 20, 18]))
    assert metrics["t_cross_peak"] == 21
    # ti 1, 11, 21 are at or before the crossing.
    assert metrics["n_timepoints_pre_cross"] == 3


def test_n_timepoints_pre_cross_is_full_length_when_never_crossing() -> None:
    """A well that never collapses has *every* timepoint usable, not zero.

    Downstream truncation (plan Step 5') keeps all timepoints for such wells, so a 0
    here would wrongly exclude them. ``t_cross_peak is None`` is the no-crossing signal.
    """
    metrics = compute_collapse_metrics(_trajectory([100, 98, 96, 95]))
    assert metrics["t_cross_peak"] is None
    assert metrics["n_timepoints_pre_cross"] == 4


def test_peak_and_first3_denominators_differ_when_decline_starts_immediately() -> None:
    """The two denominators are not interchangeable -- this is plan §B2/§B3.

    With a steep early drop, mean-of-first-3 sits well below the peak, so the same well
    reports a materially higher end-percentage against it.
    """
    counts = [1000, 400, 300, 260, 240, 100, 100, 100]
    metrics = compute_collapse_metrics(_trajectory(counts))
    assert metrics["peak_n_cells"] == 1000.0
    assert metrics["base_first3_n_cells"] == pytest.approx((1000 + 400 + 300) / 3)
    assert metrics["end_n_cells"] == pytest.approx(100.0)
    assert metrics["end_pct_of_peak"] == pytest.approx(10.0)
    assert metrics["end_pct_of_first3"] == pytest.approx(17.647059, abs=1e-5)
    # And the crossings land at different frames for the same reason: 50% of peak is
    # 500 (crossed at ti=11 by 400), but 50% of the first-3 mean is only ~283 (not
    # crossed until ti=31 by 260).
    assert metrics["t_cross_peak"] == 11
    assert metrics["t_cross_first3"] == 31


def test_unsorted_input_is_ordered_by_frame_index() -> None:
    """Rows arriving out of order must not scramble the time axis."""
    ordered = _trajectory([100, 90, 20, 18])
    shuffled = ordered.iloc[[2, 0, 3, 1]].reset_index(drop=True)
    assert compute_collapse_metrics(shuffled) == compute_collapse_metrics(ordered)


def test_edge_n_is_configurable() -> None:
    metrics = compute_collapse_metrics(_trajectory([100, 80, 60, 40, 20]), edge_n=1)
    assert metrics["base_first3_n_cells"] == 100.0
    assert metrics["end_n_cells"] == 20.0


def test_single_timepoint_well_does_not_crash() -> None:
    metrics = compute_collapse_metrics(_trajectory([42]))
    assert metrics["peak_n_cells"] == 42.0
    assert metrics["end_n_cells"] == 42.0
    assert metrics["t_cross_peak"] is None
    assert metrics["n_timepoints_pre_cross"] == 1
    assert metrics["distinct_collapses"] is False


def test_all_zero_counts_yield_nan_percentages_not_zero_division() -> None:
    metrics = compute_collapse_metrics(_trajectory([0, 0, 0]))
    assert pd.isna(metrics["end_pct_of_peak"])
    assert pd.isna(metrics["end_pct_of_first3"])
    assert metrics["t_cross_peak"] is None


def test_empty_trajectory_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        compute_collapse_metrics(_trajectory([]))


def test_missing_column_raises() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        compute_collapse_metrics(pd.DataFrame({"ti": [1, 11]}))


# --------------------------------------------------------------------------------------
# summarize_experiment
# --------------------------------------------------------------------------------------


def _cell_population(wells: Dict[str, Sequence[float]]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for well, counts in wells.items():
        frame = _trajectory(counts)
        frame["sample_id"] = well
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture()
def layout() -> Dict[str, Any]:
    return json.loads(LAYOUT_PATH.read_text())


def test_summarize_experiment_emits_the_declared_columns() -> None:
    summary = summarize_experiment(
        _cell_population({"E07": [100, 90, 20], "M11": [50, 40, 10]}),
        "TestExp",
        dmso_well="M11",
    )
    assert list(summary.columns) == list(SUMMARY_COLUMNS)
    assert len(summary) == 2
    assert "verdict" not in summary.columns  # plan Step 1.5: deliberately absent


def test_summarize_experiment_marks_the_dmso_well() -> None:
    summary = summarize_experiment(
        _cell_population({"E07": [100, 50, 10], "M11": [80, 40, 8]}),
        "TestExp",
        dmso_well="m11",  # case-insensitive
    )
    dmso = summary.set_index("well")["is_dmso"]
    assert bool(dmso["M11"]) is True
    assert bool(dmso["E07"]) is False


def test_control_well_is_labelled_from_the_layouts_control_field(
    layout: Dict[str, Any],
) -> None:
    """M11 must come out as DMSO, not as an unannotated well.

    Control rows carry ``drug=None`` and put their identity in ``control``. Reading only
    ``drug`` would leave M11 null and trip the composition guard as a false positive.
    """
    summary = summarize_experiment(
        _cell_population({"M11": [100, 50, 10]}), "TestExp", layout=layout
    )
    row = summary.iloc[0]
    assert row["drug"] == "DMSO"
    assert row["dose_rank"] is None or pd.isna(row["dose_rank"])


def test_composition_guard_accepts_a_real_shaped_experiment(
    layout: Dict[str, Any],
) -> None:
    """End-to-end: 8 drug wells + M11 must pass the guard with a real layout."""
    wells = {w: [100, 90, 20] for w in ("C09", "D07", "D08", "D10")}
    wells.update({w: [100, 90, 20] for w in ("E07", "E08", "E09", "E10")})
    wells["M11"] = [100, 50, 10]
    summary = summarize_experiment(
        _cell_population(wells), "Ew2-1", layout=layout, dmso_well="M11"
    )
    assert_well_composition(summary)  # must not raise


def test_column_7_resolves_to_top_dose_not_empty(layout: Dict[str, Any]) -> None:
    """Plate-layout regression guard: E07 is Navitoclax 75 uM, dose_rank 1.

    Before commit b35396f the layout model labelled every column-7 well ``empty``.
    """
    summary = summarize_experiment(
        _cell_population({"E07": [100, 90, 20]}), "TestExp", layout=layout
    )
    row = summary.iloc[0]
    assert row["drug"] == "Navitoclax"
    assert row["concentration_uM"] == 75.0
    assert row["dose_rank"] == 1


def test_dose_rank_decreases_across_the_imaged_columns(layout: Dict[str, Any]) -> None:
    summary = summarize_experiment(
        _cell_population({w: [100, 50, 10] for w in ("E07", "E08", "E09", "E10")}),
        "TestExp",
        layout=layout,
    )
    ranks = summary.set_index("well")["dose_rank"].to_dict()
    assert ranks == {"E07": 1, "E08": 2, "E09": 3, "E10": 4}
    concentrations = summary.set_index("well")["concentration_uM"].to_dict()
    assert concentrations == {"E07": 75.0, "E08": 10.0, "E09": 1.0, "E10": 0.1}


def test_midline_spacer_column_is_still_reported_empty(layout: Dict[str, Any]) -> None:
    """The fix must not over-correct: column 12 really is a spacer."""
    summary = summarize_experiment(
        _cell_population({"E12": [100, 50, 10]}), "TestExp", layout=layout
    )
    assert summary.iloc[0]["drug"] is None or pd.isna(summary.iloc[0]["drug"])


def test_drug_falls_back_to_the_tables_own_column_without_a_layout() -> None:
    table = _cell_population({"E07": [100, 50, 10]})
    table["drug"] = "Navitoclax"
    summary = summarize_experiment(table, "TestExp", layout=None)
    assert summary.iloc[0]["drug"] == "Navitoclax"
    assert summary.iloc[0]["dose_rank"] is None


def test_csv_drug_column_does_not_mask_a_failed_layout_lookup(
    layout: Dict[str, Any],
) -> None:
    """With a layout supplied, a spacer well must NOT borrow the CSV's ``drug`` value.

    ``cell_population.csv``'s own ``drug`` column is produced by the same
    ``get_well_annotation``, so borrowing it would hide a layout regression: ``drug``
    would still read "Navitoclax" while ``dose_rank``/``concentration_uM`` were empty,
    and :func:`assert_well_composition` would pass. Column 12 is a real spacer, so it
    stands in for any well the layout declines to annotate.
    """
    table = _cell_population({"E12": [100, 50, 10]})
    table["drug"] = "Navitoclax"  # as a post-fix CSV would have for a data column
    summary = summarize_experiment(table, "TestExp", layout=layout)
    row = summary.iloc[0]
    assert row["drug"] is None or pd.isna(row["drug"])
    assert row["dose_rank"] is None or pd.isna(row["dose_rank"])


# --------------------------------------------------------------------------------------
# assert_well_composition
# --------------------------------------------------------------------------------------


def _summary_rows(n_drug: int, n_dmso: int, drug: Any = "Navitoclax") -> pd.DataFrame:
    rows = [
        {
            "experiment": "E",
            "well": f"D{i:02d}",
            "drug": drug,
            "is_dmso": False,
            "dose_rank": 1,
            "concentration_uM": 75.0,
        }
        for i in range(n_drug)
    ]
    rows += [
        {
            "experiment": "E",
            "well": f"M{i:02d}",
            "drug": "DMSO",
            "is_dmso": True,
            "dose_rank": None,
            "concentration_uM": None,
        }
        for i in range(n_dmso)
    ]
    return pd.DataFrame(rows)


def test_composition_check_passes_on_8_drug_plus_1_dmso() -> None:
    assert_well_composition(_summary_rows(8, 1))  # must not raise


def test_composition_check_rejects_an_empty_annotated_well() -> None:
    bad = _summary_rows(8, 1)
    bad.loc[0, "drug"] = "empty"
    with pytest.raises(AssertionError, match="plate-layout regression"):
        assert_well_composition(bad)


def test_composition_check_rejects_an_unannotated_well() -> None:
    bad = _summary_rows(8, 1)
    bad.loc[0, "drug"] = None
    with pytest.raises(AssertionError, match="unannotated"):
        assert_well_composition(bad)


def test_composition_check_rejects_a_wrong_well_count() -> None:
    with pytest.raises(AssertionError, match="expected 8 drug"):
        assert_well_composition(_summary_rows(7, 1))


def test_composition_check_rejects_a_drug_well_missing_its_dose_fields() -> None:
    """`drug` alone is not enough -- the layout-derived dose fields must be present too.

    Guards the failure mode where `drug` is populated from some non-layout source while
    the layout lookup silently returned nothing, leaving the dose fields null.
    """
    bad = _summary_rows(8, 1)
    bad.loc[0, "dose_rank"] = None
    with pytest.raises(AssertionError, match="no layout-derived"):
        assert_well_composition(bad)


def test_composition_check_ignores_missing_dose_on_the_dmso_well() -> None:
    """The DMSO well legitimately has no dose, so it must not trip the dose check."""
    assert_well_composition(_summary_rows(8, 1))  # DMSO rows already carry None


# --------------------------------------------------------------------------------------
# summarize_all_experiments / dmso_reference_table
# --------------------------------------------------------------------------------------


def test_summarize_all_experiments_concatenates_and_sorts(tmp_path: Path) -> None:
    csvs = {}
    for name, counts in (("Bexp", [100, 90, 20]), ("Aexp", [80, 70, 10])):
        path = tmp_path / f"{name}.csv"
        _cell_population({"E07": counts, "M11": counts}).to_csv(path, index=False)
        csvs[name] = path

    summary = summarize_all_experiments(csvs, dmso_wells={"Aexp": "M11", "Bexp": "M11"})
    assert list(summary["experiment"]) == ["Aexp", "Aexp", "Bexp", "Bexp"]
    assert list(summary["well"]) == ["E07", "M11", "E07", "M11"]


def test_summarize_all_experiments_reports_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Aexp"):
        summarize_all_experiments({"Aexp": tmp_path / "nope.csv"})


def test_dmso_reference_table_returns_only_dmso_rows(tmp_path: Path) -> None:
    path = tmp_path / "exp.csv"
    _cell_population({"E07": [100, 90, 20], "M11": [80, 70, 10]}).to_csv(
        path, index=False
    )
    summary = summarize_all_experiments({"Exp": path}, dmso_wells={"Exp": "M11"})
    dmso = dmso_reference_table(summary)
    assert list(dmso["well"]) == ["M11"]
    assert "end_pct_of_peak" in dmso.columns
    assert "end_pct_of_first3" in dmso.columns
