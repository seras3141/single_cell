"""Unit tests for :mod:`src.dataset_analysis.gate_survival`.

Synthetic trajectories only -- no real data files, per the repo's testing convention.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import pandas as pd
import pytest

from src.dataset_analysis.gate_survival import (
    ABSOLUTE_FLOOR_LABEL,
    aggregate_by_threshold,
    build_flag_frame,
    crosscheck_against_t_cross,
    keep_window,
    per_timepoint_flags,
    ratcheted_flags,
    summarise_survival,
)


def _cell_population(wells: Dict[str, Sequence[float]], *, step: int = 10):
    frames: List[pd.DataFrame] = []
    for well, counts in wells.items():
        frames.append(
            pd.DataFrame(
                {
                    "sample_id": well,
                    "ti": [1 + step * i for i in range(len(counts))],
                    "n_cells": list(counts),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------------
# ratcheted_flags / per_timepoint_flags
# ---------------------------------------------------------------------------------


def test_ratchet_flags_the_crossing_and_everything_after() -> None:
    flags = ratcheted_flags([100, 90, 40, 30, 20], threshold=50)
    assert list(flags) == [False, False, True, True, True]


def test_ratchet_does_not_unflag_on_a_later_rebound() -> None:
    """A re-fragmenting spheroid does not restore trustworthy per-cell statistics."""
    flags = ratcheted_flags([100, 40, 90, 95], threshold=50)
    assert list(flags) == [False, True, True, True]


def test_ratchet_flags_nothing_when_never_crossing() -> None:
    flags = ratcheted_flags([100, 90, 80], threshold=50)
    assert not flags.any()


def test_ratchet_flags_everything_when_starting_below() -> None:
    flags = ratcheted_flags([10, 9, 8], threshold=50)
    assert flags.all()


def test_per_timepoint_flags_are_not_ratcheted() -> None:
    """The absolute floor asks a per-timepoint question, so a rebound clears it."""
    flags = per_timepoint_flags([100, 20, 90], threshold=30)
    assert list(flags) == [False, True, False]


def test_strictly_below_threshold_is_the_boundary() -> None:
    assert not ratcheted_flags([50, 50], threshold=50).any()
    assert ratcheted_flags([49.9], threshold=50).all()


# ---------------------------------------------------------------------------------
# build_flag_frame
# ---------------------------------------------------------------------------------


def test_flag_frame_covers_every_well_timepoint_and_threshold() -> None:
    table = _cell_population({"E07": [100, 50, 10], "M11": [80, 40, 8]})
    peaks = {"E07": 100.0, "M11": 80.0}
    flags = build_flag_frame(table, peaks, dmso_well="M11")
    # 2 wells x 3 timepoints x (4 relative + 1 absolute) thresholds
    assert len(flags) == 2 * 3 * 5
    assert set(flags["threshold"]) == {
        "rel_0.5",
        "rel_0.25",
        "rel_0.1",
        "rel_0.05",
        ABSOLUTE_FLOOR_LABEL,
    }


def test_dmso_flags_are_broadcast_to_every_well() -> None:
    """A healthy well still inherits its experiment's degraded DMSO reference."""
    table = _cell_population({"E07": [100, 100, 100], "M11": [100, 10, 10]})
    peaks = {"E07": 100.0, "M11": 100.0}
    flags = build_flag_frame(table, peaks, dmso_well="M11")
    healthy = flags[(flags["well"] == "E07") & (flags["threshold"] == "rel_0.5")]
    assert not healthy["own_flagged"].any()  # never collapses on its own
    assert list(healthy["dmso_flagged"]) == [False, True, True]
    assert list(healthy["any_flagged"]) == [False, True, True]


def test_missing_dmso_well_raises_rather_than_silently_zeroing() -> None:
    table = _cell_population({"E07": [100, 50, 10]})
    with pytest.raises(KeyError, match="not present"):
        build_flag_frame(table, {"E07": 100.0}, dmso_well="M11")


def test_missing_peak_raises_rather_than_reading_as_never_flagged() -> None:
    """A well absent from the summary is "unknown", not "healthy".

    Defaulting it to unflagged would silently inflate every survival figure, and the
    t_cross crosscheck cannot catch it because it joins against the same summary.
    """
    table = _cell_population({"E07": [100, 50, 10], "M11": [100, 90, 80]})
    with pytest.raises(KeyError, match="no usable peak_n_cells"):
        build_flag_frame(table, {"M11": 100.0}, dmso_well="M11")


@pytest.mark.parametrize("bad_peak", [float("nan"), 0.0, -5.0, None])
def test_unusable_peak_values_are_rejected(bad_peak: object) -> None:
    """NaN, zero and negative peaks cannot anchor a relative threshold."""
    table = _cell_population({"E07": [100, 50, 10], "M11": [100, 90, 80]})
    with pytest.raises(KeyError, match="no usable peak_n_cells"):
        build_flag_frame(table, {"E07": bad_peak, "M11": 100.0}, dmso_well="M11")


def test_the_error_names_the_offending_wells() -> None:
    table = _cell_population(
        {"E07": [100, 50, 10], "E08": [100, 50, 10], "M11": [100, 90, 80]}
    )
    with pytest.raises(KeyError) as excinfo:
        build_flag_frame(table, {"M11": 100.0}, dmso_well="M11")
    assert "E07" in str(excinfo.value)
    assert "E08" in str(excinfo.value)


def test_absolute_floor_is_independent_of_the_peak() -> None:
    """A well that never collapses relatively can still trip the floor, and vice
    versa."""
    table = _cell_population({"E07": [20, 19, 18], "M11": [1000, 900, 800]})
    peaks = {"E07": 20.0, "M11": 1000.0}
    flags = build_flag_frame(table, peaks, dmso_well="M11", absolute_floor=30)
    low = flags[(flags["well"] == "E07") & (flags["threshold"] == ABSOLUTE_FLOOR_LABEL)]
    assert low["own_flagged"].all()  # always under 30 cells
    relative = flags[(flags["well"] == "E07") & (flags["threshold"] == "rel_0.5")]
    assert not relative["own_flagged"].any()  # but never halves its own peak


# ---------------------------------------------------------------------------------
# summarise_survival / aggregate_by_threshold
# ---------------------------------------------------------------------------------


def test_survival_counts_and_first_flagged_timepoint() -> None:
    table = _cell_population({"E07": [100, 90, 40, 30], "M11": [100, 90, 80, 70]})
    peaks = {"E07": 100.0, "M11": 100.0}
    flags = build_flag_frame(table, peaks, dmso_well="M11")
    survival = summarise_survival(flags)
    row = survival[
        (survival["well"] == "E07") & (survival["threshold"] == "rel_0.5")
    ].iloc[0]
    assert row["n_timepoints"] == 4
    assert row["n_flagged_own"] == 2  # ti 21 and 31
    assert row["frac_kept_own"] == pytest.approx(0.5)
    assert row["first_flagged_ti"] == 21


def test_first_flagged_ti_is_none_when_a_well_never_crosses() -> None:
    table = _cell_population({"E07": [100, 99, 98], "M11": [100, 99, 98]})
    flags = build_flag_frame(table, {"E07": 100.0, "M11": 100.0}, dmso_well="M11")
    survival = summarise_survival(flags)
    row = survival[
        (survival["well"] == "E07") & (survival["threshold"] == "rel_0.5")
    ].iloc[0]
    assert row["first_flagged_ti"] is None
    assert row["frac_kept_own"] == pytest.approx(1.0)


def test_n_dmso_only_isolates_the_marginal_cost_of_the_dmso_condition() -> None:
    """Counts timepoints lost *purely* because the reference degraded."""
    table = _cell_population({"E07": [100, 100, 100], "M11": [100, 10, 10]})
    flags = build_flag_frame(table, {"E07": 100.0, "M11": 100.0}, dmso_well="M11")
    survival = summarise_survival(flags)
    healthy = survival[
        (survival["well"] == "E07") & (survival["threshold"] == "rel_0.5")
    ].iloc[0]
    assert healthy["n_flagged_own"] == 0
    assert healthy["n_dmso_only"] == 2
    assert healthy["frac_kept_own"] == pytest.approx(1.0)
    assert healthy["frac_kept_any"] == pytest.approx(1 / 3)


def test_aggregate_reports_the_dmso_share_of_total_loss() -> None:
    table = _cell_population({"E07": [100, 100, 100], "M11": [100, 10, 10]})
    flags = build_flag_frame(table, {"E07": 100.0, "M11": 100.0}, dmso_well="M11")
    aggregate = aggregate_by_threshold(summarise_survival(flags))
    row = aggregate[aggregate["threshold"] == "rel_0.5"].iloc[0]
    assert row["n_wells"] == 2
    assert row["total_timepoints"] == 6
    # E07 loses 2 to the DMSO condition; M11 loses its own 2 (not "dmso only").
    assert row["total_dmso_only"] == 2
    assert row["pct_lost_to_dmso_condition"] == pytest.approx(100.0 * 2 / 6)


def test_lower_thresholds_keep_at_least_as_much_data() -> None:
    """Survival must be monotonic in the threshold -- a sanity property of the gate."""
    table = _cell_population({"E07": [1000, 400, 200, 100, 60, 60], "M11": [100] * 6})
    flags = build_flag_frame(table, {"E07": 1000.0, "M11": 100.0}, dmso_well="M11")
    survival = summarise_survival(flags)
    e07 = survival[survival["well"] == "E07"].set_index("threshold")
    kept = {t: e07.loc[t, "frac_kept_own"] for t in ("rel_0.5", "rel_0.25", "rel_0.1")}
    assert kept["rel_0.5"] < kept["rel_0.25"] < kept["rel_0.1"]
    assert kept["rel_0.5"] == pytest.approx(1 / 6)
    assert kept["rel_0.1"] == pytest.approx(4 / 6)


# ---------------------------------------------------------------------------------
# crosscheck / keep_window
# ---------------------------------------------------------------------------------


def test_half_peak_gate_reproduces_t_cross_peak() -> None:
    """By construction the 0.5 gate and Step 1's t_cross_peak are the same crossing."""
    table = _cell_population({"E07": [100, 90, 40, 30], "M11": [100, 90, 80, 70]})
    flags = build_flag_frame(table, {"E07": 100.0, "M11": 100.0}, dmso_well="M11")
    survival = summarise_survival(flags)
    summary = pd.DataFrame(
        {
            "well": ["E07", "M11"],
            "t_cross_peak": [21.0, None],
            "collapse_shape": ["sharp step", "no material collapse"],
        }
    )
    crosscheck = crosscheck_against_t_cross(survival, summary)
    assert crosscheck["agrees"].all()


def test_crosscheck_flags_a_drift_between_the_two_tables() -> None:
    table = _cell_population({"E07": [100, 90, 40, 30], "M11": [100, 90, 80, 70]})
    flags = build_flag_frame(table, {"E07": 100.0, "M11": 100.0}, dmso_well="M11")
    survival = summarise_survival(flags)
    summary = pd.DataFrame(
        {
            "well": ["E07", "M11"],
            "t_cross_peak": [999.0, None],  # deliberately wrong
            "collapse_shape": ["sharp step", "no material collapse"],
        }
    )
    crosscheck = crosscheck_against_t_cross(survival, summary)
    assert not crosscheck.loc[crosscheck["well"] == "E07", "agrees"].iloc[0]


def test_keep_window_returns_the_last_surviving_timepoint() -> None:
    table = _cell_population({"E07": [100, 90, 40, 30], "M11": [100, 90, 80, 70]})
    flags = build_flag_frame(table, {"E07": 100.0, "M11": 100.0}, dmso_well="M11")
    survival = summarise_survival(flags)
    assert keep_window(survival, flags, "E07", "rel_0.5") == 11
    assert keep_window(survival, flags, "M11", "rel_0.5") == 31


def test_keep_window_is_none_when_a_well_is_flagged_throughout() -> None:
    table = _cell_population({"E07": [10, 9, 8], "M11": [100, 90, 80]})
    flags = build_flag_frame(table, {"E07": 100.0, "M11": 100.0}, dmso_well="M11")
    survival = summarise_survival(flags)
    assert keep_window(survival, flags, "E07", "rel_0.5") is None
