"""Tests for feature_to_mcherry.data.normalize (DMSO target z-normalization)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_to_mcherry.data.normalize import (
    CONFIDENCE_REASON_COLUMN,
    MAD_SCALE,
    apply_dmso_normalization,
    compute_confidence_flags,
    compute_dmso_reference,
    normalize_targets_to_dmso,
)


def _make_targets() -> pd.DataFrame:
    """Two timepoints; DMSO well M11 + drug well E07, 3 cells each per (well, t)."""
    rows = []
    # timepoint 1: DMSO values {10, 12, 14} -> median 12; drug shifted up
    # timepoint 2: DMSO values {20, 22, 24} -> median 22
    dmso_by_t = {"1": [10.0, 12.0, 14.0], "2": [20.0, 22.0, 24.0]}
    drug_by_t = {"1": [16.0, 18.0, 20.0], "2": [30.0, 32.0, 34.0]}
    for t, vals in dmso_by_t.items():
        for i, v in enumerate(vals):
            rows.append(
                dict(
                    sample_id="M11",
                    timepoint=t,
                    z_index="1",
                    cell_id=str(i),
                    percentile_90=v,
                )
            )
    for t, vals in drug_by_t.items():
        for i, v in enumerate(vals):
            rows.append(
                dict(
                    sample_id="E07",
                    timepoint=t,
                    z_index="1",
                    cell_id=str(i),
                    percentile_90=v,
                )
            )
    return pd.DataFrame(rows)


def test_reference_median_and_mad() -> None:
    ref = compute_dmso_reference(
        _make_targets(), "M11", target_columns=["percentile_90"]
    )
    ref = ref.set_index("timepoint")
    assert ref.loc["1", "median_percentile_90"] == 12.0
    assert ref.loc["2", "median_percentile_90"] == 22.0
    # MAD of {10,12,14} about median 12 = median(|{-2,0,2}|)=2 -> scaled 2*1.4826
    assert ref.loc["1", "mad_percentile_90"] == pytest.approx(2.0 * MAD_SCALE)
    assert bool(ref.loc["1", "valid"]) is True
    assert int(ref.loc["1", "n_dmso"]) == 3


def test_dmso_median_z_is_zero_per_timepoint() -> None:
    """DMSO's own median z must be ~0 at every timepoint by construction."""
    out = normalize_targets_to_dmso(
        _make_targets(), "M11", target_columns=["percentile_90"]
    )
    dmso = out[out["sample_id"] == "M11"]
    per_t_median = dmso.groupby("timepoint")["z_percentile_90"].median()
    assert per_t_median.abs().max() == pytest.approx(0.0, abs=1e-9)


def test_drug_z_matches_manual_per_timepoint() -> None:
    out = normalize_targets_to_dmso(
        _make_targets(), "M11", target_columns=["percentile_90"]
    )
    drug = out[out["sample_id"] == "E07"].set_index("timepoint")
    scale = 2.0 * MAD_SCALE  # both timepoints have DMSO MAD = 2*scale
    # t1: (16,18,20) vs median 12 ; t2: (30,32,34) vs median 22
    assert drug.loc["1", "z_percentile_90"].tolist() == pytest.approx(
        [(v - 12.0) / scale for v in (16.0, 18.0, 20.0)]
    )
    assert drug.loc["2", "z_percentile_90"].tolist() == pytest.approx(
        [(v - 22.0) / scale for v in (30.0, 32.0, 34.0)]
    )


def test_originals_untouched_and_rows_preserved() -> None:
    src = _make_targets()
    out = normalize_targets_to_dmso(src, "M11", target_columns=["percentile_90"])
    assert len(out) == len(src)
    assert "z_percentile_90" in out.columns
    # original column values and order unchanged
    pd.testing.assert_series_equal(out["percentile_90"], src["percentile_90"])


def test_zero_mad_timepoint_yields_nan() -> None:
    """A timepoint where all DMSO values are identical (MAD=0) is invalid -> NaN z."""
    df = _make_targets()
    df.loc[df["sample_id"] == "M11", "percentile_90"] = np.where(
        df.loc[df["sample_id"] == "M11", "timepoint"] == "1",
        50.0,  # t1 DMSO all identical -> MAD 0
        df.loc[df["sample_id"] == "M11", "percentile_90"],
    )
    ref = compute_dmso_reference(df, "M11", target_columns=["percentile_90"])
    assert bool(ref.set_index("timepoint").loc["1", "valid"]) is False
    out = normalize_targets_to_dmso(df, "M11", target_columns=["percentile_90"])
    # all cells at timepoint 1 -> NaN; timepoint 2 -> finite
    assert out[out["timepoint"] == "1"]["z_percentile_90"].isna().all()
    assert out[out["timepoint"] == "2"]["z_percentile_90"].notna().all()


def test_min_cells_marks_invalid() -> None:
    ref = compute_dmso_reference(
        _make_targets(), "M11", target_columns=["percentile_90"], min_cells=5
    )
    assert not ref["valid"].any()  # only 3 DMSO cells/timepoint < 5


def test_missing_dmso_well_raises() -> None:
    with pytest.raises(ValueError, match="not found"):
        compute_dmso_reference(_make_targets(), "Z99", target_columns=["percentile_90"])


def test_missing_column_raises() -> None:
    df = _make_targets().drop(columns=["percentile_90"])
    with pytest.raises(ValueError, match="missing required columns"):
        compute_dmso_reference(df, "M11", target_columns=["percentile_90"])


def _make_two_experiment_targets() -> pd.DataFrame:
    """Two experiments sharing DMSO well M11 but with DIFFERENT DMSO levels."""
    rows = []
    for exp, dmso_vals, drug in [
        ("A", [10.0, 12.0, 14.0], 16.0),
        ("B", [20.0, 22.0, 24.0], 26.0),
    ]:
        for i, v in enumerate(dmso_vals):
            rows.append(
                dict(
                    experiment=exp,
                    sample_id="M11",
                    timepoint="1",
                    z_index="1",
                    cell_id=str(i),
                    percentile_90=v,
                )
            )
        rows.append(
            dict(
                experiment=exp,
                sample_id="E07",
                timepoint="1",
                z_index="1",
                cell_id="0",
                percentile_90=drug,
            )
        )
    return pd.DataFrame(rows)


def test_experiment_column_scopes_reference() -> None:
    """With experiment_column, each experiment uses its OWN M11 (not pooled)."""
    df = _make_two_experiment_targets()
    out = normalize_targets_to_dmso(
        df, "M11", target_columns=["percentile_90"], experiment_column="experiment"
    )
    scale = 2.0 * MAD_SCALE  # both experiments' DMSO MAD = 2*scale
    za = out[(out["experiment"] == "A") & (out["sample_id"] == "E07")][
        "z_percentile_90"
    ].iloc[0]
    zb = out[(out["experiment"] == "B") & (out["sample_id"] == "E07")][
        "z_percentile_90"
    ].iloc[0]
    # A: (16-12)/scale ; B: (26-22)/scale -> both +4/scale (own-experiment baseline)
    assert za == pytest.approx(4.0 / scale)
    assert zb == pytest.approx(4.0 / scale)


def test_without_experiment_column_pools_shared_well() -> None:
    """Single-experiment contract: omitting experiment_column pools shared M11 wells."""
    df = _make_two_experiment_targets()
    ref = compute_dmso_reference(df, "M11", target_columns=["percentile_90"])
    # one row (grouped by timepoint only); median of pooled {10,12,14,20,22,24} = 17
    assert len(ref) == 1
    assert ref.iloc[0]["median_percentile_90"] == pytest.approx(17.0)


def test_partial_nan_dmso_uses_finite_mask() -> None:
    """A NaN among DMSO values must not NaN-poison the median (finite-mask)."""
    df = _make_targets()
    mask = (df["sample_id"] == "M11") & (df["timepoint"] == "1")
    df.loc[df[mask].index[1], "percentile_90"] = np.nan  # {10, NaN, 14}
    r1 = (
        compute_dmso_reference(df, "M11", target_columns=["percentile_90"])
        .set_index("timepoint")
        .loc["1"]
    )
    assert np.isfinite(r1["median_percentile_90"])
    assert r1["median_percentile_90"] == pytest.approx(12.0)  # median of {10, 14}
    assert bool(r1["valid"]) is True  # 2 finite values >= min_cells 2
    out = normalize_targets_to_dmso(df, "M11", target_columns=["percentile_90"])
    # every cell with a FINITE input gets a finite z (only the NaN-input cell stays NaN)
    t1 = out[out["timepoint"] == "1"]
    finite_input = t1[t1["percentile_90"].notna()]
    assert finite_input["z_percentile_90"].notna().all()
    assert int(t1["z_percentile_90"].isna().sum()) == 1  # just the NaN-input DMSO cell


def test_partial_nan_below_min_cells_invalid() -> None:
    """NaNs dropping finite count below min_cells -> invalid reference -> NaN z."""
    df = _make_targets()
    mask = (df["sample_id"] == "M11") & (df["timepoint"] == "1")
    df.loc[df[mask].index[:2], "percentile_90"] = np.nan  # only 1 finite left
    ref = compute_dmso_reference(df, "M11", target_columns=["percentile_90"])
    assert bool(ref.set_index("timepoint").loc["1", "valid"]) is False
    out = normalize_targets_to_dmso(df, "M11", target_columns=["percentile_90"])
    assert out[out["timepoint"] == "1"]["z_percentile_90"].isna().all()


def test_apply_disabled_is_passthrough() -> None:
    src = _make_targets()
    out, cols = apply_dmso_normalization(
        src, enabled=False, dmso_well=None, target_columns=["percentile_90"]
    )
    assert cols == ["percentile_90"]
    assert out is src  # unchanged object, no z_ columns added
    assert "z_percentile_90" not in out.columns


def test_apply_enabled_swaps_columns_and_drops_undefined() -> None:
    df = _make_targets()
    # make timepoint 1's DMSO degenerate (MAD=0) -> invalid ref -> those rows dropped
    df.loc[df["sample_id"] == "M11", "percentile_90"] = np.where(
        df.loc[df["sample_id"] == "M11", "timepoint"] == "1",
        50.0,
        df.loc[df["sample_id"] == "M11", "percentile_90"],
    )
    out, cols = apply_dmso_normalization(
        df, enabled=True, dmso_well="M11", target_columns=["percentile_90"]
    )
    assert cols == ["z_percentile_90"]
    assert "z_percentile_90" in out.columns
    assert out["z_percentile_90"].notna().all()  # undefined-z rows dropped
    assert (out["timepoint"] == "1").sum() == 0  # the invalid-ref timepoint is gone
    assert (out["timepoint"] == "2").sum() > 0


def test_apply_enabled_without_dmso_well_raises() -> None:
    with pytest.raises(ValueError, match="dmso_well"):
        apply_dmso_normalization(
            _make_targets(),
            enabled=True,
            dmso_well=None,
            target_columns=["percentile_90"],
        )


# ---------------------------------------------------------------------------
# Confidence gate (Step 3): three conditions over per-(well, timepoint) counts.
# ---------------------------------------------------------------------------


def _make_population(
    population: "dict[str, dict[str, int]]", slices_per_cell: int = 1
) -> pd.DataFrame:
    """Build a target table with an exact distinct-cell count per (well, timepoint).

    ``slices_per_cell`` replicates every cell across ``z_index`` values, reproducing the
    real row unit ``(cell, z_slice)`` so tests can tell distinct-cell counting apart
    from row counting.
    """
    rows = []
    for well, by_timepoint in population.items():
        for timepoint, n_cells in by_timepoint.items():
            for cell in range(n_cells):
                for z in range(slices_per_cell):
                    rows.append(
                        dict(
                            sample_id=well,
                            timepoint=timepoint,
                            z_index=str(z),
                            cell_id=f"c{cell}",
                            percentile_90=10.0 + cell,
                        )
                    )
    return pd.DataFrame(rows)


def test_gate_counts_distinct_cells_not_slice_rows() -> None:
    """3 cells x 10 slices is 3 cells, not 30 -- the slice-inflation trap."""
    targets = _make_population({"M11": {"1": 3}}, slices_per_cell=10)
    assert len(targets) == 30

    flags = compute_confidence_flags(
        targets, "M11", min_peak_fraction=None, absolute_floor=10
    )
    assert int(flags.loc[0, "n_cells"]) == 3
    # 3 < 10 -> flagged. Had it counted the 30 rows it would have passed.
    assert bool(flags.loc[0, "flag_absolute_floor"]) is True


def test_relative_condition_is_ratcheted_but_floor_is_not() -> None:
    """Relative stays flagged after an uptick; the floor re-evaluates per timepoint."""
    targets = _make_population(
        {
            "M11": {"1": 100, "2": 100, "3": 100},  # healthy reference throughout
            "E07": {"1": 100, "2": 5, "3": 50},
        }
    )
    flags = compute_confidence_flags(
        targets, "M11", min_peak_fraction=0.1, absolute_floor=30
    )
    well = flags[flags["sample_id"] == "E07"].set_index("timepoint")

    # peak 100 -> relative threshold 10.
    assert bool(well.loc["1", "flag_relative"]) is False
    assert bool(well.loc["2", "flag_relative"]) is True
    # t3 is back to 50, above the threshold, but the ratchet holds it flagged.
    assert bool(well.loc["3", "flag_relative"]) is True
    # The floor is NOT ratcheted: 5 < 30 fires, 50 >= 30 does not.
    assert bool(well.loc["2", "flag_absolute_floor"]) is True
    assert bool(well.loc["3", "flag_absolute_floor"]) is False


def test_healthy_well_is_flagged_when_the_dmso_reference_degrades() -> None:
    """Condition 3: no trustworthy baseline flags every well at that timepoint."""
    targets = _make_population(
        {
            "M11": {"1": 100, "2": 4},  # reference collapses at t2
            "E07": {"1": 100, "2": 100},  # drug well perfectly healthy
        }
    )
    flags = compute_confidence_flags(
        targets, "M11", min_peak_fraction=0.1, absolute_floor=30
    )
    drug = flags[flags["sample_id"] == "E07"].set_index("timepoint")

    assert bool(drug.loc["2", "flag_relative"]) is False
    assert bool(drug.loc["2", "flag_absolute_floor"]) is False
    assert bool(drug.loc["2", "flag_dmso_reference"]) is True
    assert bool(drug.loc["2", "low_confidence"]) is True
    assert "DMSO reference degraded" in drug.loc["2", CONFIDENCE_REASON_COLUMN]
    # ...and t1 is untouched.
    assert bool(drug.loc["1", "low_confidence"]) is False


def test_timepoints_are_ordered_numerically_not_lexically() -> None:
    """Labels are 1, 2, 11 -- a string sort would put 11 before 2 and mis-ratchet."""
    targets = _make_population({"M11": {"1": 100, "2": 100, "11": 5}})
    flags = compute_confidence_flags(
        targets, "M11", min_peak_fraction=0.1, absolute_floor=None
    ).set_index("timepoint")

    # True order is 1, 2, 11 with the collapse last, so only t11 is flagged.
    assert bool(flags.loc["1", "flag_relative"]) is False
    assert bool(flags.loc["2", "flag_relative"]) is False
    assert bool(flags.loc["11", "flag_relative"]) is True


def test_timepoint_missing_from_the_dmso_well_is_flagged_not_trusted() -> None:
    """An unmatched timepoint has no reference at all; it must not read as confident."""
    targets = _make_population({"M11": {"1": 100}, "E07": {"1": 100, "2": 100}})
    flags = compute_confidence_flags(
        targets, "M11", min_peak_fraction=0.1, absolute_floor=None
    )
    drug = flags[flags["sample_id"] == "E07"].set_index("timepoint")
    assert bool(drug.loc["2", "flag_dmso_reference"]) is True


def test_gate_is_off_by_default() -> None:
    """apply_dmso_normalization must not change behaviour for existing callers."""
    targets = _make_targets()
    ungated, columns = apply_dmso_normalization(
        targets,
        enabled=True,
        dmso_well="M11",
        target_columns=["percentile_90"],
    )
    assert columns == ["z_percentile_90"]
    # Every (well, timepoint) here has only 3 cells, so any active gate would fire.
    assert len(ungated) == len(targets)
    assert "low_confidence" not in ungated.columns


def test_gate_drops_low_confidence_rows_when_enabled() -> None:
    targets = _make_population(
        {
            "M11": {"1": 100, "2": 100},
            "E07": {"1": 100, "2": 4},
        }
    )
    gated, columns = apply_dmso_normalization(
        targets,
        enabled=True,
        dmso_well="M11",
        target_columns=["percentile_90"],
        min_peak_fraction=0.1,
        absolute_floor=30,
    )
    assert columns == ["z_percentile_90"]
    surviving = set(zip(gated["sample_id"].tolist(), gated["timepoint"].tolist()))
    assert ("E07", "2") not in surviving
    assert ("E07", "1") in surviving
    assert ("M11", "1") in surviving
    assert ("M11", "2") in surviving
    # The gate filters rows; it does not leak its bookkeeping columns downstream.
    assert "low_confidence" not in gated.columns


def test_relative_condition_can_never_flag_a_whole_trajectory() -> None:
    """A well's peak timepoint is never below a fraction of its own peak.

    So the ratchet alone cannot empty a table however strict the fraction -- the only
    route to an all-flagged reference is the absolute floor. Worth pinning: it is what
    makes the empty-gate error message's diagnosis correct.
    """
    targets = _make_population({"M11": {"1": 100, "2": 4, "3": 4}})
    flags = compute_confidence_flags(
        targets, "M11", min_peak_fraction=1.0, absolute_floor=None
    ).set_index("timepoint")
    assert bool(flags.loc["1", "flag_relative"]) is False  # the peak survives
    assert bool(flags.loc["2", "flag_relative"]) is True
    assert bool(flags.loc["3", "flag_relative"]) is True


def test_gate_that_discards_everything_raises() -> None:
    """A reference below the floor throughout flags every well; do not model nothing."""
    targets = _make_population({"M11": {"1": 5, "2": 4}, "E07": {"1": 100, "2": 100}})
    with pytest.raises(ValueError, match="discarded every row"):
        apply_dmso_normalization(
            targets,
            enabled=True,
            dmso_well="M11",
            target_columns=["percentile_90"],
            min_peak_fraction=None,
            absolute_floor=30,
        )


def test_missing_cell_id_raises_rather_than_counting_rows() -> None:
    targets = _make_population({"M11": {"1": 100}}).drop(columns=["cell_id"])
    with pytest.raises(ValueError, match="cell_id"):
        compute_confidence_flags(targets, "M11")


def test_absent_dmso_well_raises() -> None:
    targets = _make_population({"E07": {"1": 100}})
    with pytest.raises(ValueError, match="not found"):
        compute_confidence_flags(targets, "M11")


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.5])
def test_out_of_range_min_peak_fraction_raises(fraction: float) -> None:
    targets = _make_population({"M11": {"1": 100}})
    with pytest.raises(ValueError, match="min_peak_fraction"):
        compute_confidence_flags(targets, "M11", min_peak_fraction=fraction)


def test_reason_names_every_condition_that_fired() -> None:
    targets = _make_population({"M11": {"1": 100, "2": 2}})
    flags = compute_confidence_flags(
        targets, "M11", min_peak_fraction=0.1, absolute_floor=30
    ).set_index("timepoint")
    reason = flags.loc["2", CONFIDENCE_REASON_COLUMN]
    assert "0.1x peak" in reason
    assert "fewer than 30 cells" in reason
    assert "DMSO reference degraded" in reason
    assert flags.loc["1", CONFIDENCE_REASON_COLUMN] == ""


def test_peak_survives_the_undefined_z_drop() -> None:
    """The relative threshold must use the FULL trajectory's peak.

    ``apply_dmso_normalization`` drops undefined-z rows before gating. If the gate
    recomputed each well's peak from that filtered frame, a well whose peak timepoint
    had an invalid DMSO reference would get a lowered threshold and genuinely collapsed
    timepoints would survive. Here E07 peaks at t1 (100 cells) but t1's DMSO reference
    has zero MAD, so every t1 row is dropped; t3 (8 cells) must still be gated out on
    the true peak of 100 (threshold 10), not the post-drop peak of 50 (threshold 5).
    """
    rows = []
    # t1: DMSO values all identical -> MAD 0 -> invalid reference -> z NaN -> dropped.
    populations = {
        "M11": {"1": (100, 10.0), "2": (100, None), "3": (100, None)},
        "E07": {"1": (100, None), "2": (50, None), "3": (8, None)},
    }
    for well, by_timepoint in populations.items():
        for timepoint, (n_cells, constant) in by_timepoint.items():
            for cell in range(n_cells):
                rows.append(
                    dict(
                        sample_id=well,
                        timepoint=timepoint,
                        z_index="1",
                        cell_id=f"c{cell}",
                        percentile_90=constant if constant is not None else 10.0 + cell,
                    )
                )
    targets = pd.DataFrame(rows)

    gated, _ = apply_dmso_normalization(
        targets,
        enabled=True,
        dmso_well="M11",
        target_columns=["percentile_90"],
        min_peak_fraction=0.1,
        absolute_floor=None,
    )
    drug_timepoints = set(gated.loc[gated["sample_id"] == "E07", "timepoint"])
    assert "1" not in drug_timepoints  # dropped: undefined z, not the gate
    assert "2" in drug_timepoints
    # 8 < 0.1 x 100 -- only true when the peak comes from the full trajectory.
    assert "3" not in drug_timepoints


@pytest.mark.parametrize("floor", [0, -1])
def test_non_positive_absolute_floor_raises(floor: int) -> None:
    """A floor no count can fall below is vacuous; disabling has its own spelling."""
    targets = _make_population({"M11": {"1": 100}})
    with pytest.raises(ValueError, match="absolute_floor must be at least 1"):
        compute_confidence_flags(targets, "M11", absolute_floor=floor)
