"""Tests for feature_to_mcherry.data.normalize (DMSO target z-normalization)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_to_mcherry.data.normalize import (
    MAD_SCALE,
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
