"""Unit tests for Phase 2: heterogeneity (#5), integrity (#6), confluence-onset (#4)."""
import numpy as np
import pandas as pd
import pytest

from src.feature_analysis import (
    compute_confluence_onset,
    compute_heterogeneity,
    compute_heterogeneity_trend,
    compute_integrity_flags,
)
from src.feature_analysis.integrity import (
    STATUS_OK,
    STATUS_SKIPPED_LOW_N,
    STATUS_SKIPPED_ZERO_MAD,
)
from src.feature_to_mcherry.pre_collapse import VERDICT_ESTIMABLE


def _cells(rows):
    df = pd.DataFrame(rows)
    df["ti"] = df["timepoint"].astype(int)
    df["timepoint"] = df["timepoint"].astype(str)
    return df


# ----------------------------- #5 heterogeneity -----------------------------
@pytest.fixture
def het_cells():
    rows = []
    # area: median ~10 (rIQR valid); skewness: median ~0 (rIQR invalid)
    for t in (1, 11, 21):
        for cid, a in enumerate([9.0, 10.0, 11.0, 10.5, 9.5]):
            rows.append(dict(sample_id="N11", timepoint=t, cell_id=f"{t}_{cid}",
                             area=a + 0.3 * t, skewness=(a - 10.0) * 0.01))
    return _cells(rows)


def test_heterogeneity_riqr_guard_and_overall(het_cells):
    het = compute_heterogeneity(het_cells, ["area", "skewness"], n_floor=3)
    area = het[(het.feature == "area") & (het.timepoint == "1")].iloc[0]
    assert area["riqr_valid"] and np.isfinite(area["rIQR"])
    skew = het[(het.feature == "skewness") & (het.timepoint == "1")].iloc[0]
    assert not skew["riqr_valid"] and np.isnan(skew["rIQR"])  # median ~0 -> guarded
    ov = het[het.feature == "__overall__"]
    assert len(ov) == 3 and ov["rIQR"].notna().all()  # overall from the valid feature (area)


def test_heterogeneity_trend(het_cells):
    het = compute_heterogeneity(het_cells, ["area", "skewness"], n_floor=3)
    tr = compute_heterogeneity_trend(het, dmso_n_timepoints_pre_cross=12)
    assert tr["estimable"].iloc[0] == VERDICT_ESTIMABLE
    assert (tr.feature == "area").any()
    assert "skewness" not in set(tr.feature)  # invalid rIQR -> excluded from trend


# ----------------------------- #6 integrity -----------------------------
@pytest.fixture
def jump_cells():
    rows = []
    base = {1: 10.0, 11: 10.0, 21: 10.0, 31: 100.0}  # jump between t21 and t31
    for t, m in base.items():
        for cid, d in enumerate([-0.5, -0.2, 0.0, 0.2, 0.5]):
            rows.append(dict(sample_id="W", timepoint=t, cell_id=f"{t}_{cid}",
                             area=m + d, const=5.0))
    return _cells(rows)


def test_integrity_flags_a_real_jump(jump_cells):
    out = compute_integrity_flags(jump_cells, ["area"], n_floor=3)
    stable = out[(out.feature == "area") & (out.ti == 1)].iloc[0]  # t1->t11
    assert not stable["flagged"] and stable["status"] == STATUS_OK
    jump = out[(out.feature == "area") & (out.ti == 21)].iloc[0]   # t21->t31
    assert jump["median_jump"] and jump["shape_jump"] and jump["flagged"]
    assert abs(jump["robust_z"]) >= 3.5 and jump["ks_stat"] >= 0.3


def test_integrity_zero_mad_and_low_n(jump_cells):
    out = compute_integrity_flags(jump_cells, ["area", "const"], n_floor=3)
    const = out[out.feature == "const"]
    assert (const["status"] == STATUS_SKIPPED_ZERO_MAD).all()  # constant -> no within-t spread
    assert not const["flagged"].any()
    hi = compute_integrity_flags(jump_cells, ["area"], n_floor=100)  # every tp has 5 cells
    assert (hi["status"] == STATUS_SKIPPED_LOW_N).all() and not hi["flagged"].any()


# ----------------------------- #4 confluence-onset -----------------------------
def test_confluence_onset_three_signals_and_consensus():
    # traj: one well, one feature, rising after the early window
    tis = [1, 11, 21, 31, 41, 51]
    med = {1: 10, 11: 10, 21: 10, 31: 20, 41: 20, 51: 20}  # moves >1 IQR (iqr=2) at t31, sustained
    traj = pd.DataFrame([
        dict(sample_id="W", feature="area", ti=t, median=med[t], iqr=2.0) for t in tis
    ])
    cov = pd.DataFrame([
        dict(sample_id="W", ti=t, coverage_fraction=c)
        for t, c in zip(tis, [0.2, 0.2, 0.2, 0.2, 0.05, 0.05])  # drops <0.1 at t41, sustained
    ])
    summary = pd.DataFrame([
        dict(experiment="TESTEXP", well="W", t_cross_peak=21, n_timepoints_pre_cross=np.nan, is_dmso=False),
        dict(experiment="TESTEXP", well="N11", t_cross_peak=41, n_timepoints_pre_cross=12, is_dmso=True),
    ])
    out = compute_confluence_onset(traj, cov, summary, "TESTEXP", dmso_well="N11", features=["area"])
    w = out[out.sample_id == "W"].iloc[0]
    assert w["onset_pop_ti"] == 21
    assert w["onset_cov_ti"] == 41
    assert w["onset_drift_ti"] == 31
    assert w["onset_consensus_ti"] == 31  # median(21,41,31)
    assert w["n_signals"] == 3 and w["disagreement_span"] == 20
    assert w["estimable"] == VERDICT_ESTIMABLE  # from DMSO n_pre_cross=12


def test_confluence_onset_requires_columns():
    traj = pd.DataFrame([dict(sample_id="W", feature="area", ti=1, median=10.0, iqr=2.0)])
    cov = pd.DataFrame([dict(sample_id="W", ti=1, coverage_fraction=0.2)])
    bad_summary = pd.DataFrame([dict(experiment="X", well="W")])  # missing t_cross_peak
    with pytest.raises(ValueError, match="summary is missing required columns"):
        compute_confluence_onset(traj, cov, bad_summary, "X", dmso_well="N11", features=["area"])


def test_confluence_onset_requires_experiment_column():
    # Wells repeat across experiments; a summary without `experiment` must raise, not silently
    # mix rows from other experiments (the last-written well would win the population onset).
    traj = pd.DataFrame([dict(sample_id="W", feature="area", ti=1, median=10.0, iqr=2.0)])
    cov = pd.DataFrame([dict(sample_id="W", ti=1, coverage_fraction=0.2)])
    no_exp_summary = pd.DataFrame([
        dict(well="W", t_cross_peak=21),   # this experiment's well
        dict(well="W", t_cross_peak=99),   # a same-named well from a DIFFERENT experiment
    ])
    with pytest.raises(ValueError, match="summary is missing required columns.*experiment"):
        compute_confluence_onset(traj, cov, no_exp_summary, "X", dmso_well="N11", features=["area"])
