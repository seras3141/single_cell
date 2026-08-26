"""Unit tests for src/feature_analysis/feature_trajectories.py (Phase 1)."""
import numpy as np
import pandas as pd
import pytest

from src.feature_analysis import (
    BIOLOGICAL_FEATURES,
    FEATURE_GROUP,
    GABOR_FEATURES,
    SPATIAL_FEATURES,
    collapse_to_cell,
    compute_divergence_from_dmso,
    compute_drift,
    compute_trajectories,
    load_features,
)
from src.feature_to_mcherry.pre_collapse import (
    VERDICT_ESTIMABLE,
    VERDICT_NOT_ESTIMABLE,
)


def _rows(well, t, cell, area, z_values):
    """One incarta-style row per z-slice for a cell."""
    return [
        dict(sample_id=well, timepoint=t, cell_id=cell, z_index=z, area=a, mean_intensity=100.0)
        for z, a in z_values
    ]


@pytest.fixture
def cells_df():
    """Two wells, rising area over time in the drug well; DMSO flat."""
    rows = []
    # DMSO N11: flat area ~15/30 across 3 timepoints, 2 cells each, cell 1 has two z-slices
    for t in (1, 11, 21):
        rows += _rows("N11", t, f"{t}a", None, [(1, 10.0), (2, 20.0)])  # median over z = 15
        rows += _rows("N11", t, f"{t}b", None, [(1, 30.0)])
    # Drug A02: area climbs 100 -> 200 -> 300
    for t, base in ((1, 100.0), (11, 200.0), (21, 300.0)):
        rows += _rows("A02", t, f"{t}c", None, [(1, base), (2, base + 20)])
        rows += _rows("A02", t, f"{t}d", None, [(1, base + 10)])
    return pd.DataFrame(rows)


def test_group_maps_and_biological_set():
    assert set(GABOR_FEATURES).isdisjoint(BIOLOGICAL_FEATURES)  # gabor dropped
    assert set(SPATIAL_FEATURES).isdisjoint(BIOLOGICAL_FEATURES)  # spatial excluded from biology
    assert FEATURE_GROUP["gabor_mean"] == "gabor"
    assert FEATURE_GROUP["centroid_x"] == "spatial"
    assert FEATURE_GROUP["area"] == "shape"
    assert FEATURE_GROUP["mean_intensity"] == "intensity"


def test_collapse_to_cell_median_over_z(cells_df):
    cells = collapse_to_cell(cells_df, ["area"])
    # N11 cell "1a" had z-slices 10,20 -> median 15
    v = cells[(cells.sample_id == "N11") & (cells.timepoint == "1") & (cells.cell_id == "1a")]
    assert v.iloc[0]["area"] == pytest.approx(15.0)
    # one row per (well, timepoint, cell)
    assert len(cells) == cells[["sample_id", "timepoint", "cell_id"]].drop_duplicates().shape[0]


def test_trajectories_median_iqr(cells_df):
    cells = collapse_to_cell(cells_df, ["area"])
    traj = compute_trajectories(cells, ["area"], dmso_well="N11")
    n11_t1 = traj[(traj.sample_id == "N11") & (traj.timepoint == "1") & (traj.feature == "area")].iloc[0]
    # per-cell area values at N11 t1 = {15, 30}
    assert n11_t1["median"] == pytest.approx(22.5)
    assert n11_t1["n_cells"] == 2
    assert n11_t1["group"] == "shape"
    assert bool(n11_t1["is_dmso"]) is True
    assert not traj[traj.sample_id == "A02"]["is_dmso"].any()


def test_drift_direction_and_estimability(cells_df):
    cells = collapse_to_cell(cells_df, ["area"])
    traj = compute_trajectories(cells, ["area"], dmso_well="N11")
    drift_est = compute_drift(traj, dmso_n_timepoints_pre_cross=12)
    drift_not = compute_drift(traj, dmso_n_timepoints_pre_cross=2)
    a02 = drift_est[(drift_est.sample_id == "A02") & (drift_est.feature == "area")].iloc[0]
    assert a02["spearman_rho"] == pytest.approx(1.0)  # monotonic rise
    assert a02["theil_slope"] > 0
    assert a02["late_median"] > a02["early_median"]
    assert drift_est["estimable"].iloc[0] == VERDICT_ESTIMABLE      # 12 >= 10
    assert drift_not["estimable"].iloc[0] == VERDICT_NOT_ESTIMABLE  # 2 < 5


def test_divergence_from_dmso(cells_df):
    cells = collapse_to_cell(cells_df, ["area"])
    div = compute_divergence_from_dmso(cells, "N11", ["area"], n_floor=30)
    # DMSO must not appear as a drug well
    assert "N11" not in set(div["sample_id"])
    a02_t1 = div[(div.sample_id == "A02") & (div.timepoint == "1") & (div.feature == "area")].iloc[0]
    assert a02_t1["std_shift"] > 0            # drug area >> DMSO area
    assert a02_t1["low_confidence"]           # n well/dmso are tiny (< 30)
    assert 0.0 <= a02_t1["ks_stat"] <= 1.0
    # an overall row exists per (well, timepoint)
    assert ((div.feature == "__overall__") & (div.sample_id == "A02")).any()


def test_load_features_from_dir(tmp_path):
    for i, well in enumerate(("N11", "A02")):
        pd.DataFrame([
            dict(cell_id="c1", sample_id=well, timepoint=1, z_index=1,
                 area=10.0 + i, mean_intensity=5.0, gabor_mean=999.0, centroid_x=1.0),
        ]).to_csv(tmp_path / f"pMF5V1_{well}_t1_z1_BF_features.csv", index=False)
    df = load_features(tmp_path, features=["area", "mean_intensity"])
    assert set(df.columns) >= {"cell_id", "sample_id", "timepoint", "z_index", "area", "mean_intensity", "ti"}
    assert "gabor_mean" not in df.columns  # not requested -> not loaded
    assert "centroid_x" not in df.columns
    assert set(df["ti"]) == {1}
