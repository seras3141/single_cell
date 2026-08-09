"""Unit tests for src/dataset_analysis/cell_population.py (segmentation-only)."""
import numpy as np
import pandas as pd
import pytest

from src.dataset_analysis.cell_population import (
    compute_cell_population,
    plot_population,
)


@pytest.fixture
def metrics_csv(tmp_path):
    """Synthetic instance_metrics: two wells, mixed z-slices and repeated cell_ids."""
    rows = [
        # A01 t1: cell_ids {1,2} on z1, {1,3} on z2 -> 3 distinct cells;
        #         z1 area sum=200, z2 area sum=200
        dict(sample_id="A01", timepoint=1, z_index=1, cell_id=1, area=100),
        dict(sample_id="A01", timepoint=1, z_index=1, cell_id=2, area=100),
        dict(sample_id="A01", timepoint=1, z_index=2, cell_id=1, area=100),
        dict(sample_id="A01", timepoint=1, z_index=2, cell_id=3, area=100),
        # A01 t11: one cell, one slice
        dict(sample_id="A01", timepoint=11, z_index=1, cell_id=1, area=50),
        # N11 (DMSO) t1: one cell
        dict(sample_id="N11", timepoint=1, z_index=1, cell_id=1, area=300),
        # extra mCherry-only column must be ignored
    ]
    df = pd.DataFrame(rows)
    df["mean_intensity"] = 999.0  # ignored by the reader
    p = tmp_path / "instance_metrics.csv"
    df.to_csv(p, index=False)
    return p


def test_n_cells_and_coverage(metrics_csv):
    out = compute_cell_population(metrics_csv, fov_pixels=1000, dmso_well="N11")

    a01_t1 = out[(out.sample_id == "A01") & (out.timepoint == 1)].iloc[0]
    assert a01_t1.n_cells == 3  # distinct cell_id across z
    assert a01_t1.coverage_fraction == pytest.approx(0.2)  # mean of 0.2, 0.2

    a01_t11 = out[(out.sample_id == "A01") & (out.timepoint == 11)].iloc[0]
    assert a01_t11.n_cells == 1
    assert a01_t11.coverage_fraction == pytest.approx(0.05)

    n11 = out[out.sample_id == "N11"].iloc[0]
    assert n11.coverage_fraction == pytest.approx(0.3)


def test_dmso_flag_and_ti(metrics_csv):
    out = compute_cell_population(metrics_csv, fov_pixels=1000, dmso_well="N11")
    assert out.loc[out.sample_id == "N11", "is_dmso"].all()
    assert not out.loc[out.sample_id == "A01", "is_dmso"].any()
    assert out["ti"].dtype.kind in "iu"  # integer timepoint
    assert set(out.loc[out.sample_id == "A01", "ti"]) == {1, 11}


def test_bad_fov_raises(metrics_csv):
    with pytest.raises(ValueError, match="fov_pixels"):
        compute_cell_population(metrics_csv, fov_pixels=0)


def test_missing_columns_raise(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"sample_id": ["A01"], "timepoint": [1]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        compute_cell_population(bad, fov_pixels=1000)


def test_plot_population_writes_png(metrics_csv, tmp_path):
    out = compute_cell_population(metrics_csv, fov_pixels=1000, dmso_well="N11")
    png = tmp_path / "n_cells.png"
    plot_population(out, "n_cells", png, title="test", dmso_well="N11")
    assert png.exists() and png.stat().st_size > 0
