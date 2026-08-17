"""Tests for feature_to_mcherry.informativeness.plots.

Runs headless (Agg backend forced in plots.py, no display required).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.feature_to_mcherry.informativeness import plots as plots_module
from src.feature_to_mcherry.informativeness.plots import write_figures


def _synthetic_bundle_inputs():
    feature_names = ["area", "perimeter"]
    target_names = ["percentile_75", "percentile_90"]
    n = 40
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 10, size=(n, 2))
    y = np.column_stack(
        [2 * X[:, 0] + rng.normal(0, 1, n), 3 * X[:, 1] + rng.normal(0, 1, n)]
    )

    # Two wells; timepoints span a small integer range (with repeats) so the
    # well-timepoint scatter has multi-well, multi-timepoint data to colour.
    groups = np.array(["A01" if i % 2 == 0 else "A02" for i in range(n)])
    timepoints = np.array([(i % 3) + 1 for i in range(n)])

    rows = []
    for target in target_names:
        for feature in feature_names:
            rows.append(
                {
                    "feature": feature,
                    "target": target,
                    "scope": "pooled",
                    "group_id": "",
                    "rho": 0.5,
                    "pvalue": 0.01,
                    "n": n,
                }
            )
            for group_id in ["A01", "A02"]:
                rows.append(
                    {
                        "feature": feature,
                        "target": target,
                        "scope": "per_group",
                        "group_id": group_id,
                        "rho": 0.4,
                        "pvalue": 0.05,
                        "n": n // 2,
                    }
                )
    univariate_df = pd.DataFrame(rows)

    floor_metrics_df = pd.DataFrame(
        [
            {
                "variant": "with_suspect",
                "model": "ridge",
                "backend": "sklearn",
                "target": target,
                "tau": tau,
                "mae": 1.0,
                "r2": 0.5,
                "pinball_loss": 0.2,
            }
            for target, tau in zip(target_names, [0.75, 0.90])
        ]
        + [
            {
                "variant": "with_suspect",
                "model": "gradient_boosting",
                "backend": "lightgbm",
                "target": target,
                "tau": tau,
                "mae": 0.9,
                "r2": 0.6,
                "pinball_loss": 0.18,
            }
            for target, tau in zip(target_names, [0.75, 0.90])
        ]
    )

    noise_ceiling_df = pd.DataFrame(
        [
            {
                "target": target,
                "method": "icc1_variance_decomposition",
                "ceiling": 0.7,
                "n_conditions": 2,
                "n_replicate_wells": 4,
                "reason": "",
            }
            for target in target_names
        ]
    )

    return (
        X,
        y,
        feature_names,
        target_names,
        univariate_df,
        floor_metrics_df,
        noise_ceiling_df,
        groups,
        timepoints,
    )


def test_write_figures_produces_nonempty_png_files(tmp_path: Path) -> None:
    (
        X,
        y,
        feature_names,
        target_names,
        univariate_df,
        floor_metrics_df,
        noise_ceiling_df,
        groups,
        timepoints,
    ) = _synthetic_bundle_inputs()

    figures = write_figures(
        tmp_path,
        univariate_df,
        X,
        y,
        feature_names,
        target_names,
        floor_metrics_df,
        noise_ceiling_df,
        top_k=2,
        groups=groups,
        timepoints=timepoints,
    )

    for paths in figures.values():
        for path in paths:
            assert path.exists()
            assert path.stat().st_size > 0

    # The new well-timepoint scatter key is present and non-empty (two wells, two
    # targets, up to top_k features each), rather than only checking global counts.
    assert "well_timepoint_scatter" in figures
    assert figures["well_timepoint_scatter"]
    well_timepoint_pngs = list((tmp_path / "figures").glob("well_timepoint_*.png"))
    assert well_timepoint_pngs

    png_files = list((tmp_path / "figures").glob("*.png"))
    assert len(png_files) >= 5


def test_write_figures_produces_html_when_plotly_available(tmp_path: Path) -> None:
    if not plots_module.HAVE_PLOTLY:
        pytest.skip("plotly not installed")

    (
        X,
        y,
        feature_names,
        target_names,
        univariate_df,
        floor_metrics_df,
        noise_ceiling_df,
        groups,
        timepoints,
    ) = _synthetic_bundle_inputs()

    write_figures(
        tmp_path,
        univariate_df,
        X,
        y,
        feature_names,
        target_names,
        floor_metrics_df,
        noise_ceiling_df,
        top_k=2,
        groups=groups,
        timepoints=timepoints,
    )

    html_files = list((tmp_path / "figures").glob("*.html"))
    assert len(html_files) >= 5


def test_write_figures_skips_html_when_plotly_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(plots_module, "HAVE_PLOTLY", False)

    (
        X,
        y,
        feature_names,
        target_names,
        univariate_df,
        floor_metrics_df,
        noise_ceiling_df,
        groups,
        timepoints,
    ) = _synthetic_bundle_inputs()

    write_figures(
        tmp_path,
        univariate_df,
        X,
        y,
        feature_names,
        target_names,
        floor_metrics_df,
        noise_ceiling_df,
        top_k=2,
        groups=groups,
        timepoints=timepoints,
    )

    html_files = list((tmp_path / "figures").glob("*.html"))
    assert len(html_files) == 0

    png_files = list((tmp_path / "figures").glob("*.png"))
    assert len(png_files) >= 5


def test_write_figures_returns_empty_well_timepoint_when_timepoints_none(
    tmp_path: Path,
) -> None:
    (
        X,
        y,
        feature_names,
        target_names,
        univariate_df,
        floor_metrics_df,
        noise_ceiling_df,
        groups,
        _timepoints,
    ) = _synthetic_bundle_inputs()

    figures = write_figures(
        tmp_path,
        univariate_df,
        X,
        y,
        feature_names,
        target_names,
        floor_metrics_df,
        noise_ceiling_df,
        top_k=2,
        groups=groups,
        timepoints=None,
    )

    # timepoints=None => key present but empty, and no files written.
    assert figures["well_timepoint_scatter"] == []
    assert list((tmp_path / "figures").glob("well_timepoint_*.png")) == []
    assert list((tmp_path / "figures").glob("well_timepoint_*.html")) == []


def _scatter_inputs(n: int = 30, seed: int = 0):
    """Minimal inputs for plot_feature_scatter_by_well_timepoint: 3 wells, a few
    timepoints, one feature, two targets."""
    feature_names = ["area"]
    target_names = ["percentile_75", "percentile_90"]
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 10, size=(n, 1))
    y = np.column_stack(
        [2 * X[:, 0] + rng.normal(0, 1, n), 3 * X[:, 0] + rng.normal(0, 1, n)]
    )
    groups = np.array(["A01", "A02", "A03"])[np.arange(n) % 3]
    timepoints = np.array([(i % 4) + 1 for i in range(n)])
    univariate_df = pd.DataFrame(
        [
            {"feature": "area", "target": target, "scope": "pooled", "rho": 0.6}
            for target in target_names
        ]
    )
    return X, y, groups, timepoints, feature_names, target_names, univariate_df


def test_select_wells_keeps_highest_count_wells() -> None:
    groups = np.array(["A"] * 5 + ["B"] * 4 + ["C"] * 3 + ["D"] * 2 + ["E"] * 1)

    kept = plots_module._select_wells(groups, max_wells=2)
    assert set(kept.tolist()) == {"A", "B"}

    all_wells = plots_module._select_wells(groups, max_wells=None)
    assert set(all_wells.tolist()) == {"A", "B", "C", "D", "E"}


def test_subsample_indices_caps_passthrough_and_is_deterministic() -> None:
    capped = plots_module._subsample_indices(
        np.arange(5000), 100, np.random.default_rng(0)
    )
    assert len(capped) == 100
    assert len(set(capped.tolist())) == 100

    passthrough = plots_module._subsample_indices(
        np.arange(50), 100, np.random.default_rng(0)
    )
    np.testing.assert_array_equal(passthrough, np.arange(50))

    a = plots_module._subsample_indices(np.arange(5000), 100, np.random.default_rng(1))
    b = plots_module._subsample_indices(np.arange(5000), 100, np.random.default_rng(1))
    np.testing.assert_array_equal(a, b)


def test_plot_feature_scatter_by_well_timepoint_writes_one_png_per_target(
    tmp_path: Path,
) -> None:
    X, y, groups, timepoints, feature_names, target_names, univariate_df = (
        _scatter_inputs()
    )

    paths = plots_module.plot_feature_scatter_by_well_timepoint(
        X,
        y,
        groups,
        timepoints,
        feature_names,
        target_names,
        univariate_df,
        tmp_path,
        top_k=1,
    )

    png_files = list(tmp_path.glob("well_timepoint_*.png"))
    assert len(png_files) == len(target_names)
    for path in png_files:
        assert path.stat().st_size > 0
    assert all(path in paths for path in png_files)


def test_plot_feature_scatter_by_well_timepoint_skips_html_when_plotly_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(plots_module, "HAVE_PLOTLY", False)

    X, y, groups, timepoints, feature_names, target_names, univariate_df = (
        _scatter_inputs()
    )

    plots_module.plot_feature_scatter_by_well_timepoint(
        X,
        y,
        groups,
        timepoints,
        feature_names,
        target_names,
        univariate_df,
        tmp_path,
        top_k=1,
    )

    assert list(tmp_path.glob("*.html")) == []
    assert len(list(tmp_path.glob("well_timepoint_*.png"))) == len(target_names)
