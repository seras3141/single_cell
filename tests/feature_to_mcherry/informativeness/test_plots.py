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
    )

    for paths in figures.values():
        for path in paths:
            assert path.exists()
            assert path.stat().st_size > 0

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
    )

    html_files = list((tmp_path / "figures").glob("*.html"))
    assert len(html_files) == 0

    png_files = list((tmp_path / "figures").glob("*.png"))
    assert len(png_files) >= 5
