"""Tests for feature_to_mcherry.report_figures.group_b."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.feature_to_mcherry.report_figures import group_b
from src.feature_to_mcherry.report_figures.config import ReportFiguresConfig


def _make_config(tmp_path: Path, experiments) -> ReportFiguresConfig:
    return ReportFiguresConfig(
        experiments=experiments,
        scportrait_experiments=[],
        a5_experiments=[],
        feature_to_mcherry_dir=str(tmp_path / "feature_to_mcherry"),
        informativeness_dir=str(tmp_path / "morphology_informativeness"),
        data_quality_dir=str(tmp_path / "data_quality"),
        data_root=str(tmp_path / "data_root"),
        output_dir=str(tmp_path / "out"),
        formats=["png"],
    )


def _write_univariate(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for feature in ("area", "gabor_mean"):
        for target in ("percentile_75", "percentile_90"):
            rows.append(
                {
                    "feature": feature,
                    "target": target,
                    "scope": "pooled",
                    "group_id": "",
                    "rho": 0.5 if feature == "area" else 0.05,
                    "pvalue": 0.0,
                    "n": 100,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_figure_b1_generated_writes_per_experiment_heatmaps(tmp_path: Path) -> None:
    config = _make_config(tmp_path, ["Ew2-1", "Ew2-2"])
    for exp in ("Ew2-1", "Ew2-2"):
        _write_univariate(
            Path(config.informativeness_dir) / exp / "univariate_correlations.csv"
        )

    status = group_b.figure_b1(config)

    assert status.status == "generated"
    assert len(status.output_paths) >= 2
    for path in status.output_paths:
        assert Path(path).exists()


def test_figure_b1_blocked_when_nothing_present(tmp_path: Path) -> None:
    config = _make_config(tmp_path, ["Ew2-1"])

    status = group_b.figure_b1(config)

    assert status.status == "blocked"


def _write_floor_metrics_with_variants(
    path: Path, with_r2: float, without_r2: float
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "variant": "with_suspect",
            "model": "gradient_boosting",
            "backend": "lightgbm",
            "target": "percentile_75",
            "tau": 0.75,
            "mae": 8.0,
            "r2": with_r2,
            "pinball_loss": 4.0,
        },
        {
            "variant": "without_suspect",
            "model": "gradient_boosting",
            "backend": "lightgbm",
            "target": "percentile_75",
            "tau": 0.75,
            "mae": 8.5,
            "r2": without_r2,
            "pinball_loss": 4.2,
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def test_figure_b2_generated(tmp_path: Path) -> None:
    config = _make_config(tmp_path, ["Ew2-1"])
    _write_floor_metrics_with_variants(
        Path(config.informativeness_dir) / "Ew2-1" / "floor_metrics.csv", 0.33, 0.30
    )

    status = group_b.figure_b2(config)

    assert status.status == "generated"
    assert Path(status.output_paths[0]).exists()


def test_figure_b3_generated_from_feature_importances(tmp_path: Path) -> None:
    config = _make_config(tmp_path, ["Ew2-1"])
    path = Path(config.informativeness_dir) / "Ew2-1" / "feature_importances.csv"
    path.parent.mkdir(parents=True)
    rows = []
    for target in ("percentile_75", "percentile_90"):
        for feature, imp in [("area", 100.0), ("gabor_mean", 5.0)]:
            rows.append(
                {
                    "target": target,
                    "feature": feature,
                    "importance_mean": imp,
                    "importance_std": 1.0,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)

    status = group_b.figure_b3(config)

    assert status.status == "generated"
    assert Path(status.output_paths[0]).exists()


def test_figure_b3_blocked_without_feature_importances(tmp_path: Path) -> None:
    config = _make_config(tmp_path, ["Ew2-1"])

    status = group_b.figure_b3(config)

    assert status.status == "blocked"


def _write_oof_predictions(
    path: Path, n: int, seed: int, with_outlier: bool = False
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    y_true = rng.uniform(0, 100, size=n)
    y_pred = y_true + rng.normal(0, 5, size=n)
    if with_outlier:
        # A single extreme value, matching the real data's few 10-20x-scale
        # mCherry outliers -- must not blow out the plotted axis range.
        y_true = np.append(y_true, 10_000.0)
        y_pred = np.append(y_pred, 200.0)
    pd.DataFrame(
        {
            "group_id": ["A01"] * len(y_true),
            "target_name": ["percentile_95"] * len(y_true),
            "y_true": y_true,
            "y_pred": y_pred,
        }
    ).to_csv(path, index=False)


def test_figure_b4_generated_with_two_experiments(tmp_path: Path) -> None:
    config = _make_config(tmp_path, ["Ew2-1", "Ew2-2"])
    for exp, seed, r2 in [("Ew2-1", 0, 0.15), ("Ew2-2", 1, 0.18)]:
        exp_dir = Path(config.feature_to_mcherry_dir) / exp
        exp_dir.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "model": "ridge",
                    "target": "percentile_95",
                    "tau": 0.95,
                    "mae": 10.0,
                    "r2": r2,
                    "pinball_loss": 5.0,
                    "quantile_crossing_rate": 0.0,
                }
            ]
        ).to_csv(exp_dir / "baseline_ladder.csv", index=False)
        _write_oof_predictions(exp_dir / "oof_predictions.csv", 50, seed)

    status = group_b.figure_b4(config)

    assert status.status == "generated"
    assert Path(status.output_paths[0]).exists()


def test_figure_b4_blocked_without_oof_predictions(tmp_path: Path) -> None:
    config = _make_config(tmp_path, ["Ew2-1"])

    status = group_b.figure_b4(config)

    assert status.status == "blocked"


def test_figure_b4_axis_limits_are_robust_to_a_single_extreme_outlier(
    tmp_path: Path, monkeypatch
) -> None:
    # Regression test: one extreme mCherry outlier (real data has a handful, up to
    # ~10x the bulk of the distribution) must not stretch the axis so far that the
    # bulk of the fit becomes an unreadable sliver in a corner.
    config = _make_config(tmp_path, ["Ew2-1"])
    exp_dir = Path(config.feature_to_mcherry_dir) / "Ew2-1"
    exp_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "model": "ridge",
                "target": "percentile_95",
                "tau": 0.95,
                "mae": 10.0,
                "r2": 0.15,
                "pinball_loss": 5.0,
                "quantile_crossing_rate": 0.0,
            }
        ]
    ).to_csv(exp_dir / "baseline_ladder.csv", index=False)
    _write_oof_predictions(
        exp_dir / "oof_predictions.csv", 200, seed=0, with_outlier=True
    )

    captured_figs = []
    real_save_fig = group_b.save_fig

    def _capturing_save_fig(fig, *args, **kwargs):
        captured_figs.append(fig)
        return real_save_fig(fig, *args, **kwargs)

    monkeypatch.setattr(group_b, "save_fig", _capturing_save_fig)

    status = group_b.figure_b4(config)

    assert status.status == "generated"
    assert len(captured_figs) == 1
    ax_scatter = captured_figs[0].axes[0]
    xlim = ax_scatter.get_xlim()
    # The outlier (y_true=10_000) must be clipped out of the visible range --
    # otherwise the bulk of the data (y_true in [0, 100]) would be invisible.
    assert xlim[1] < 1000


# --- DMSO-normalized runs (z_-prefixed targets) -----------------------------------


def _write_floor_metrics_for_target(path: Path, target: str) -> None:
    """A minimal two-variant floor_metrics.csv for an arbitrary target name."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "variant": variant,
            "model": "gradient_boosting",
            "backend": "lightgbm",
            "target": target,
            "tau": 0.75,
            "mae": 8.0,
            "r2": r2,
            "pinball_loss": 4.0,
        }
        for variant, r2 in (("with_suspect", 0.20), ("without_suspect", 0.18))
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def test_figure_b2_handles_dmso_normalized_targets(tmp_path: Path) -> None:
    """Normalized targets must still draw; they previously selected zero percentiles.

    ``plt.subplots(1, 0)`` is an error, not an empty grid, so this crashed outright.
    """
    config = _make_config(tmp_path, ["Ew2-1"])
    _write_floor_metrics_for_target(
        Path(config.informativeness_dir) / "Ew2-1" / "floor_metrics.csv",
        "z_percentile_75",
    )

    status = group_b.figure_b2(config)

    assert status.status == "generated"
    assert status.output_paths


def test_figure_b2_blocks_when_no_percentile_targets(tmp_path: Path) -> None:
    config = _make_config(tmp_path, ["Ew2-1"])
    _write_floor_metrics_for_target(
        Path(config.informativeness_dir) / "Ew2-1" / "floor_metrics.csv", "area"
    )

    status = group_b.figure_b2(config)

    assert status.status == "blocked"
    assert "No percentile targets" in (status.note or "")
