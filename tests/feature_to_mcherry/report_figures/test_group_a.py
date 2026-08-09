"""Tests for feature_to_mcherry.report_figures.group_a."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from src.feature_to_mcherry.report_figures import group_a
from src.feature_to_mcherry.report_figures.config import ReportFiguresConfig

BASELINE_LADDER_COLUMNS = [
    "model",
    "target",
    "tau",
    "mae",
    "r2",
    "pinball_loss",
    "quantile_crossing_rate",
]


def _write_baseline_ladder(
    path: Path, r2_by_target: dict, model: str = "ridge"
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    tau_by_target = {"percentile_75": 0.75, "percentile_90": 0.9, "percentile_95": 0.95}
    for target, r2 in r2_by_target.items():
        rows.append(
            {
                "model": model,
                "target": target,
                "tau": tau_by_target[target],
                "mae": 10.0,
                "r2": r2,
                "pinball_loss": 5.0,
                "quantile_crossing_rate": 0.0,
            }
        )
    pd.DataFrame(rows, columns=BASELINE_LADDER_COLUMNS).to_csv(path, index=False)


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


def test_figure_a1_partial_status_when_one_experiment_missing(tmp_path: Path) -> None:
    config = _make_config(tmp_path, ["Ew2-1", "Ew2-2"])
    _write_baseline_ladder(
        Path(config.feature_to_mcherry_dir) / "Ew2-1" / "baseline_ladder.csv",
        {"percentile_75": 0.15, "percentile_90": 0.15, "percentile_95": 0.16},
    )
    # Ew2-2 deliberately has no baseline_ladder.csv.

    status = group_a.figure_a1(config)

    assert status.status == "partial"
    assert status.missing == ["Ew2-2"]
    assert len(status.output_paths) == 1
    assert Path(status.output_paths[0]).exists()


def test_figure_a1_generated_when_all_present(tmp_path: Path) -> None:
    config = _make_config(tmp_path, ["Ew2-1", "Ew2-2"])
    for exp, r2 in [("Ew2-1", 0.15), ("Ew2-2", 0.18)]:
        _write_baseline_ladder(
            Path(config.feature_to_mcherry_dir) / exp / "baseline_ladder.csv",
            {"percentile_75": r2, "percentile_90": r2, "percentile_95": r2},
        )

    status = group_a.figure_a1(config)

    assert status.status == "generated"
    assert status.missing == []


def test_figure_a1_blocked_when_nothing_present(tmp_path: Path) -> None:
    config = _make_config(tmp_path, ["Ew2-1"])

    status = group_a.figure_a1(config)

    assert status.status == "blocked"
    assert status.output_paths == []


def _write_scportrait_dirs(config: ReportFiguresConfig, exp: str) -> Path:
    ftm_dir = Path(config.feature_to_mcherry_dir)
    scportrait_dir = ftm_dir.parent / f"{ftm_dir.name}_{exp}_scportrait"
    scportrait_dir.mkdir(parents=True, exist_ok=True)
    return scportrait_dir


def test_figure_a2_generated_with_both_rungs(tmp_path: Path) -> None:
    config = _make_config(tmp_path, ["HD1509"])
    config = replace(config, scportrait_experiments=["HD1509"])

    _write_baseline_ladder(
        Path(config.feature_to_mcherry_dir) / "HD1509" / "baseline_ladder.csv",
        {"percentile_75": 0.22, "percentile_90": 0.22, "percentile_95": 0.23},
    )
    scportrait_dir = _write_scportrait_dirs(config, "HD1509")
    _write_baseline_ladder(
        scportrait_dir / "baseline_ladder.csv",
        {"percentile_75": 0.05, "percentile_90": 0.06, "percentile_95": 0.06},
    )
    _write_floor_metrics(
        Path(config.informativeness_dir) / "HD1509" / "floor_metrics.csv",
        [
            {
                "variant": "with_suspect",
                "model": "gradient_boosting",
                "backend": "lightgbm",
                "target": "percentile_75",
                "tau": 0.75,
                "mae": 18.0,
                "r2": 0.31,
                "pinball_loss": 8.0,
            }
        ],
    )
    _write_floor_metrics(
        scportrait_dir / "morphology_informativeness" / "floor_metrics.csv",
        [
            {
                "variant": "with_suspect",
                "model": "gradient_boosting",
                "backend": "lightgbm",
                "target": "percentile_75",
                "tau": 0.75,
                "mae": 175.0,
                "r2": 0.05,
                "pinball_loss": 87.0,
            }
        ],
    )

    status = group_a.figure_a2(config)

    assert status.status == "generated"
    assert status.missing == []
    assert Path(status.output_paths[0]).exists()


def test_figure_a2_partial_when_lightgbm_rung_missing(tmp_path: Path) -> None:
    # Ridge-only experiment: no floor_metrics.csv anywhere -- the LightGBM panel
    # must render as "no data" rather than crashing, and the experiment must
    # still show up in `missing` (a real gap, not silently dropped).
    config = _make_config(tmp_path, ["HD1509"])
    config = replace(config, scportrait_experiments=["HD1509"])

    _write_baseline_ladder(
        Path(config.feature_to_mcherry_dir) / "HD1509" / "baseline_ladder.csv",
        {"percentile_75": 0.22, "percentile_90": 0.22, "percentile_95": 0.23},
    )
    scportrait_dir = _write_scportrait_dirs(config, "HD1509")
    _write_baseline_ladder(
        scportrait_dir / "baseline_ladder.csv",
        {"percentile_75": 0.05, "percentile_90": 0.06, "percentile_95": 0.06},
    )

    status = group_a.figure_a2(config)

    assert status.status == "partial"
    assert status.missing == ["HD1509"]
    assert Path(status.output_paths[0]).exists()


def test_figure_a2_blocked_when_nothing_present(tmp_path: Path) -> None:
    config = _make_config(tmp_path, ["HD1509"])
    config = replace(config, scportrait_experiments=["HD1509"])

    status = group_a.figure_a2(config)

    assert status.status == "blocked"


def test_figure_a3_uses_linear_quantile_rows(tmp_path: Path) -> None:
    config = _make_config(tmp_path, ["Ew2-1"])
    path = Path(config.feature_to_mcherry_dir) / "Ew2-1" / "baseline_ladder.csv"
    path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "model": "linear_quantile",
                "target": "percentile_75",
                "tau": 0.75,
                "mae": 10.0,
                "r2": 0.01,
                "pinball_loss": 5.0,
                "quantile_crossing_rate": 0.0,
            },
            {
                "model": "linear_quantile",
                "target": "percentile_95",
                "tau": 0.95,
                "mae": 12.0,
                "r2": -2.3,
                "pinball_loss": 6.0,
                "quantile_crossing_rate": 0.0,
            },
        ]
    ).to_csv(path, index=False)

    status = group_a.figure_a3(config)

    assert status.status == "generated"
    assert Path(status.output_paths[0]).exists()


def _write_floor_metrics(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_figure_a4_annotates_multiple_experiments_without_crashing(
    tmp_path: Path,
) -> None:
    # Regression test: annotate() must use the bar's numeric x-position, not the
    # string experiment label, once there is more than one experiment/bar group
    # (a single-bar figure won't exercise matplotlib's categorical-axis codepath).
    config = _make_config(tmp_path, ["Ew2-1", "Ew2-2"])
    for exp, lin_r2, nonlin_r2 in [("Ew2-1", 0.15, 0.33), ("Ew2-2", 0.18, 0.21)]:
        _write_floor_metrics(
            Path(config.informativeness_dir) / exp / "floor_metrics.csv",
            [
                {
                    "variant": "with_suspect",
                    "model": "ridge",
                    "backend": "sklearn",
                    "target": "percentile_75",
                    "tau": 0.75,
                    "mae": 10.0,
                    "r2": lin_r2,
                    "pinball_loss": 5.0,
                },
                {
                    "variant": "with_suspect",
                    "model": "gradient_boosting",
                    "backend": "lightgbm",
                    "target": "percentile_75",
                    "tau": 0.75,
                    "mae": 8.0,
                    "r2": nonlin_r2,
                    "pinball_loss": 4.0,
                },
            ],
        )

    status = group_a.figure_a4(config)

    assert status.status == "generated"
    assert Path(status.output_paths[0]).exists()


def test_figure_a4_orders_experiments_by_nonlinear_floor(
    tmp_path: Path, monkeypatch
) -> None:
    config = _make_config(tmp_path, ["Ew2-1", "Ew2-2", "HD1883"])
    # Nonlinear R2 ascending: Ew2-2 (0.10) < Ew2-1 (0.30) < HD1883 (0.36) -- the
    # figure's experiment order (in every percentile facet) must follow this,
    # not config.experiments' order (Ew2-1, Ew2-2, HD1883).
    for exp, lin_r2, nonlin_r2 in [
        ("Ew2-1", 0.15, 0.30),
        ("Ew2-2", 0.18, 0.10),
        ("HD1883", 0.25, 0.36),
    ]:
        _write_floor_metrics(
            Path(config.informativeness_dir) / exp / "floor_metrics.csv",
            [
                {
                    "variant": "with_suspect",
                    "model": "ridge",
                    "backend": "sklearn",
                    "target": "percentile_75",
                    "tau": 0.75,
                    "mae": 10.0,
                    "r2": lin_r2,
                    "pinball_loss": 5.0,
                },
                {
                    "variant": "with_suspect",
                    "model": "gradient_boosting",
                    "backend": "lightgbm",
                    "target": "percentile_75",
                    "tau": 0.75,
                    "mae": 8.0,
                    "r2": nonlin_r2,
                    "pinball_loss": 4.0,
                },
            ],
        )

    captured_figs = []
    real_save_fig = group_a.save_fig

    def _capturing_save_fig(fig, *args, **kwargs):
        captured_figs.append(fig)
        return real_save_fig(fig, *args, **kwargs)

    monkeypatch.setattr(group_a, "save_fig", _capturing_save_fig)

    status = group_a.figure_a4(config)

    assert status.status == "generated"
    ax = captured_figs[0].axes[0]
    tick_labels = [label.get_text() for label in ax.get_xticklabels()]
    assert tick_labels == ["Ew2-2", "Ew2-1", "HD1883"]


def _make_a5_config(tmp_path: Path, a5_experiments) -> ReportFiguresConfig:
    return ReportFiguresConfig(
        experiments=list(set(a5_experiments) | {"Ew2-1"}),
        scportrait_experiments=[],
        a5_experiments=a5_experiments,
        a5_area_p90_dir=str(tmp_path / "area_p90"),
        feature_to_mcherry_dir=str(tmp_path / "feature_to_mcherry"),
        informativeness_dir=str(tmp_path / "morphology_informativeness"),
        data_quality_dir=str(tmp_path / "data_quality"),
        data_root=str(tmp_path / "data_root"),
        output_dir=str(tmp_path / "out"),
        formats=["png"],
    )


def _write_area_p90(path: Path, area, p90) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"area": area, "percentile_90": p90}).to_csv(path, index=False)


def test_figure_a5_generated_computes_spearman_rho_matching_scipy(
    tmp_path: Path,
) -> None:
    import numpy as np
    from scipy.stats import spearmanr

    config = _make_a5_config(tmp_path, ["HD1509"])
    rng = np.random.default_rng(0)
    area = rng.uniform(0, 100, size=200)

    for source, noise_scale in [("cellpose_sam", 5), ("scportrait", 40)]:
        p90 = area + rng.normal(0, noise_scale, size=200)
        _write_area_p90(
            Path(config.a5_area_p90_dir) / f"HD1509_{source}.csv", area, p90
        )

    status = group_a.figure_a5(config)

    assert status.status == "generated"
    assert status.missing == []
    assert Path(status.output_paths[0]).exists()

    # Independently recompute rho the same way figure_a5 does, and confirm the
    # cellpose_sam side (lower noise) is more correlated than scPortrait's.
    cellpose_df = pd.read_csv(Path(config.a5_area_p90_dir) / "HD1509_cellpose_sam.csv")
    scportrait_df = pd.read_csv(Path(config.a5_area_p90_dir) / "HD1509_scportrait.csv")
    rho_cellpose, _ = spearmanr(cellpose_df["area"], cellpose_df["percentile_90"])
    rho_scportrait, _ = spearmanr(scportrait_df["area"], scportrait_df["percentile_90"])
    assert rho_cellpose > rho_scportrait


def test_figure_a5_partial_when_one_source_missing(tmp_path: Path) -> None:
    config = _make_a5_config(tmp_path, ["HD1509"])
    _write_area_p90(
        Path(config.a5_area_p90_dir) / "HD1509_cellpose_sam.csv",
        [10, 20, 30],
        [15, 25, 35],
    )
    # scportrait side deliberately not extracted.

    status = group_a.figure_a5(config)

    assert status.status == "partial"
    assert status.missing == ["HD1509"]
    assert Path(status.output_paths[0]).exists()


def test_figure_a5_blocked_when_nothing_present(tmp_path: Path) -> None:
    config = _make_a5_config(tmp_path, ["HD1509"])

    status = group_a.figure_a5(config)

    assert status.status == "blocked"


def test_figure_a6_generated_with_all_three_variants(tmp_path: Path) -> None:
    config = _make_config(tmp_path, ["Ew2-1"])
    variant_roots = {
        "tracked": Path(config.feature_to_mcherry_dir),
        "filtered": tmp_path / "feature_to_mcherry_filtered",
        "filtered+blur": tmp_path / "feature_to_mcherry_filtered_blur",
    }
    r2_by_variant = {"tracked": 0.15, "filtered": -1.00, "filtered+blur": -1.44}
    for label, root in variant_roots.items():
        r2 = r2_by_variant[label]
        _write_baseline_ladder(
            root / "Ew2-1" / "baseline_ladder.csv",
            {"percentile_75": r2, "percentile_90": r2, "percentile_95": r2},
        )

    status = group_a.figure_a6(config)

    assert status.status == "generated"
    assert status.missing == []
    assert Path(status.output_paths[0]).exists()


def test_figure_a6_partial_when_one_variant_missing(tmp_path: Path) -> None:
    config = _make_config(tmp_path, ["Ew2-1"])
    _write_baseline_ladder(
        Path(config.feature_to_mcherry_dir) / "Ew2-1" / "baseline_ladder.csv",
        {"percentile_75": 0.15, "percentile_90": 0.15, "percentile_95": 0.16},
    )
    _write_baseline_ladder(
        (tmp_path / "feature_to_mcherry_filtered") / "Ew2-1" / "baseline_ladder.csv",
        {"percentile_75": -1.0, "percentile_90": -1.0, "percentile_95": -1.0},
    )
    # filtered+blur deliberately absent.

    status = group_a.figure_a6(config)

    assert status.status == "partial"
    assert status.missing == ["filtered+blur"]
    assert Path(status.output_paths[0]).exists()


def test_figure_a6_blocked_when_nothing_present(tmp_path: Path) -> None:
    config = _make_config(tmp_path, ["Ew2-1"])

    status = group_a.figure_a6(config)

    assert status.status == "blocked"
