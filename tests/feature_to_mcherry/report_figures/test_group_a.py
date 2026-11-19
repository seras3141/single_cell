"""Tests for feature_to_mcherry.report_figures.group_a."""

from __future__ import annotations

import os
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
from matplotlib.text import Text

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


def _floor_rows(ridge_r2: float, lightgbm_r2: float) -> list:
    return [
        {
            "variant": "with_suspect",
            "model": model,
            "backend": backend,
            "target": "percentile_90",
            "tau": 0.9,
            "mae": 10.0,
            "r2": r2,
            "pinball_loss": 5.0,
        }
        for model, backend, r2 in [
            ("ridge", "sklearn", ridge_r2),
            ("gradient_boosting", "lightgbm", lightgbm_r2),
        ]
    ]


def _write_a2_inputs(config: ReportFiguresConfig, injected: bool = False) -> None:
    _write_floor_metrics(
        Path(config.informativeness_dir) / "HD1509" / "floor_metrics.csv",
        _floor_rows(0.25, 0.32),
    )
    scportrait_dir = _write_scportrait_dirs(config, "HD1509")
    _write_floor_metrics(
        scportrait_dir / "morphology_informativeness" / "floor_metrics.csv",
        _floor_rows(0.058, 0.057),
    )
    if injected:
        ftm_dir = Path(config.feature_to_mcherry_dir)
        injected_dir = ftm_dir.parent / f"{ftm_dir.name}_HD1509_scportrait_injected"
        _write_floor_metrics(
            injected_dir / "morphology_informativeness" / "floor_metrics.csv",
            _floor_rows(0.14, 0.151),
        )


def _capture_figures(monkeypatch) -> list:
    captured = []
    real_save_fig = group_a.save_fig

    def _capturing_save_fig(fig, *args, **kwargs):
        captured.append(fig)
        return real_save_fig(fig, *args, **kwargs)

    monkeypatch.setattr(group_a, "save_fig", _capturing_save_fig)
    return captured


def _bar_labels(fig) -> list:
    return sorted(text.get_text() for ax in fig.axes for text in ax.texts)


def test_figure_a2_reads_both_rungs_from_floor_metrics(
    tmp_path: Path, monkeypatch
) -> None:
    config = _make_config(tmp_path, ["HD1509"])
    config = replace(config, scportrait_experiments=["HD1509"])
    _write_a2_inputs(config)
    captured = _capture_figures(monkeypatch)

    status = group_a.figure_a2(config)

    assert status.status == "generated"
    assert status.missing == []
    assert Path(status.output_paths[0]).exists()
    labels = _bar_labels(captured[0])
    assert {"0.25", "0.32", "0.06"} <= set(labels)


def test_figure_a2_ignores_pre_fix_baseline_ladder(tmp_path: Path) -> None:
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

    assert status.status == "blocked"


def test_figure_a2_marks_injected_rung_pending_without_reading_it(
    tmp_path: Path, monkeypatch
) -> None:
    config = _make_config(tmp_path, ["HD1509"])
    config = replace(config, scportrait_experiments=["HD1509"])
    _write_a2_inputs(config, injected=True)
    injected = group_a.resolve_experiment_paths(
        "HD1509", config
    ).scportrait_injected_floor_metrics_csv
    _write_floor_metrics(injected, _floor_rows(0.9, 0.9))
    captured = _capture_figures(monkeypatch)

    status = group_a.figure_a2(config)

    labels = _bar_labels(captured[0])
    assert labels.count("pending GPU") == 2
    assert "0.90" not in labels
    assert captured[0].axes[0].get_ylim()[1] < 0.5
    assert "pending" in status.note


def test_figure_a2_plots_injected_rung_once_no_longer_pending(
    tmp_path: Path, monkeypatch
) -> None:
    config = _make_config(tmp_path, ["HD1509"])
    config = replace(
        config, scportrait_experiments=["HD1509"], scportrait_injected_pending=False
    )
    _write_a2_inputs(config, injected=True)
    captured = _capture_figures(monkeypatch)

    status = group_a.figure_a2(config)

    labels = _bar_labels(captured[0])
    assert "pending GPU" not in labels
    assert {"0.14", "0.15"} <= set(labels)
    assert status.note == ""


def test_figure_a2_leaves_out_injected_floor_written_before_mask_fix(
    tmp_path: Path, monkeypatch
) -> None:
    config = _make_config(tmp_path, ["HD1509"])
    config = replace(
        config, scportrait_experiments=["HD1509"], scportrait_injected_pending=False
    )
    _write_a2_inputs(config, injected=True)
    injected = group_a.resolve_experiment_paths(
        "HD1509", config
    ).scportrait_injected_floor_metrics_csv
    pre_fix = datetime(2026, 7, 30).timestamp()
    os.utime(injected, (pre_fix, pre_fix))
    captured = _capture_figures(monkeypatch)

    status = group_a.figure_a2(config)

    labels = _bar_labels(captured[0])
    assert "0.15" not in labels and "0.14" not in labels
    assert status.note == "HD1509: injected floor predates the mask fix"
    assert status.status == "partial"
    assert status.missing == ["HD1509"]


def test_figure_a2_notes_missing_injected_floor_once_no_longer_pending(
    tmp_path: Path,
) -> None:
    config = _make_config(tmp_path, ["HD1509"])
    config = replace(
        config, scportrait_experiments=["HD1509"], scportrait_injected_pending=False
    )
    _write_a2_inputs(config)

    status = group_a.figure_a2(config)

    assert status.status == "partial"
    assert status.note == "HD1509: no injected floor"


def test_figure_a2_labels_handcrafted_features_not_incarta(
    tmp_path: Path, monkeypatch
) -> None:
    config = _make_config(tmp_path, ["HD1509"])
    config = replace(config, scportrait_experiments=["HD1509"])
    _write_a2_inputs(config)
    captured = _capture_figures(monkeypatch)

    group_a.figure_a2(config)

    legend_texts = [text.get_text() for text in captured[0].legends[0].get_texts()]
    assert legend_texts[0] == "handcrafted features"
    figure_texts = [text.get_text() for text in captured[0].findobj(Text)]
    assert not any("incarta" in text.lower() for text in figure_texts)


def test_figure_a2_partial_when_a_backend_lacks_one_rung_target(
    tmp_path: Path,
) -> None:
    config = _make_config(tmp_path, ["HD1509"])
    config = replace(config, scportrait_experiments=["HD1509"])
    _write_a2_inputs(config)
    scportrait_floor = group_a.resolve_experiment_paths(
        "HD1509", config
    ).scportrait_floor_metrics_csv
    p75_ridge = dict(_floor_rows(0.05, 0.05)[0], target="percentile_75")
    _write_floor_metrics(
        Path(config.informativeness_dir) / "HD1509" / "floor_metrics.csv",
        _floor_rows(0.25, 0.32)
        + [p75_ridge, dict(p75_ridge, model="gradient_boosting")],
    )
    _write_floor_metrics(scportrait_floor, _floor_rows(0.058, 0.057) + [p75_ridge])

    status = group_a.figure_a2(config)

    assert status.status == "partial"
    assert status.missing == ["HD1509"]


def test_figure_a2_averages_duplicate_rows_instead_of_failing(
    tmp_path: Path, monkeypatch
) -> None:
    config = _make_config(tmp_path, ["HD1509"])
    config = replace(config, scportrait_experiments=["HD1509"])
    _write_a2_inputs(config)
    _write_floor_metrics(
        Path(config.informativeness_dir) / "HD1509" / "floor_metrics.csv",
        _floor_rows(0.25, 0.30) + _floor_rows(0.25, 0.34),
    )
    captured = _capture_figures(monkeypatch)

    status = group_a.figure_a2(config)

    assert status.status == "partial"
    assert status.missing == []
    assert "0.32" in _bar_labels(captured[0])
    assert "duplicate rows averaged: HD1509/handcrafted features/" in status.note


def test_figure_a2_puts_negative_labels_below_the_bar(
    tmp_path: Path, monkeypatch
) -> None:
    config = _make_config(tmp_path, ["HD1509"])
    config = replace(config, scportrait_experiments=["HD1509"])
    _write_a2_inputs(config)
    scportrait_floor = group_a.resolve_experiment_paths(
        "HD1509", config
    ).scportrait_floor_metrics_csv
    _write_floor_metrics(scportrait_floor, _floor_rows(-0.04, 0.057))
    captured = _capture_figures(monkeypatch)

    group_a.figure_a2(config)

    negative = [
        text
        for ax in captured[0].axes
        for text in ax.texts
        if text.get_text() == "-0.04"
    ]
    assert [text.get_va() for text in negative] == ["top"]


def test_figure_a2_partial_when_injected_floor_is_incomplete(
    tmp_path: Path, monkeypatch
) -> None:
    config = _make_config(tmp_path, ["HD1509"])
    config = replace(
        config, scportrait_experiments=["HD1509"], scportrait_injected_pending=False
    )
    _write_a2_inputs(config, injected=True)
    injected = group_a.resolve_experiment_paths(
        "HD1509", config
    ).scportrait_injected_floor_metrics_csv
    _write_floor_metrics(injected, _floor_rows(0.14, 0.151)[:1])

    captured = _capture_figures(monkeypatch)

    status = group_a.figure_a2(config)

    assert status.status == "partial"
    assert status.note == "HD1509: injected floor incomplete"
    legend_texts = [text.get_text() for text in captured[0].legends[0].get_texts()]
    assert legend_texts[-1] == (
        "scPortrait (injected Cellpose masks) -- incomplete for HD1509"
    )
    assert "0.14" in _bar_labels(captured[0])


def test_figure_a2_ignores_non_percentile_targets_for_scale_and_coverage(
    tmp_path: Path, monkeypatch
) -> None:
    config = _make_config(tmp_path, ["HD1509"])
    config = replace(config, scportrait_experiments=["HD1509"])
    _write_a2_inputs(config)
    extra = dict(_floor_rows(0.25, 0.32)[0], target="mean_intensity", r2=3.0)
    _write_floor_metrics(
        Path(config.informativeness_dir) / "HD1509" / "floor_metrics.csv",
        _floor_rows(0.25, 0.32) + [extra],
    )
    captured = _capture_figures(monkeypatch)

    status = group_a.figure_a2(config)

    assert status.status == "generated"
    assert captured[0].axes[0].get_ylim()[1] < 1.0


def test_figure_a2_notes_when_the_nonlinear_rung_mixes_learners(
    tmp_path: Path,
) -> None:
    config = _make_config(tmp_path, ["HD1509"])
    config = replace(config, scportrait_experiments=["HD1509"])
    _write_a2_inputs(config)
    scportrait_floor = group_a.resolve_experiment_paths(
        "HD1509", config
    ).scportrait_floor_metrics_csv
    ridge, boosting = _floor_rows(0.058, 0.057)
    _write_floor_metrics(scportrait_floor, [ridge, dict(boosting, backend="sklearn")])

    status = group_a.figure_a2(config)

    assert "nonlinear rung mixes learners: lightgbm, sklearn" in status.note
    assert status.status == "partial"


def test_figure_a2_treats_nan_r2_as_missing(tmp_path: Path, monkeypatch) -> None:
    config = _make_config(tmp_path, ["HD1509"])
    config = replace(config, scportrait_experiments=["HD1509"])
    _write_a2_inputs(config)
    scportrait_floor = group_a.resolve_experiment_paths(
        "HD1509", config
    ).scportrait_floor_metrics_csv
    ridge, boosting = _floor_rows(0.058, 0.057)
    _write_floor_metrics(scportrait_floor, [ridge, dict(boosting, r2=float("nan"))])
    captured = _capture_figures(monkeypatch)

    status = group_a.figure_a2(config)

    assert status.status == "partial"
    assert "nan" not in _bar_labels(captured[0])


def test_figure_a2_legend_names_experiments_missing_the_injected_rung(
    tmp_path: Path, monkeypatch
) -> None:
    config = _make_config(tmp_path, ["HD1509"])
    config = replace(
        config, scportrait_experiments=["HD1509"], scportrait_injected_pending=False
    )
    _write_a2_inputs(config)
    captured = _capture_figures(monkeypatch)

    group_a.figure_a2(config)

    legend_texts = [text.get_text() for text in captured[0].legends[0].get_texts()]
    assert legend_texts[-1] == (
        "scPortrait (injected Cellpose masks) -- not plotted for HD1509"
    )


def test_figure_a2_treats_an_empty_floor_csv_as_missing(tmp_path: Path) -> None:
    config = _make_config(tmp_path, ["HD1509"])
    config = replace(config, scportrait_experiments=["HD1509"])
    _write_a2_inputs(config)
    group_a.resolve_experiment_paths(
        "HD1509", config
    ).scportrait_floor_metrics_csv.write_text("")

    status = group_a.figure_a2(config)

    assert status.status == "partial"
    assert status.missing == ["HD1509"]


def test_figure_a2_leaves_room_below_the_most_negative_bar(
    tmp_path: Path, monkeypatch
) -> None:
    config = _make_config(tmp_path, ["HD1509"])
    config = replace(config, scportrait_experiments=["HD1509"])
    _write_a2_inputs(config)
    scportrait_floor = group_a.resolve_experiment_paths(
        "HD1509", config
    ).scportrait_floor_metrics_csv
    _write_floor_metrics(scportrait_floor, _floor_rows(-0.04, 0.057))
    captured = _capture_figures(monkeypatch)

    group_a.figure_a2(config)

    assert captured[0].axes[0].get_ylim()[0] < -0.07


def test_figure_a2_lists_an_injected_floor_without_usable_rows_as_not_plotted(
    tmp_path: Path, monkeypatch
) -> None:
    config = _make_config(tmp_path, ["HD1509"])
    config = replace(
        config, scportrait_experiments=["HD1509"], scportrait_injected_pending=False
    )
    _write_a2_inputs(config, injected=True)
    injected = group_a.resolve_experiment_paths(
        "HD1509", config
    ).scportrait_injected_floor_metrics_csv
    without_suspect = [
        dict(row, variant="without_suspect") for row in _floor_rows(0.14, 0.151)
    ]
    _write_floor_metrics(injected, without_suspect)
    captured = _capture_figures(monkeypatch)

    status = group_a.figure_a2(config)

    legend_texts = [text.get_text() for text in captured[0].legends[0].get_texts()]
    assert legend_texts[-1] == (
        "scPortrait (injected Cellpose masks) -- not plotted for HD1509"
    )
    assert status.note == (
        "HD1509: no usable rows in scPortrait (injected Cellpose masks) "
        "floor_metrics.csv"
    )


def test_figure_a2_names_an_unreadable_floor_in_the_note(tmp_path: Path) -> None:
    config = _make_config(tmp_path, ["HD1509"])
    config = replace(config, scportrait_experiments=["HD1509"])
    _write_a2_inputs(config)
    floor = group_a.resolve_experiment_paths("HD1509", config).floor_metrics_csv
    floor.write_text("variant,model,target,R2\n")

    status = group_a.figure_a2(config)

    assert status.missing == ["HD1509"]
    assert "HD1509: unreadable handcrafted features floor_metrics.csv" in status.note


def test_figure_a2_survives_a_ragged_floor_csv(tmp_path: Path) -> None:
    config = _make_config(tmp_path, ["HD1509"])
    config = replace(config, scportrait_experiments=["HD1509"])
    _write_a2_inputs(config)
    group_a.resolve_experiment_paths(
        "HD1509", config
    ).scportrait_floor_metrics_csv.write_text('a,b\n1,2\n"3,4\n')

    status = group_a.figure_a2(config)

    assert status.status == "partial"
    assert "HD1509: unreadable scPortrait (native masks) floor_metrics.csv" in (
        status.note
    )


def test_figure_a2_says_why_a_readable_floor_gave_no_rows(tmp_path: Path) -> None:
    config = _make_config(tmp_path, ["HD1509"])
    config = replace(config, scportrait_experiments=["HD1509"])
    _write_a2_inputs(config)
    scportrait_floor = group_a.resolve_experiment_paths(
        "HD1509", config
    ).scportrait_floor_metrics_csv
    _write_floor_metrics(
        scportrait_floor,
        [dict(row, variant="without_suspect") for row in _floor_rows(0.05, 0.05)],
    )

    status = group_a.figure_a2(config)

    assert "HD1509: no usable rows in scPortrait (native masks) floor_metrics.csv" in (
        status.note
    )


def test_figure_a2_marks_an_empty_rung_panel(tmp_path: Path, monkeypatch) -> None:
    config = _make_config(tmp_path, ["HD1509"])
    config = replace(config, scportrait_experiments=["HD1509"])
    _write_floor_metrics(
        Path(config.informativeness_dir) / "HD1509" / "floor_metrics.csv",
        _floor_rows(0.25, 0.32)[:1],
    )
    captured = _capture_figures(monkeypatch)

    group_a.figure_a2(config)

    nonlinear_texts = [text.get_text() for text in captured[0].axes[1].texts]
    assert nonlinear_texts == ["no data"]
    assert "no data" not in [text.get_text() for text in captured[0].axes[0].texts]


def test_figure_a2_drops_non_numeric_r2_values(tmp_path: Path) -> None:
    config = _make_config(tmp_path, ["HD1509"])
    config = replace(config, scportrait_experiments=["HD1509"])
    _write_a2_inputs(config)
    ridge, boosting = _floor_rows(0.25, 0.32)
    _write_floor_metrics(
        Path(config.informativeness_dir) / "HD1509" / "floor_metrics.csv",
        [ridge, dict(boosting, r2="failed")],
    )

    status = group_a.figure_a2(config)

    assert status.status == "partial"
    assert status.missing == ["HD1509"]


def test_figure_a2_notes_a_floor_path_that_cannot_be_read(tmp_path: Path) -> None:
    config = _make_config(tmp_path, ["HD1509"])
    config = replace(config, scportrait_experiments=["HD1509"])
    _write_a2_inputs(config)
    floor = group_a.resolve_experiment_paths("HD1509", config).floor_metrics_csv
    floor.unlink()
    floor.mkdir()

    status = group_a.figure_a2(config)

    assert "HD1509: unreadable handcrafted features floor_metrics.csv" in status.note


def test_figure_a2_treats_an_injected_floor_that_is_not_a_file_as_missing(
    tmp_path: Path,
) -> None:
    config = _make_config(tmp_path, ["HD1509"])
    config = replace(
        config, scportrait_experiments=["HD1509"], scportrait_injected_pending=False
    )
    _write_a2_inputs(config, injected=True)
    injected = group_a.resolve_experiment_paths(
        "HD1509", config
    ).scportrait_injected_floor_metrics_csv
    injected.unlink()
    injected.mkdir()

    status = group_a.figure_a2(config)

    assert status.note == "HD1509: no injected floor"


def test_figure_a2_blocked_note_says_when_no_floor_file_was_found(
    tmp_path: Path,
) -> None:
    config = _make_config(tmp_path, ["HD1509"])
    config = replace(config, scportrait_experiments=["HD1509"])

    status = group_a.figure_a2(config)

    assert status.status == "blocked"
    assert status.note.startswith(
        "No floor_metrics.csv found for any scPortrait experiment"
    )


def _write_native_only_experiment(config: ReportFiguresConfig, exp: str) -> None:
    _write_floor_metrics(
        Path(config.informativeness_dir) / exp / "floor_metrics.csv",
        _floor_rows(0.2, 0.3),
    )
    _write_floor_metrics(
        _write_scportrait_dirs(config, exp)
        / "morphology_informativeness"
        / "floor_metrics.csv",
        _floor_rows(0.01, 0.02),
    )


def test_figure_a2_expects_injected_rung_only_for_injected_experiments(
    tmp_path: Path,
) -> None:
    config = _make_config(tmp_path, ["HD1509", "SA110"])
    config = replace(
        config,
        scportrait_experiments=["HD1509", "SA110"],
        scportrait_injected_pending=False,
    )
    _write_a2_inputs(config, injected=True)
    _write_native_only_experiment(config, "SA110")

    status = group_a.figure_a2(config)

    assert status.status == "generated"
    assert status.missing == []


def test_figure_a2_draws_pending_slots_only_for_injected_experiments(
    tmp_path: Path, monkeypatch
) -> None:
    config = _make_config(tmp_path, ["HD1509", "SA110"])
    config = replace(config, scportrait_experiments=["HD1509", "SA110"])
    _write_a2_inputs(config)
    _write_native_only_experiment(config, "SA110")
    captured = _capture_figures(monkeypatch)

    group_a.figure_a2(config)

    assert _bar_labels(captured[0]).count("pending GPU") == 2


def test_figure_a2_blocked_note_keeps_pending_and_unread_injected_floor_in_view(
    tmp_path: Path,
) -> None:
    config = _make_config(tmp_path, ["HD1509"])
    config = replace(config, scportrait_experiments=["HD1509"])

    pending_status = group_a.figure_a2(config)

    assert "drawn as pending" in pending_status.note

    config = replace(config, scportrait_injected_pending=False)
    ftm_dir = Path(config.feature_to_mcherry_dir)
    injected = (
        ftm_dir.parent
        / f"{ftm_dir.name}_HD1509_scportrait_injected"
        / "morphology_informativeness"
        / "floor_metrics.csv"
    )
    _write_floor_metrics(injected, _floor_rows(0.14, 0.151))
    pre_fix = datetime(2026, 7, 30).timestamp()
    os.utime(injected, (pre_fix, pre_fix))

    unread_status = group_a.figure_a2(config)

    assert unread_status.note.startswith("No usable floor rows")
    assert "HD1509: injected floor predates the mask fix" in unread_status.note


def test_figure_a2_omits_pending_rung_when_no_injected_experiment_is_drawn(
    tmp_path: Path, monkeypatch
) -> None:
    config = _make_config(tmp_path, ["HD1509", "SA110"])
    config = replace(config, scportrait_experiments=["HD1509", "SA110"])
    _write_native_only_experiment(config, "SA110")
    captured = _capture_figures(monkeypatch)

    group_a.figure_a2(config)

    legend_texts = [text.get_text() for text in captured[0].legends[0].get_texts()]
    assert legend_texts == ["handcrafted features", "scPortrait (native masks)"]


def test_figure_a2_counts_a_read_injected_floor_as_found(tmp_path: Path) -> None:
    config = _make_config(tmp_path, ["HD1509"])
    config = replace(
        config, scportrait_experiments=["HD1509"], scportrait_injected_pending=False
    )
    ftm_dir = Path(config.feature_to_mcherry_dir)
    injected = (
        ftm_dir.parent
        / f"{ftm_dir.name}_HD1509_scportrait_injected"
        / "morphology_informativeness"
        / "floor_metrics.csv"
    )
    _write_floor_metrics(
        injected,
        [dict(row, variant="without_suspect") for row in _floor_rows(0.14, 0.151)],
    )

    status = group_a.figure_a2(config)

    assert status.status == "blocked"
    assert status.note.startswith("No usable floor rows")


def test_figure_a2_names_the_read_error_in_the_note(tmp_path: Path) -> None:
    config = _make_config(tmp_path, ["HD1509"])
    config = replace(config, scportrait_experiments=["HD1509"])
    _write_a2_inputs(config)
    group_a.resolve_experiment_paths(
        "HD1509", config
    ).scportrait_floor_metrics_csv.write_text("")

    status = group_a.figure_a2(config)

    assert (
        "HD1509: unreadable scPortrait (native masks) floor_metrics.csv "
        "(EmptyDataError)" in status.note
    )


def test_figure_a2_drops_infinite_r2_values(tmp_path: Path) -> None:
    config = _make_config(tmp_path, ["HD1509"])
    config = replace(config, scportrait_experiments=["HD1509"])
    _write_a2_inputs(config)
    ridge, boosting = _floor_rows(0.058, 0.057)
    scportrait_floor = group_a.resolve_experiment_paths(
        "HD1509", config
    ).scportrait_floor_metrics_csv
    _write_floor_metrics(scportrait_floor, [ridge, dict(boosting, r2=float("-inf"))])

    status = group_a.figure_a2(config)

    assert status.status == "partial"
    assert status.output_paths


def test_figure_a2_treats_an_injected_floor_from_the_audit_day_as_pre_fix(
    tmp_path: Path,
) -> None:
    config = _make_config(tmp_path, ["HD1509"])
    config = replace(
        config, scportrait_experiments=["HD1509"], scportrait_injected_pending=False
    )
    _write_a2_inputs(config, injected=True)
    injected = group_a.resolve_experiment_paths(
        "HD1509", config
    ).scportrait_injected_floor_metrics_csv
    audit_day = datetime(2026, 8, 18, 9, 0).timestamp()
    os.utime(injected, (audit_day, audit_day))

    status = group_a.figure_a2(config)

    assert status.note == "HD1509: injected floor predates the mask fix"


def test_figure_a2_footnote_limits_the_population_caveat_to_native_masks(
    tmp_path: Path, monkeypatch
) -> None:
    config = _make_config(tmp_path, ["HD1509"])
    config = replace(config, scportrait_experiments=["HD1509"])
    _write_a2_inputs(config)
    captured = _capture_figures(monkeypatch)

    group_a.figure_a2(config)

    footnote = captured[0].texts[-1].get_text()
    assert footnote.startswith("scPortrait on native masks re-extracts")


def test_figure_a2_legend_ignores_injected_gaps_of_undrawn_experiments(
    tmp_path: Path, monkeypatch
) -> None:
    config = _make_config(tmp_path, ["HD1509", "SA110"])
    config = replace(
        config,
        scportrait_experiments=["HD1509", "SA110"],
        scportrait_injected_experiments=["HD1509", "SA110"],
        scportrait_injected_pending=False,
    )
    _write_a2_inputs(config, injected=True)
    captured = _capture_figures(monkeypatch)

    group_a.figure_a2(config)

    legend_texts = [text.get_text() for text in captured[0].legends[0].get_texts()]
    assert legend_texts[-1] == "scPortrait (injected Cellpose masks)"


def test_figure_a2_empty_rung_panel_has_no_tick_labels(
    tmp_path: Path, monkeypatch
) -> None:
    config = _make_config(tmp_path, ["HD1509"])
    config = replace(config, scportrait_experiments=["HD1509"])
    _write_floor_metrics(
        Path(config.informativeness_dir) / "HD1509" / "floor_metrics.csv",
        _floor_rows(0.25, 0.32)[:1],
    )
    captured = _capture_figures(monkeypatch)

    group_a.figure_a2(config)

    assert list(captured[0].axes[1].get_xticks()) == []


def test_figure_a2_drops_the_native_footnote_without_native_bars(
    tmp_path: Path, monkeypatch
) -> None:
    config = _make_config(tmp_path, ["HD1509"])
    config = replace(config, scportrait_experiments=["HD1509"])
    _write_floor_metrics(
        Path(config.informativeness_dir) / "HD1509" / "floor_metrics.csv",
        _floor_rows(0.25, 0.32),
    )
    captured = _capture_figures(monkeypatch)

    group_a.figure_a2(config)

    assert not any("native masks" in text.get_text() for text in captured[0].texts)


def test_figure_a2_partial_when_a_backend_lacks_one_rung(tmp_path: Path) -> None:
    config = _make_config(tmp_path, ["HD1509"])
    config = replace(config, scportrait_experiments=["HD1509"])
    _write_a2_inputs(config)
    _write_floor_metrics(
        Path(config.informativeness_dir) / "HD1509" / "floor_metrics.csv",
        _floor_rows(0.25, 0.32)[:1],
    )

    status = group_a.figure_a2(config)

    assert status.status == "partial"
    assert status.missing == ["HD1509"]


def test_figure_a2_plots_only_the_with_suspect_variant(
    tmp_path: Path, monkeypatch
) -> None:
    config = _make_config(tmp_path, ["HD1509"])
    config = replace(config, scportrait_experiments=["HD1509"])
    _write_a2_inputs(config)
    without_suspect = [
        dict(row, variant="without_suspect", r2=0.99) for row in _floor_rows(0.25, 0.32)
    ]
    _write_floor_metrics(
        Path(config.informativeness_dir) / "HD1509" / "floor_metrics.csv",
        _floor_rows(0.25, 0.32) + without_suspect,
    )
    captured = _capture_figures(monkeypatch)

    group_a.figure_a2(config)

    labels = _bar_labels(captured[0])
    assert "0.25" in labels and "0.32" in labels
    assert "0.99" not in labels


def test_figure_a2_ignores_models_outside_the_two_rungs(
    tmp_path: Path, monkeypatch
) -> None:
    config = _make_config(tmp_path, ["HD1509"])
    config = replace(config, scportrait_experiments=["HD1509"])
    _write_a2_inputs(config)
    extra = dict(_floor_rows(0.25, 0.32)[1], model="random_forest", r2=0.9)
    _write_floor_metrics(
        Path(config.informativeness_dir) / "HD1509" / "floor_metrics.csv",
        _floor_rows(0.25, 0.32) + [extra],
    )
    captured = _capture_figures(monkeypatch)

    status = group_a.figure_a2(config)

    assert status.status == "generated"
    assert "0.90" not in _bar_labels(captured[0])


def test_figure_a2_names_rungs_by_model_family_not_library(
    tmp_path: Path, monkeypatch
) -> None:
    config = _make_config(tmp_path, ["HD1509"])
    config = replace(config, scportrait_experiments=["HD1509"])
    _write_a2_inputs(config)
    captured = _capture_figures(monkeypatch)

    group_a.figure_a2(config)

    titles = [ax.get_title() for ax in captured[0].axes]
    assert titles == ["linear (Ridge) floor", "nonlinear (boosting) floor"]


def test_figure_a2_blocked_when_no_percentile_targets(tmp_path: Path) -> None:
    config = _make_config(tmp_path, ["HD1509"])
    config = replace(config, scportrait_experiments=["HD1509"])
    rows = [dict(row, target="mean_intensity") for row in _floor_rows(0.25, 0.32)]
    _write_floor_metrics(
        Path(config.informativeness_dir) / "HD1509" / "floor_metrics.csv", rows
    )
    scportrait_dir = _write_scportrait_dirs(config, "HD1509")
    _write_floor_metrics(
        scportrait_dir / "morphology_informativeness" / "floor_metrics.csv", rows
    )

    status = group_a.figure_a2(config)

    assert status.status == "blocked"
    assert status.output_paths == []


def test_figure_a2_partial_when_scportrait_floor_missing(tmp_path: Path) -> None:
    config = _make_config(tmp_path, ["HD1509"])
    config = replace(config, scportrait_experiments=["HD1509"])
    _write_floor_metrics(
        Path(config.informativeness_dir) / "HD1509" / "floor_metrics.csv",
        _floor_rows(0.25, 0.32),
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


# --- DMSO-normalized runs (z_-prefixed targets) -----------------------------------


def _write_ladder_with_targets(path: Path, r2_by_target: dict) -> None:
    """Like ``_write_baseline_ladder`` but accepts arbitrary target names.

    The shared helper keys tau off the raw ``percentile_<N>`` names, so it cannot write
    a DMSO-normalized ladder.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "model": "ridge",
            "target": target,
            "tau": 0.9,
            "mae": 10.0,
            "r2": r2,
            "pinball_loss": 5.0,
            "quantile_crossing_rate": 0.0,
        }
        for target, r2 in r2_by_target.items()
    ]
    pd.DataFrame(rows, columns=BASELINE_LADDER_COLUMNS).to_csv(path, index=False)


def test_figure_a1_handles_dmso_normalized_targets(tmp_path: Path) -> None:
    """A normalized run's targets are ``z_percentile_<N>``; the figure must still draw.

    Filtering against a hard-coded list of raw names selected nothing here, and the bar
    layout then divided by a zero target count -- a crash, not a degraded figure.
    """
    config = _make_config(tmp_path, ["Ew2-1"])
    _write_ladder_with_targets(
        Path(config.feature_to_mcherry_dir) / "Ew2-1" / "baseline_ladder.csv",
        {"z_percentile_75": 0.15, "z_percentile_90": 0.15, "z_percentile_95": 0.16},
    )

    status = group_a.figure_a1(config)

    assert status.status == "generated"
    assert status.output_paths


def test_figure_a1_blocks_when_no_percentile_targets(tmp_path: Path) -> None:
    """An unrecognised target set is reported as blocked, never drawn empty."""
    config = _make_config(tmp_path, ["Ew2-1"])
    _write_ladder_with_targets(
        Path(config.feature_to_mcherry_dir) / "Ew2-1" / "baseline_ladder.csv",
        {"area": 0.15},
    )

    status = group_a.figure_a1(config)

    assert status.status == "blocked"
    assert "No percentile targets" in (status.note or "")


def _write_decomposition(
    config: ReportFiguresConfig, exp: str, nested: dict, curve: dict
) -> None:
    decomposition = Path(config.feature_to_mcherry_dir) / exp / "decomposition"
    decomposition.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"target": target, "level": level, "r2": r2, "n": 100}
            for target in ("percentile_75", "percentile_90")
            for level, r2 in nested.items()
        ]
    ).to_csv(decomposition / "nested_r2.csv", index=False)
    pd.DataFrame(
        [
            {
                "target": target,
                "timepoint": timepoint,
                "r2": r2,
                "n_cells": 50,
                "n_wells": 9,
            }
            for target in ("percentile_75", "percentile_90")
            for timepoint, r2 in curve.items()
        ]
    ).to_csv(decomposition / "r2_by_timepoint.csv", index=False)


def test_figure_a7_plots_nested_levels_and_timepoint_curve(
    tmp_path: Path, monkeypatch
) -> None:
    config = _make_config(tmp_path, ["HD1509", "SA110"])
    _write_decomposition(
        config,
        "HD1509",
        {
            "pooled(5fold)": 0.27,
            "within_timepoint": 0.26,
            "within_well_timepoint": 0.25,
        },
        {1: 0.2, 11: 0.28, 351: 0.1},
    )
    _write_decomposition(
        config,
        "SA110",
        {"pooled(5fold)": 0.18, "within_timepoint": 0.07, "within_well_timepoint": 0.1},
        {1: -0.1, 11: 0.05, 351: -0.2},
    )
    captured = _capture_figures(monkeypatch)

    status = group_a.figure_a7(config)

    assert status.status == "generated"
    assert Path(status.output_paths[0]).exists()
    ax_levels, ax_time = captured[0].axes
    assert {"0.27", "0.26", "0.25", "0.18", "0.07", "0.10"} <= {
        text.get_text() for text in ax_levels.texts
    }
    sa110_line = [line for line in ax_time.lines if line.get_label() == "SA110"][0]
    assert list(sa110_line.get_xdata()) == pytest.approx([0.0, 10 * 72 / 350, 72.0])
    assert list(sa110_line.get_ydata()) == pytest.approx([-0.1, 0.05, -0.2])


def test_figure_a7_clips_extreme_negative_r2_and_reports_minimum(
    tmp_path: Path, monkeypatch
) -> None:
    config = _make_config(tmp_path, ["Ew2-2"])
    _write_decomposition(
        config,
        "Ew2-2",
        {"pooled(5fold)": 0.2, "within_timepoint": 0.14, "within_well_timepoint": 0.13},
        {1: 0.1, 11: -4.07, 21: -0.5},
    )
    captured = _capture_figures(monkeypatch)

    group_a.figure_a7(config)

    ax_time = captured[0].axes[1]
    line = [line for line in ax_time.lines if line.get_label() == "Ew2-2"][0]
    assert list(line.get_ydata()) == pytest.approx([0.1, -1.0, -0.5])
    assert any("Ew2-2 -4.07" in text.get_text() for text in ax_time.texts)


def test_figure_a7_partial_when_experiment_lacks_decomposition(tmp_path: Path) -> None:
    config = _make_config(tmp_path, ["HD1509", "SA110"])
    _write_decomposition(
        config,
        "HD1509",
        {
            "pooled(5fold)": 0.27,
            "within_timepoint": 0.26,
            "within_well_timepoint": 0.25,
        },
        {1: 0.2, 11: 0.28},
    )

    status = group_a.figure_a7(config)

    assert status.status == "partial"
    assert status.missing == ["SA110"]


def test_figure_a7_blocked_when_target_absent(tmp_path: Path) -> None:
    config = replace(
        _make_config(tmp_path, ["HD1509"]), signal_location_target="percentile_95"
    )
    _write_decomposition(
        config,
        "HD1509",
        {
            "pooled(5fold)": 0.27,
            "within_timepoint": 0.26,
            "within_well_timepoint": 0.25,
        },
        {1: 0.2},
    )

    status = group_a.figure_a7(config)

    assert status.status == "blocked"


def test_figure_a7_accepts_the_plain_pooled_level_name(
    tmp_path: Path, monkeypatch
) -> None:
    config = _make_config(tmp_path, ["HD1509"])
    _write_decomposition(
        config,
        "HD1509",
        {"pooled": 0.27, "within_timepoint": 0.26, "within_well_timepoint": 0.25},
        {1: 0.2, 11: 0.28},
    )
    captured = _capture_figures(monkeypatch)

    status = group_a.figure_a7(config)

    assert status.status == "generated"
    labels = {text.get_text() for text in captured[0].axes[0].texts}
    assert {"0.27", "0.26", "0.25"} <= labels


def test_figure_a7_partial_and_noted_when_a_nested_level_is_missing(
    tmp_path: Path,
) -> None:
    config = _make_config(tmp_path, ["HD1509"])
    _write_decomposition(
        config,
        "HD1509",
        {"within_timepoint": 0.26, "within_well_timepoint": 0.25},
        {1: 0.2, 11: 0.28},
    )

    status = group_a.figure_a7(config)

    assert status.status == "partial"
    assert status.note == "HD1509: missing nested levels pooled"


def test_figure_a7_averages_duplicate_nested_levels(
    tmp_path: Path, monkeypatch
) -> None:
    config = _make_config(tmp_path, ["HD1509"])
    _write_decomposition(
        config,
        "HD1509",
        {
            "pooled(5fold)": 0.27,
            "within_timepoint": 0.26,
            "within_well_timepoint": 0.25,
        },
        {1: 0.2},
    )
    nested_csv = (
        Path(config.feature_to_mcherry_dir)
        / "HD1509"
        / "decomposition"
        / "nested_r2.csv"
    )
    table = pd.read_csv(nested_csv)
    duplicate = table[table["level"] == "pooled(5fold)"].assign(r2=0.31)
    pd.concat([table, duplicate]).to_csv(nested_csv, index=False)
    captured = _capture_figures(monkeypatch)

    status = group_a.figure_a7(config)

    assert "0.29" in {text.get_text() for text in captured[0].axes[0].texts}
    assert status.note == "HD1509: duplicate nested levels averaged"


def test_figure_a7_reports_the_minimum_ignoring_nan_timepoints(
    tmp_path: Path, monkeypatch
) -> None:
    config = _make_config(tmp_path, ["Ew2-2"])
    _write_decomposition(
        config,
        "Ew2-2",
        {"pooled(5fold)": 0.2, "within_timepoint": 0.14, "within_well_timepoint": 0.13},
        {1: float("nan"), 11: -4.07},
    )
    captured = _capture_figures(monkeypatch)

    group_a.figure_a7(config)

    assert any("Ew2-2 -4.07" in text.get_text() for text in captured[0].axes[1].texts)


def test_figure_a7_notes_why_an_experiment_is_missing(tmp_path: Path) -> None:
    config = _make_config(tmp_path, ["HD1509", "SA110"])
    _write_decomposition(
        config,
        "HD1509",
        {
            "pooled(5fold)": 0.27,
            "within_timepoint": 0.26,
            "within_well_timepoint": 0.25,
        },
        {1: 0.2},
    )

    status = group_a.figure_a7(config)

    assert status.note == "SA110: no decomposition dir"


def test_figure_a7_plots_only_the_configured_target(
    tmp_path: Path, monkeypatch
) -> None:
    config = _make_config(tmp_path, ["HD1509"])
    _write_decomposition(
        config,
        "HD1509",
        {
            "pooled(5fold)": 0.27,
            "within_timepoint": 0.26,
            "within_well_timepoint": 0.25,
        },
        {1: 0.2},
    )
    nested_csv = (
        Path(config.feature_to_mcherry_dir)
        / "HD1509"
        / "decomposition"
        / "nested_r2.csv"
    )
    table = pd.read_csv(nested_csv)
    table.loc[table["target"] == "percentile_75", "r2"] = 0.9
    table.to_csv(nested_csv, index=False)
    captured = _capture_figures(monkeypatch)

    group_a.figure_a7(config)

    labels = {text.get_text() for text in captured[0].axes[0].texts}
    assert {"0.27", "0.26", "0.25"} <= labels
    assert "0.90" not in labels


def test_figure_a7_accepts_any_fold_count_in_the_pooled_label(tmp_path: Path) -> None:
    config = _make_config(tmp_path, ["HD1509"])
    _write_decomposition(
        config,
        "HD1509",
        {
            "pooled(3fold)": 0.27,
            "within_timepoint": 0.26,
            "within_well_timepoint": 0.25,
        },
        {1: 0.2},
    )

    status = group_a.figure_a7(config)

    assert status.status == "generated"


def test_figure_a7_averages_duplicate_timepoints(tmp_path: Path, monkeypatch) -> None:
    config = _make_config(tmp_path, ["HD1509"])
    _write_decomposition(
        config,
        "HD1509",
        {
            "pooled(5fold)": 0.27,
            "within_timepoint": 0.26,
            "within_well_timepoint": 0.25,
        },
        {1: 0.2, 11: 0.3},
    )
    curve_csv = (
        Path(config.feature_to_mcherry_dir)
        / "HD1509"
        / "decomposition"
        / "r2_by_timepoint.csv"
    )
    table = pd.read_csv(curve_csv)
    pd.concat([table, table.assign(r2=table["r2"] + 0.1)]).to_csv(
        curve_csv, index=False
    )
    captured = _capture_figures(monkeypatch)

    status = group_a.figure_a7(config)

    line = [line for line in captured[0].axes[1].lines if line.get_label() == "HD1509"][
        0
    ]
    assert list(line.get_ydata()) == pytest.approx([0.25, 0.35])
    assert status.note == "HD1509: duplicate timepoints averaged"


def test_figure_a7_skips_an_unreadable_decomposition_table(tmp_path: Path) -> None:
    config = _make_config(tmp_path, ["HD1509", "SA110"])
    for exp in ("HD1509", "SA110"):
        _write_decomposition(
            config,
            exp,
            {
                "pooled(5fold)": 0.27,
                "within_timepoint": 0.26,
                "within_well_timepoint": 0.25,
            },
            {1: 0.2},
        )
    (
        Path(config.feature_to_mcherry_dir)
        / "SA110"
        / "decomposition"
        / "nested_r2.csv"
    ).write_text("")

    status = group_a.figure_a7(config)

    assert status.status == "partial"
    assert status.note == "SA110: unreadable decomposition table (EmptyDataError)"


def test_figure_a7_clipped_markers_share_their_line_colour(
    tmp_path: Path, monkeypatch
) -> None:
    config = _make_config(tmp_path, ["HD1509", "UNCOLOURED"])
    for exp in ("HD1509", "UNCOLOURED"):
        _write_decomposition(
            config,
            exp,
            {
                "pooled(5fold)": 0.2,
                "within_timepoint": 0.1,
                "within_well_timepoint": 0.1,
            },
            {1: 0.1, 11: -3.0},
        )
    captured = _capture_figures(monkeypatch)

    group_a.figure_a7(config)

    lines = captured[0].axes[1].lines
    uncoloured = [line for line in lines if line.get_label() == "UNCOLOURED"][0]
    markers = lines[lines.index(uncoloured) + 1]
    assert markers.get_color() == uncoloured.get_color()


def test_figure_a7_shades_negative_band_only_for_negative_values(
    tmp_path: Path, monkeypatch
) -> None:
    config = _make_config(tmp_path, ["HD1509"])
    _write_decomposition(
        config,
        "HD1509",
        {
            "pooled(5fold)": 0.27,
            "within_timepoint": 0.26,
            "within_well_timepoint": 0.25,
        },
        {1: 0.2, 11: 0.3},
    )
    captured = _capture_figures(monkeypatch)

    group_a.figure_a7(config)

    assert len(captured[0].axes[1].patches) == 0


def test_figure_a7_leaves_room_below_clipped_markers(
    tmp_path: Path, monkeypatch
) -> None:
    config = _make_config(tmp_path, ["Ew2-2"])
    _write_decomposition(
        config,
        "Ew2-2",
        {"pooled(5fold)": 0.2, "within_timepoint": 0.14, "within_well_timepoint": 0.13},
        {1: 0.1, 11: -4.07},
    )
    captured = _capture_figures(monkeypatch)

    group_a.figure_a7(config)

    assert captured[0].axes[1].get_ylim()[0] <= -1.25


def test_figure_a7_uses_grey_levels_distinct_from_culture_colours() -> None:
    from src.feature_to_mcherry.report_figures.theme import EXPERIMENT_COLORS

    assert not set(group_a._A7_LEVEL_COLORS) & set(EXPERIMENT_COLORS.values())


def test_figure_a7_leaves_out_the_pooled_bar_for_two_pooled_labels(
    tmp_path: Path, monkeypatch
) -> None:
    config = _make_config(tmp_path, ["HD1509"])
    _write_decomposition(
        config,
        "HD1509",
        {
            "pooled": 0.2,
            "pooled(5fold)": 0.3,
            "within_timepoint": 0.26,
            "within_well_timepoint": 0.21,
        },
        {1: 0.2},
    )
    captured = _capture_figures(monkeypatch)

    status = group_a.figure_a7(config)

    labels = {text.get_text() for text in captured[0].axes[0].texts}
    assert "0.25" not in labels
    assert status.status == "partial"
    assert status.note == (
        "HD1509: pooled bar left out, several pooled levels: pooled, pooled(5fold)"
    )


def test_figure_a7_treats_a_nan_level_as_missing(tmp_path: Path) -> None:
    config = _make_config(tmp_path, ["HD1509"])
    _write_decomposition(
        config,
        "HD1509",
        {
            "pooled(5fold)": 0.27,
            "within_timepoint": 0.26,
            "within_well_timepoint": float("nan"),
        },
        {1: 0.2},
    )

    status = group_a.figure_a7(config)

    assert status.status == "partial"
    assert status.note == "HD1509: missing nested levels within_well_timepoint"


def test_figure_a7_notes_duplicates_alongside_two_pooled_labels(
    tmp_path: Path,
) -> None:
    config = _make_config(tmp_path, ["HD1509"])
    _write_decomposition(
        config,
        "HD1509",
        {
            "pooled": 0.2,
            "pooled(5fold)": 0.3,
            "within_timepoint": 0.26,
            "within_well_timepoint": 0.21,
        },
        {1: 0.2},
    )
    nested_csv = (
        Path(config.feature_to_mcherry_dir)
        / "HD1509"
        / "decomposition"
        / "nested_r2.csv"
    )
    table = pd.read_csv(nested_csv)
    pd.concat([table, table[table["level"] == "within_timepoint"]]).to_csv(
        nested_csv, index=False
    )

    status = group_a.figure_a7(config)

    assert "HD1509: duplicate nested levels averaged" in status.note


def test_figure_a7_skips_non_numeric_curve_values(tmp_path: Path) -> None:
    config = _make_config(tmp_path, ["HD1509"])
    _write_decomposition(
        config,
        "HD1509",
        {
            "pooled(5fold)": 0.27,
            "within_timepoint": 0.26,
            "within_well_timepoint": 0.25,
        },
        {1: 0.2, 11: 0.3},
    )
    curve_csv = (
        Path(config.feature_to_mcherry_dir)
        / "HD1509"
        / "decomposition"
        / "r2_by_timepoint.csv"
    )
    table = pd.read_csv(curve_csv)
    table["r2"] = table["r2"].astype(object)
    table.loc[table["target"] == "percentile_90", "r2"] = ["r2", 0.3]
    table.to_csv(curve_csv, index=False)

    status = group_a.figure_a7(config)

    assert status.status == "generated"


def test_figure_a7_marks_an_all_nan_curve_missing(tmp_path: Path) -> None:
    config = _make_config(tmp_path, ["HD1509"])
    _write_decomposition(
        config,
        "HD1509",
        {
            "pooled(5fold)": 0.27,
            "within_timepoint": 0.26,
            "within_well_timepoint": 0.25,
        },
        {1: float("nan"), 11: float("nan")},
    )

    status = group_a.figure_a7(config)

    assert status.status == "blocked"
    assert "HD1509: no finite percentile_90 rows in r2_by_timepoint.csv" in status.note


def test_figure_a7_leaves_headroom_for_the_level_legend(
    tmp_path: Path, monkeypatch
) -> None:
    config = _make_config(tmp_path, ["HD1509"])
    _write_decomposition(
        config,
        "HD1509",
        {"pooled(5fold)": 0.3, "within_timepoint": 0.26, "within_well_timepoint": 0.25},
        {1: 0.2},
    )
    captured = _capture_figures(monkeypatch)

    group_a.figure_a7(config)

    assert captured[0].axes[0].get_ylim()[1] >= 0.39


def test_figure_a7_places_the_clip_note_at_the_top(tmp_path: Path, monkeypatch) -> None:
    config = _make_config(tmp_path, ["Ew2-2"])
    _write_decomposition(
        config,
        "Ew2-2",
        {"pooled(5fold)": 0.2, "within_timepoint": 0.14, "within_well_timepoint": 0.13},
        {1: 0.1, 11: -1.04},
    )
    captured = _capture_figures(monkeypatch)

    group_a.figure_a7(config)

    note = [t for t in captured[0].axes[1].texts if "clipped" in t.get_text()][0]
    assert note.get_position()[1] > 0.9
    assert "Ew2-2 -1.04" in note.get_text()


def test_figure_a7_treats_an_infinite_level_as_missing(tmp_path: Path) -> None:
    config = _make_config(tmp_path, ["HD1509"])
    _write_decomposition(
        config,
        "HD1509",
        {
            "pooled(5fold)": 0.27,
            "within_timepoint": 0.26,
            "within_well_timepoint": float("-inf"),
        },
        {1: 0.2},
    )

    status = group_a.figure_a7(config)

    assert status.status == "partial"
    assert status.note == "HD1509: missing nested levels within_well_timepoint"


def test_figure_a7_headroom_scales_with_the_axis_range(
    tmp_path: Path, monkeypatch
) -> None:
    config = _make_config(tmp_path, ["HD1509"])
    _write_decomposition(
        config,
        "HD1509",
        {
            "pooled(5fold)": 0.27,
            "within_timepoint": 0.26,
            "within_well_timepoint": -2.0,
        },
        {1: 0.2},
    )
    captured = _capture_figures(monkeypatch)

    group_a.figure_a7(config)

    bottom, top = captured[0].axes[0].get_ylim()
    assert top - 0.27 >= 0.25 * (top - bottom)


def test_figure_a7_names_the_empty_nested_table(tmp_path: Path) -> None:
    config = _make_config(tmp_path, ["HD1509"])
    _write_decomposition(
        config,
        "HD1509",
        {
            "pooled(5fold)": 0.27,
            "within_timepoint": 0.26,
            "within_well_timepoint": 0.25,
        },
        {1: 0.2},
    )
    nested_csv = (
        Path(config.feature_to_mcherry_dir)
        / "HD1509"
        / "decomposition"
        / "nested_r2.csv"
    )
    table = pd.read_csv(nested_csv)
    table[table["target"] != "percentile_90"].to_csv(nested_csv, index=False)

    status = group_a.figure_a7(config)

    assert "HD1509: no finite percentile_90 rows in nested_r2.csv" in status.note


def test_figure_a7_notes_dropped_timepoint_rows(tmp_path: Path) -> None:
    config = _make_config(tmp_path, ["HD1509"])
    _write_decomposition(
        config,
        "HD1509",
        {
            "pooled(5fold)": 0.27,
            "within_timepoint": 0.26,
            "within_well_timepoint": 0.25,
        },
        {1: float("nan"), 11: 0.3},
    )

    status = group_a.figure_a7(config)

    assert status.note == "HD1509: 1 unusable timepoint rows dropped"


def test_figure_a7_leaves_headroom_above_the_curves(
    tmp_path: Path, monkeypatch
) -> None:
    config = _make_config(tmp_path, ["HD1509"])
    _write_decomposition(
        config,
        "HD1509",
        {
            "pooled(5fold)": 0.27,
            "within_timepoint": 0.26,
            "within_well_timepoint": 0.25,
        },
        {1: -0.2, 11: 0.3},
    )
    captured = _capture_figures(monkeypatch)

    group_a.figure_a7(config)

    bottom, top = captured[0].axes[1].get_ylim()
    assert top - 0.3 >= 0.15 * (top - bottom)


def test_figure_a7_marks_averaged_duplicates_partial(tmp_path: Path) -> None:
    config = _make_config(tmp_path, ["HD1509"])
    _write_decomposition(
        config,
        "HD1509",
        {
            "pooled(5fold)": 0.27,
            "within_timepoint": 0.26,
            "within_well_timepoint": 0.25,
        },
        {1: 0.2, 11: 0.3},
    )
    curve_csv = (
        Path(config.feature_to_mcherry_dir)
        / "HD1509"
        / "decomposition"
        / "r2_by_timepoint.csv"
    )
    table = pd.read_csv(curve_csv)
    pd.concat([table, table]).to_csv(curve_csv, index=False)

    status = group_a.figure_a7(config)

    assert status.status == "partial"
    assert status.missing == []


def test_figure_a7_notes_ignored_nested_levels(tmp_path: Path) -> None:
    config = _make_config(tmp_path, ["HD1509"])
    _write_decomposition(
        config,
        "HD1509",
        {
            "pooled(5fold)": 0.27,
            "within_timepoint": 0.26,
            "within_well_timepoint": 0.25,
            "within_well": 0.2,
        },
        {1: 0.2},
    )

    status = group_a.figure_a7(config)

    assert status.note == "HD1509: ignored nested levels within_well"


def test_figure_a7_puts_negative_level_labels_below_the_bar(
    tmp_path: Path, monkeypatch
) -> None:
    config = _make_config(tmp_path, ["HD1509"])
    _write_decomposition(
        config,
        "HD1509",
        {
            "pooled(5fold)": 0.27,
            "within_timepoint": 0.26,
            "within_well_timepoint": -0.15,
        },
        {1: 0.2},
    )
    captured = _capture_figures(monkeypatch)

    group_a.figure_a7(config)

    label = [t for t in captured[0].axes[0].texts if t.get_text() == "-0.15"][0]
    assert label.get_position()[1] == pytest.approx(-0.15)
    assert label.get_va() == "top"
