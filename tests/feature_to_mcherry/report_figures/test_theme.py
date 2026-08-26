"""Tests for feature_to_mcherry.report_figures.theme."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

import pandas as pd

from src.feature_to_mcherry.report_figures.theme import (
    EXPERIMENT_COLORS,
    ordered_percentiles,
    save_fig,
)


def test_experiment_colors_cover_all_five_experiments() -> None:
    assert set(EXPERIMENT_COLORS) == {"HD1509", "SA110", "HD1883", "Ew2-1", "Ew2-2"}


def test_ordered_percentiles_matches_target_columns() -> None:
    """The raw target set, in ascending percentile order (contract.TARGET_COLUMNS)."""
    available = ["percentile_95", "percentile_75", "percentile_90"]
    assert ordered_percentiles(available) == [
        "percentile_75",
        "percentile_90",
        "percentile_95",
    ]


def test_ordered_percentiles_accepts_dmso_normalized_targets() -> None:
    """The regression this helper exists for: a z_-prefixed run must not select zero.

    Filtering against a hard-coded list of raw names returned [] here, and the figure
    builders then crashed on a zero target count instead of degrading.
    """
    available = ["z_percentile_90", "z_percentile_75", "z_percentile_95"]
    assert ordered_percentiles(available) == [
        "z_percentile_75",
        "z_percentile_90",
        "z_percentile_95",
    ]


def test_ordered_percentiles_ignores_non_percentile_columns() -> None:
    available = ["experiment", "z_index", "area", "percentile_90", "r2"]
    assert ordered_percentiles(available) == ["percentile_90"]


def test_ordered_percentiles_is_empty_when_nothing_matches() -> None:
    """Callers must read [] as "cannot draw"; it must not invent a target."""
    assert ordered_percentiles(["area", "mae", "r2"]) == []
    assert ordered_percentiles([]) == []


def test_ordered_percentiles_sorts_numerically_not_lexically() -> None:
    """p100 must sort after p90, which a string sort would reverse."""
    assert ordered_percentiles(["percentile_100", "percentile_90"]) == [
        "percentile_90",
        "percentile_100",
    ]


def test_ordered_percentiles_accepts_pandas_inputs() -> None:
    """Call sites pass DataFrame.columns and Series.unique(), not plain lists."""
    frame = pd.DataFrame(
        {"z_percentile_90": [1.0], "z_percentile_75": [2.0], "experiment": ["Ew2-1"]}
    )
    assert ordered_percentiles(frame.columns) == [
        "z_percentile_75",
        "z_percentile_90",
    ]
    series = pd.Series(["percentile_95", "percentile_75", "percentile_75"])
    assert ordered_percentiles(series.unique()) == [
        "percentile_75",
        "percentile_95",
    ]


def test_save_fig_writes_all_requested_formats(tmp_path: Path) -> None:
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    written = save_fig(fig, tmp_path, "smoke_test", formats=("png", "pdf"))

    assert len(written) == 2
    for path in written:
        assert path.exists()
        assert path.stat().st_size > 0
    assert {p.suffix for p in written} == {".png", ".pdf"}
