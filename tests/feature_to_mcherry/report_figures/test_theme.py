"""Tests for feature_to_mcherry.report_figures.theme."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from src.feature_to_mcherry.report_figures.theme import (
    EXPERIMENT_COLORS,
    PERCENTILE_ORDER,
    save_fig,
)


def test_experiment_colors_cover_all_five_experiments() -> None:
    assert set(EXPERIMENT_COLORS) == {"HD1509", "SA110", "HD1883", "Ew2-1", "Ew2-2"}


def test_percentile_order_matches_target_columns() -> None:
    assert PERCENTILE_ORDER == ["percentile_75", "percentile_90", "percentile_95"]


def test_save_fig_writes_all_requested_formats(tmp_path: Path) -> None:
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    written = save_fig(fig, tmp_path, "smoke_test", formats=("png", "pdf"))

    assert len(written) == 2
    for path in written:
        assert path.exists()
        assert path.stat().st_size > 0
    assert {p.suffix for p in written} == {".png", ".pdf"}
