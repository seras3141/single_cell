"""Analytics output tests for mCherry metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from src.mcherry_metrics.analytics import distributions, generate_standard_outputs


def _metrics_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_path": [
                "/tmp/p2426_A01_t1_z1_mCherry.tif",
                "/tmp/p2426_A02_t1_z1_mCherry.tif",
            ],
            "label_path": [
                "/tmp/p2426_A01_t1_z1_Cells.tif",
                "/tmp/p2426_A02_t1_z1_Cells.tif",
            ],
            "label_id": [1, 1],
            "area": [20, 30],
            "mean_intensity": [100.0, 200.0],
            "max_intensity": [150.0, 250.0],
            "min_intensity": [80.0, 150.0],
            "sum_intensity": [2000.0, 6000.0],
            "percentile_75": [120.0, 220.0],
            "percentile_90": [140.0, 240.0],
            "percentile_95": [145.0, 245.0],
            "sample_id": ["A01", "A02"],
            "z_index": [1, 1],
            "timepoint": ["1", "1"],
            "image": [
                "p2426_A01_t1_z1_mCherry.tif",
                "p2426_A02_t1_z1_mCherry.tif",
            ],
            "sample": ["A01", "A02"],
            "time": ["1", "1"],
            "ID": ["p2426_A01_t1", "p2426_A02_t1"],
        }
    )


def test_generate_standard_outputs_writes_expected_files(tmp_path: Path) -> None:
    outputs = generate_standard_outputs(
        _metrics_df(),
        tmp_path,
        processed_image_paths=[
            Path("/tmp/p2426_A01_t1_z1_mCherry.tif"),
            Path("/tmp/p2426_A02_t1_z1_mCherry.tif"),
        ],
    )

    expected = {
        "distribution_mean_intensity.png",
        "violin_mean_intensity_by_sample.png",
        "metric_correlation_heatmap.png",
        "metrics_summary.csv",
        "qc_report.txt",
        "area_vs_intensity.png",
    }
    output_names = {path.name for path in outputs}
    assert expected.issubset(output_names)

    for path in outputs:
        assert path.exists()
        assert path.stat().st_size > 0


@pytest.fixture
def closed_figures(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[Figure]]:
    real_close = plt.close
    captured: list[Figure] = []
    monkeypatch.setattr(distributions.plt, "close", captured.append)
    yield captured
    real_close("all")


def _two_groups(n_large: int, n_small: int, small_sigma: float) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "percentile_90": np.concatenate(
                [rng.normal(600, 50, n_large), rng.normal(200, small_sigma, n_small)]
            ),
            "sample_id": ["large"] * n_large + ["small"] * n_small,
        }
    )


def _occupied_bars_below(fig: Figure, x: float) -> list[Rectangle]:
    return [
        bar
        for container in fig.axes[0].containers
        for bar in container
        if bar.get_x() < x and bar.get_height() > 0
    ]


def test_plot_metric_distributions_density_normalises_each_group(
    tmp_path: Path, closed_figures: list[Figure]
) -> None:
    outputs = distributions.plot_metric_distributions(
        _two_groups(5000, 200, 20), tmp_path, stat="density"
    )

    assert [path.name for path in outputs] == ["distribution_percentile_90.png"]
    ax = closed_figures[0].axes[0]
    assert ax.get_ylabel() == "Density (per group)"
    assert len(ax.containers) == 2
    for container in ax.containers:
        area = sum(bar.get_height() * bar.get_width() for bar in container)
        assert area == pytest.approx(1.0, rel=1e-6)


@pytest.mark.parametrize(
    "stat, label",
    [("frequency", "Frequency"), ("probability", "Probability (per group)")],
)
def test_plot_metric_distributions_marks_only_per_group_normalised_stats(
    tmp_path: Path, closed_figures: list[Figure], stat: str, label: str
) -> None:
    distributions.plot_metric_distributions(
        _two_groups(5000, 200, 20), tmp_path, stat=stat  # type: ignore[arg-type]
    )

    assert closed_figures[0].axes[0].get_ylabel() == label


def test_plot_metric_distributions_clips_each_group_to_its_own_tails(
    tmp_path: Path, closed_figures: list[Figure]
) -> None:
    distributions.plot_metric_distributions(
        _two_groups(30000, 15, 5), tmp_path, stat="density", clip_per_group=True
    )

    assert _occupied_bars_below(closed_figures[0], 350)


def test_plot_metric_distributions_defaults_to_pooled_counts(
    tmp_path: Path, closed_figures: list[Figure]
) -> None:
    distributions.plot_metric_distributions(_two_groups(30000, 15, 5), tmp_path)

    assert closed_figures[0].axes[0].get_ylabel() == "Count"
    assert not _occupied_bars_below(closed_figures[0], 350)


def test_plot_metric_distributions_rejects_unknown_stat_before_plotting(
    tmp_path: Path,
) -> None:
    open_before = plt.get_fignums()

    metrics_df = _two_groups(5000, 200, 20)
    save_dir = tmp_path / "plots"

    with pytest.raises(ValueError, match="stat must be one of"):
        distributions.plot_metric_distributions(
            metrics_df, save_dir, stat="densty"  # type: ignore[arg-type]
        )

    assert plt.get_fignums() == open_before
    assert not save_dir.exists()
