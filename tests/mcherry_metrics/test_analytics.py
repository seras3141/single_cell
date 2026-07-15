"""Analytics output tests for mCherry metrics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.mcherry_metrics.analytics import generate_standard_outputs


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