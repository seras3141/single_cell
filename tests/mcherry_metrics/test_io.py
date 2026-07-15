"""I/O contract tests for mCherry metrics outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.mcherry_metrics.io.exporters import (
    INSTANCE_METRICS_COLUMNS,
    validate_metrics_dataframe,
    write_instance_metrics,
)


def test_write_instance_metrics_preserves_contract_order(tmp_path: Path) -> None:
    metrics_df = pd.DataFrame(
        {
            "image_path": ["/tmp/p2426_A01_t1_z1_mCherry.tif"],
            "label_path": ["/tmp/p2426_A01_t1_z1_Cells.tif"],
            "cell_id": [1],
            "area": [20],
            "mean_intensity": [100.0],
            "max_intensity": [150.0],
            "min_intensity": [80.0],
            "sum_intensity": [2000.0],
            "percentile_75": [120.0],
            "percentile_90": [140.0],
            "percentile_95": [145.0],
            "sample_id": ["A01"],
            "z_index": [1],
            "timepoint": ["1"],
            "image": ["p2426_A01_t1_z1_mCherry.tif"],
            "sample": ["A01"],
            "time": ["1"],
            "ID": ["p2426_A01_t1"],
            "percentile_99": [149.0],
        }
    )

    validate_metrics_dataframe(metrics_df)

    output_path = tmp_path / "instance_metrics.csv"
    write_instance_metrics(metrics_df, output_path)

    written = pd.read_csv(output_path)
    assert list(written.columns[: len(INSTANCE_METRICS_COLUMNS)]) == INSTANCE_METRICS_COLUMNS
    assert written.columns[-1] == "percentile_99"