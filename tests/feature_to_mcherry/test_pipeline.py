"""End-to-end test for feature_to_mcherry.pipeline.run."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from src.feature_to_mcherry.config import FeatureToMcherryConfig
from src.feature_to_mcherry.pipeline import run


def _write_synthetic_csvs(
    tmp_path: Path, n_per_group: int = 20, seed: int = 0
) -> Tuple[Path, Path]:
    rng = np.random.default_rng(seed)
    sample_ids = ["A01", "A02"]

    rows_features = []
    rows_targets = []
    for sample_id in sample_ids:
        f1 = rng.uniform(0, 10, size=n_per_group)
        f2 = rng.uniform(0, 10, size=n_per_group)
        noise = rng.normal(0, 0.5, size=n_per_group)
        base = 2 * f1 + 3 * f2 + noise

        for i in range(n_per_group):
            label_id = i + 1
            rows_features.append(
                {
                    "instance_id": label_id,
                    "well": sample_id,
                    "frame": 1,
                    "feature_1": f1[i],
                    "feature_2": f2[i],
                }
            )
            rows_targets.append(
                {
                    "sample_id": sample_id,
                    "timepoint": 1,
                    "label_id": label_id,
                    "percentile_75": base[i],
                    "percentile_90": base[i] + 5.0,
                    "percentile_95": base[i] + 10.0,
                }
            )

    feature_csv = tmp_path / "features.csv"
    target_csv = tmp_path / "instance_metrics.csv"
    pd.DataFrame(rows_features).to_csv(feature_csv, index=False)
    pd.DataFrame(rows_targets).to_csv(target_csv, index=False)
    return feature_csv, target_csv


def test_pipeline_end_to_end_produces_finite_metrics_for_both_models(
    tmp_path: Path,
) -> None:
    feature_csv, target_csv = _write_synthetic_csvs(tmp_path)
    output_dir = tmp_path / "results"

    config = FeatureToMcherryConfig(
        feature_csv=str(feature_csv),
        target_csv=str(target_csv),
        id_column="instance_id",
        sample_id_column="well",
        timepoint_column="frame",
        group_by="sample_id",
        n_splits=2,
        output_dir=str(output_dir),
    )

    results = run(config)

    assert results.n_cells == 40
    assert results.n_features == 2
    assert set(results.feature_names) == {"feature_1", "feature_2"}

    for model_result in (results.ridge, results.linear_quantile):
        assert len(model_result.pooled_metrics) == 3
        for metrics in model_result.pooled_metrics:
            assert math.isfinite(metrics["mae"])
            assert math.isfinite(metrics["r2"])
            assert math.isfinite(metrics["pinball_loss"])
        assert math.isfinite(model_result.pooled_crossing_rate)

    assert (output_dir / "baseline_ladder.csv").exists()
    assert (output_dir / "baseline_ladder.md").exists()

    report = pd.read_csv(output_dir / "baseline_ladder.csv")
    assert set(report["model"]) == {"ridge", "linear_quantile"}
    assert len(report) == 6  # 2 models x 3 targets
