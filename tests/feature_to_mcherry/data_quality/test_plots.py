"""Tests for feature_to_mcherry.data_quality.plots."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.feature_to_mcherry.data_quality.plots import build_interactive_report_html


def _synthetic_source(label: str, wells, seed: int):
    rng = np.random.default_rng(seed)
    n_per_well = 20

    rows_features = []
    rows_targets = []
    cell_id = 0
    for well in wells:
        for i in range(n_per_well):
            rows_features.append(
                {
                    "sample_id": well,
                    "timepoint": 1 if i % 2 == 0 else 11,
                    "z_index": 1,
                    "cell_id": cell_id,
                    "area": rng.uniform(0, 10),
                    "perimeter": rng.uniform(0, 10),
                    "mean_intensity": rng.uniform(0, 10),
                }
            )
            rows_targets.append(
                {
                    "sample_id": well,
                    "timepoint": 1 if i % 2 == 0 else 11,
                    "z_index": 1,
                    "cell_id": cell_id,
                    "percentile_75": rng.uniform(0, 100),
                }
            )
            cell_id += 1
    return label, pd.DataFrame(rows_features), pd.DataFrame(rows_targets)


def test_build_interactive_report_html_contains_sources_and_groups() -> None:
    sources_data = [
        _synthetic_source("SrcA", ["A01", "A02"], seed=0),
        _synthetic_source("SrcB", ["A01", "B02"], seed=1),
    ]

    html = build_interactive_report_html(
        sources_data,
        target_columns=["percentile_75"],
        feature_column_groups={
            "Size & shape": ["area", "perimeter"],
            "Intensity": ["mean_intensity"],
        },
        flag_timepoints=[11],
    )

    assert "SrcA" in html
    assert "SrcB" in html
    assert "Size &amp; shape" in html or "Size & shape" in html
    assert "Intensity" in html
    assert "percentile_75" in html
    assert "A01" in html  # well shared between sources appears
    # plotly.js is inlined (self-contained) rather than loaded from a CDN — check
    # for the absence of an external <script src="..."> tag, not the substring
    # "cdn.plot.ly" (which appears harmlessly inside plotly.js's own inlined
    # source, e.g. in embedded doc-string URLs).
    assert "Plotly" in html
    assert '<script src="https://cdn' not in html
    assert '<script src="http://cdn' not in html


def test_build_interactive_report_html_defaults_to_single_ungrouped_bucket() -> None:
    sources_data = [_synthetic_source("SrcA", ["A01"], seed=0)]

    html = build_interactive_report_html(
        sources_data, target_columns=["percentile_75"], feature_column_groups=None
    )

    assert "Features" in html
    assert "area" in html
    assert "mean_intensity" in html


def test_build_interactive_report_html_shared_well_gets_same_color() -> None:
    sources_data = [
        _synthetic_source("SrcA", ["A01", "A02"], seed=0),
        _synthetic_source("SrcB", ["A01", "B02"], seed=1),
    ]

    html = build_interactive_report_html(sources_data, target_columns=["percentile_75"])

    # Both sources' A01 traces should reference the same palette color.
    assert (
        html.count("#e63946") >= 2
    )  # first palette color, assigned to sorted-first well "A01"
