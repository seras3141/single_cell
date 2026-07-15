"""Interactive (Plotly) per-well, per-timepoint median+IQR report for feature and
target columns — generalizes the ad hoc well/timepoint chart investigation that
surfaced the timepoint=11 / E07-well findings during the feature_to_mcherry
real-data run. Self-contained HTML (plotly.js inlined once, no CDN calls) so the
output works under the Claude Artifact tool's CSP.
"""

from __future__ import annotations

import fnmatch
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

from ..data.contract import CELL_KEY

WELL_PALETTE = [
    "#e63946",
    "#f3722c",
    "#f8961e",
    "#f9c74f",
    "#90be6d",
    "#43aa8b",
    "#4d908e",
    "#577590",
    "#277da1",
    "#6a4c93",
    "#b5838d",
    "#9d4edd",
    "#c9184a",
    "#6d6875",
    "#2a9d8f",
]

SourceData = Tuple[str, pd.DataFrame, pd.DataFrame]  # (label, features_df, targets_df)


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _feature_columns(df: pd.DataFrame) -> List[str]:
    return [
        column
        for column in df.columns
        if column not in CELL_KEY and pd.api.types.is_numeric_dtype(df[column])
    ]


def _resolve_feature_groups(
    available_columns: Sequence[str],
    feature_column_groups: Optional[Dict[str, List[str]]],
) -> List[Tuple[str, List[str]]]:
    """Return [(group_label, matched_columns), ...] in a stable order.

    If ``feature_column_groups`` is None, every available column goes into a single
    ungrouped "Features" bucket — this is what makes the report usable out of the
    box on an experiment with unfamiliar column names.
    """
    if not feature_column_groups:
        return [("Features", list(available_columns))]

    groups = []
    for label, patterns in feature_column_groups.items():
        matched = [
            column
            for column in available_columns
            if any(
                fnmatch.fnmatchcase(column.lower(), pattern.lower())
                for pattern in patterns
            )
        ]
        if matched:
            groups.append((label, matched))
    return groups


def _assign_well_colors(sources_data: Sequence[SourceData]) -> Dict[str, str]:
    all_wells = set()
    for _, features_df, _ in sources_data:
        if "sample_id" in features_df.columns:
            all_wells |= set(features_df["sample_id"].unique())
    return {
        well: WELL_PALETTE[i % len(WELL_PALETTE)]
        for i, well in enumerate(sorted(all_wells))
    }


def _build_figure(
    df: pd.DataFrame,
    value_col: str,
    well_color: Dict[str, str],
    flag_timepoints: Sequence[int],
) -> go.Figure:
    fig = go.Figure()

    for well, sub in sorted(df.groupby("sample_id"), key=lambda kv: kv[0]):
        sub = sub.dropna(subset=[value_col])
        if sub.empty:
            continue
        agg = (
            sub.groupby("timepoint")[value_col]
            .quantile([0.25, 0.5, 0.75])
            .unstack()
            .sort_index()
        )
        if agg.empty:
            continue
        color = well_color.get(well, "#888888")
        n = len(sub)

        x = agg.index.tolist()
        fig.add_trace(
            go.Scatter(
                x=x + x[::-1],
                y=agg[0.75].tolist() + agg[0.25].tolist()[::-1],
                fill="toself",
                fillcolor=_hex_to_rgba(color, 0.14),
                line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip",
                showlegend=False,
                legendgroup=well,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=agg[0.5].tolist(),
                mode="lines+markers",
                name=well,
                legendgroup=well,
                line=dict(color=color, width=2),
                marker=dict(size=4),
                customdata=[n] * len(x),
                hovertemplate=(
                    f"<b>{well}</b><br>timepoint=%{{x}}<br>"
                    f"median {value_col}=%{{y:.3g}}<br>"
                    "well n=%{customdata}<extra></extra>"
                ),
            )
        )

    for timepoint in flag_timepoints:
        fig.add_vline(x=timepoint, line=dict(color="#888888", width=1, dash="dash"))

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=50, r=10, t=10, b=40),
        height=340,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=11, color="#7a8783"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=9),
        ),
        xaxis=dict(
            title="timepoint", gridcolor="rgba(128,128,128,0.15)", zeroline=False
        ),
        yaxis=dict(title=value_col, gridcolor="rgba(128,128,128,0.15)", zeroline=False),
        hoverlabel=dict(bgcolor="rgba(30,30,30,0.85)", font=dict(color="white")),
    )
    return fig


_PAGE_STYLE = """
<style>
  :root {
    --bg: #f6f4ee;
    --surface: #ffffff;
    --text: #1b2422;
    --text-dim: #5c6864;
    --accent-a: #c9184a;
    --accent-b: #0f7a8c;
    --border: #e1ddd2;
    --mono: ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", Menlo,
      monospace;
    --sans: ui-sans-serif, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #0c1416;
      --surface: #131d20;
      --text: #e7ece7;
      --text-dim: #93a6a1;
      --accent-a: #ff5d7a;
      --accent-b: #57c7d4;
      --border: #223034;
    }
  }
  :root[data-theme="dark"] {
    --bg: #0c1416;
    --surface: #131d20;
    --text: #e7ece7;
    --text-dim: #93a6a1;
    --accent-a: #ff5d7a;
    --accent-b: #57c7d4;
    --border: #223034;
  }
  :root[data-theme="light"] {
    --bg: #f6f4ee;
    --surface: #ffffff;
    --text: #1b2422;
    --text-dim: #5c6864;
    --accent-a: #c9184a;
    --accent-b: #0f7a8c;
    --border: #e1ddd2;
  }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    line-height: 1.5;
  }
  .page {
    max-width: 1180px;
    margin: 0 auto;
    padding: 2rem 1.5rem 5rem;
  }
  .masthead {
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
  }
  .eyebrow {
    font-family: var(--mono);
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-dim);
  }
  h1 {
    font-size: 1.7rem;
    margin: 0;
    text-wrap: balance;
    letter-spacing: -0.01em;
  }
  .lede {
    color: var(--text-dim);
    max-width: 68ch;
    font-size: 0.95rem;
  }
  .lede code {
    font-family: var(--mono);
    background: var(--border);
    padding: 0.05em 0.35em;
    border-radius: 3px;
    font-size: 0.85em;
  }
  .flag-note {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    font-family: var(--mono);
    font-size: 0.78rem;
    color: var(--text-dim);
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.4rem 0.7rem;
    width: fit-content;
  }
  .flag-dash {
    display: inline-block;
    width: 1.1rem;
    border-top: 1px dashed #888888;
  }
  .well-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-top: 0.5rem;
  }
  .well-chip {
    font-family: var(--mono);
    font-size: 0.72rem;
    letter-spacing: 0.02em;
    padding: 0.15rem 0.5rem;
    border-radius: 999px;
    background: color-mix(in srgb, var(--chip-color) 22%, var(--surface));
    color: var(--text);
    border: 1px solid color-mix(in srgb, var(--chip-color) 55%, transparent);
    font-variant-numeric: tabular-nums;
  }
  .toc {
    position: sticky;
    top: 0;
    z-index: 5;
    background: color-mix(in srgb, var(--bg) 88%, transparent);
    backdrop-filter: blur(6px);
    border-bottom: 1px solid var(--border);
    padding: 0.7rem 1.5rem;
    display: flex;
    gap: 1.2rem;
    align-items: baseline;
    flex-wrap: wrap;
    margin: 0 -1.5rem 2rem;
  }
  .toc-sample {
    font-family: var(--mono);
    font-size: 0.82rem;
    letter-spacing: 0.04em;
    color: var(--text);
    text-decoration: none;
    border-bottom: 2px solid var(--accent-b);
    padding-bottom: 0.1rem;
  }
  .sample-section {
    margin-bottom: 3rem;
  }
  .sample-title {
    font-size: 1.25rem;
    display: flex;
    align-items: baseline;
    gap: 0.7rem;
    flex-wrap: wrap;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.6rem;
  }
  .sample-meta {
    font-family: var(--mono);
    font-size: 0.78rem;
    color: var(--text-dim);
    font-weight: 400;
    font-variant-numeric: tabular-nums;
  }
  .category {
    margin: 1.6rem 0 2.2rem;
  }
  .category-title {
    font-family: var(--mono);
    font-size: 0.78rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin: 0 0 0.2rem;
  }
  .category-title.accent-a {
    color: var(--accent-a);
  }
  .category-title.accent-b {
    color: var(--accent-b);
  }
  .chart-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
    gap: 1rem;
  }
  .chart-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.6rem 0.7rem 0.4rem;
    margin: 0;
    overflow-x: auto;
  }
  .chart-card figcaption {
    font-family: var(--mono);
    font-size: 0.78rem;
    color: var(--text);
    margin-bottom: 0.2rem;
  }
  footer {
    color: var(--text-dim);
    font-size: 0.8rem;
    margin-top: 3rem;
    border-top: 1px solid var(--border);
    padding-top: 1rem;
  }
</style>
"""


def build_interactive_report_html(
    sources_data: Sequence[SourceData],
    target_columns: Sequence[str],
    feature_column_groups: Optional[Dict[str, List[str]]] = None,
    flag_timepoints: Optional[Sequence[int]] = None,
    title: str = "Per-well, per-timepoint feature & target distributions",
) -> str:
    """Build a self-contained interactive HTML report.

    Parameters
    ----------
    sources_data : sequence of (label, features_df, targets_df)
        One entry per source/experiment. ``features_df``/``targets_df`` are the
        outputs of :func:`feature_to_mcherry.data.loaders.load_features`/
        ``load_targets`` (or their directory-loading counterparts).
    target_columns : sequence of str
        mCherry percentile columns to chart from each source's ``targets_df``.
    feature_column_groups : dict[str, list[str]], optional
        Category label -> ``fnmatch`` patterns for grouping feature-column charts.
        ``None`` puts every feature column in one ungrouped "Features" bucket.
    flag_timepoints : sequence of int, optional
        Timepoints to mark with a dashed vertical reference line on every chart.
    title : str
        Page heading.

    Returns
    -------
    str
        Self-contained HTML fragment (no ``<!DOCTYPE>``/``<html>``/``<head>``/
        ``<body>`` — ready to wrap or publish directly via the Artifact tool).
    """
    flag_timepoints = list(flag_timepoints or [])
    well_color = _assign_well_colors(sources_data)

    sections_html = []
    toc_html = []

    for label, features_df, targets_df in sources_data:
        toc_html.append(f'<a href="#source-{label}" class="toc-sample">{label}</a>')
        n_wells = (
            features_df["sample_id"].nunique()
            if "sample_id" in features_df.columns
            else 0
        )
        n_cells = len(features_df)
        sections_html.append(
            f'<section class="sample-section" id="source-{label}">'
            f'<h2 class="sample-title">{label} <span class="sample-meta">'
            f"{n_cells:,} cells &middot; {n_wells} wells</span></h2>"
        )

        feature_groups = _resolve_feature_groups(
            _feature_columns(features_df), feature_column_groups
        )
        for group_label, columns in feature_groups:
            sections_html.append(
                '<div class="category">'
                f'<h3 class="category-title accent-b">{group_label}</h3>'
                '<div class="chart-grid">'
            )
            for column in columns:
                fig = _build_figure(features_df, column, well_color, flag_timepoints)
                div_id = f"chart-{label}-{group_label}-{column}".replace(" ", "_")
                div = fig.to_html(
                    full_html=False,
                    include_plotlyjs=False,
                    div_id=div_id,
                    config={"displaylogo": False, "responsive": True},
                )
                sections_html.append(
                    f'<figure class="chart-card"><figcaption>{column}</figcaption>'
                    f"{div}</figure>"
                )
            sections_html.append("</div></div>")

        available_targets = [c for c in target_columns if c in targets_df.columns]
        if available_targets:
            sections_html.append(
                '<div class="category"><h3 class="category-title accent-a">'
                'mCherry activity targets</h3><div class="chart-grid">'
            )
            for column in available_targets:
                fig = _build_figure(targets_df, column, well_color, flag_timepoints)
                div_id = f"chart-{label}-target-{column}".replace(" ", "_")
                div = fig.to_html(
                    full_html=False,
                    include_plotlyjs=False,
                    div_id=div_id,
                    config={"displaylogo": False, "responsive": True},
                )
                sections_html.append(
                    f'<figure class="chart-card"><figcaption>{column}</figcaption>'
                    f"{div}</figure>"
                )
            sections_html.append("</div></div>")

        sections_html.append("</section>")

    legend_chips = "".join(
        f'<span class="well-chip" style="--chip-color:{color}">{well}</span>'
        for well, color in well_color.items()
    )
    flag_note = ""
    if flag_timepoints:
        flagged = ", ".join(f"<code>{t}</code>" for t in flag_timepoints)
        flag_note = (
            '<span class="flag-note"><span class="flag-dash"></span> dashed line(s) '
            f"mark timepoint(s) {flagged}</span>"
        )

    return f"""
<title>{title}</title>
<script>{get_plotlyjs()}</script>
{_PAGE_STYLE}
<div class="page">
  <div class="masthead">
    <span class="eyebrow">feature_to_mcherry data_quality diagnostic</span>
    <h1>{title}</h1>
    <p class="lede">
      Median (line) and interquartile range (band) of every feature and target
      column, grouped by well, plotted across <code>timepoint</code>. Hover a point
      for the exact value; click a legend entry to isolate/hide a well.
    </p>
    {flag_note}
    <div class="well-legend">{legend_chips}</div>
  </div>
  <nav class="toc">{"".join(toc_html)}</nav>
  {"".join(sections_html)}
  <footer>
    Generated by feature_to_mcherry.data_quality.plots.build_interactive_report_html
  </footer>
</div>
"""
