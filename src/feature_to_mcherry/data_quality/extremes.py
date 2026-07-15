"""Extreme-value clustering diagnostic: for every numeric feature/target column,
flag rows outside a [lo, hi] quantile range and report whether extremes concentrate
by well (sample_id), z-slice, or timepoint — generalizing the ad hoc investigation
that surfaced the timepoint=11 / E07-well findings during the feature_to_mcherry
real-data run.
"""

from __future__ import annotations

import logging
from typing import List

import pandas as pd

from ..data.contract import CELL_KEY

logger = logging.getLogger(__name__)

GROUP_TYPES = ["sample_id", "z_index", "timepoint"]

REPORT_COLUMNS = [
    "source",
    "value_column",
    "group_type",
    "group_value",
    "n",
    "n_extreme",
    "extreme_rate",
    "overall_rate",
    "enrichment",
]


def _value_columns(df: pd.DataFrame) -> List[str]:
    return [
        column
        for column in df.columns
        if column not in CELL_KEY and pd.api.types.is_numeric_dtype(df[column])
    ]


def _report_for_table(
    df: pd.DataFrame, source: str, quantile_lo: float, quantile_hi: float
) -> pd.DataFrame:
    rows = []
    for column in _value_columns(df):
        series = df[column]
        non_null = series.dropna()
        if non_null.empty:
            logger.warning(
                "Skipping %r for source %r: column is entirely NaN", column, source
            )
            continue

        q_lo, q_hi = non_null.quantile([quantile_lo, quantile_hi])
        extreme = (series < q_lo) | (series > q_hi)
        overall_rate = extreme.mean()

        for group_type in GROUP_TYPES:
            if group_type not in df.columns:
                continue
            grouped = extreme.groupby(df[group_type], observed=True).agg(
                ["mean", "sum", "count"]
            )
            for group_value, row in grouped.iterrows():
                enrichment = (
                    row["mean"] / overall_rate if overall_rate > 0 else float("nan")
                )
                rows.append(
                    {
                        "source": source,
                        "value_column": column,
                        "group_type": group_type,
                        "group_value": group_value,
                        "n": int(row["count"]),
                        "n_extreme": int(row["sum"]),
                        "extreme_rate": row["mean"],
                        "overall_rate": overall_rate,
                        "enrichment": enrichment,
                    }
                )

    return pd.DataFrame(rows, columns=REPORT_COLUMNS)


def compute_extreme_value_report(
    features_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    source_label: str,
    quantile_lo: float = 0.001,
    quantile_hi: float = 0.999,
) -> pd.DataFrame:
    """Flag extreme values in every feature/target column and report clustering.

    Parameters
    ----------
    features_df : pd.DataFrame
        Output of :func:`feature_to_mcherry.data.loaders.load_features` (or
        ``load_features_from_directory``) — ``CELL_KEY + feature columns``.
    targets_df : pd.DataFrame
        Output of :func:`feature_to_mcherry.data.loaders.load_targets` (or
        ``load_targets_from_directory``) — ``CELL_KEY + target columns``.
    source_label : str
        Identifier for this source, recorded in the ``source`` column (e.g. the
        experiment/sample name).
    quantile_lo, quantile_hi : float
        A row is "extreme" for a given column if its value falls outside
        ``[quantile(lo), quantile(hi)]`` of that column's *own* non-null values in
        this source (thresholds are per-source, not pooled across sources).

    Returns
    -------
    pd.DataFrame
        Long-form, one row per (value_column, group_type, group_value):
        ``source, value_column, group_type ("sample_id"|"z_index"|"timepoint"),
        group_value, n, n_extreme, extreme_rate, overall_rate, enrichment``.
        ``enrichment`` is ``extreme_rate / overall_rate`` for that column — values
        well above 1 indicate that group concentrates extreme values relative to
        the source as a whole.
    """
    feature_report = _report_for_table(
        features_df, source_label, quantile_lo, quantile_hi
    )
    target_report = _report_for_table(
        targets_df, source_label, quantile_lo, quantile_hi
    )
    combined = pd.concat([feature_report, target_report], ignore_index=True)
    logger.info(
        "Computed extreme-value report for source %r: %d columns x %d group types, "
        "%d rows",
        source_label,
        combined["value_column"].nunique(),
        len(GROUP_TYPES),
        len(combined),
    )
    return combined
