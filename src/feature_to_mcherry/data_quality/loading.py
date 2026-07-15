"""Shared source-loading helper for the data_quality CLI scripts."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from ..data.loaders import (
    load_features,
    load_features_from_directory,
    load_targets,
    load_targets_from_directory,
)
from .config import SourceConfig


def load_source(
    source: SourceConfig, target_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load one source's feature/target tables, applying its config knobs.

    Mirrors the directory-vs-file auto-detection and column-resolution behavior of
    :func:`feature_to_mcherry.pipeline.run` — a single CSV or a directory of
    per-(well, timepoint, z) CSVs for both ``feature_csv``/``target_csv``.
    """
    feature_path = Path(source.feature_csv)
    if feature_path.is_dir():
        features_df = load_features_from_directory(
            feature_path,
            id_column=source.id_column,
            sample_id_column=source.sample_id_column,
            timepoint_column=source.timepoint_column,
            z_index_column=source.z_index_column,
        )
    else:
        features_df = load_features(
            feature_path,
            id_column=source.id_column,
            sample_id_column=source.sample_id_column,
            timepoint_column=source.timepoint_column,
            z_index_column=source.z_index_column,
        )

    if source.exclude_feature_columns:
        to_drop = [
            column
            for column in source.exclude_feature_columns
            if column in features_df.columns
        ]
        if to_drop:
            features_df = features_df.drop(columns=to_drop)

    target_path = Path(source.target_csv)
    if target_path.is_dir():
        targets_df = load_targets_from_directory(
            target_path, target_columns=target_columns
        )
    else:
        targets_df = load_targets(target_path, target_columns=target_columns)

    return features_df, targets_df
