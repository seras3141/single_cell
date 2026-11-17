"""Rebuild the feature matrix a feature_to_mcherry run was fitted on."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

from ..config import FeatureToMcherryConfig
from ..data.join import build_matrix_with_metadata
from ..data.loaders import (
    load_features,
    load_features_from_directory,
    load_targets,
    load_targets_from_directory,
)
from ..data.normalize import apply_dmso_normalization


def drop_wells(
    features_df: pd.DataFrame, targets_df: pd.DataFrame, wells: Sequence[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Both tables without the rows of ``wells``; a well in neither is an error."""
    excluded = set(wells)
    if not excluded:
        return features_df, targets_df
    present = set(features_df["sample_id"].astype(str)) | set(
        targets_df["sample_id"].astype(str)
    )
    unknown = sorted(excluded - present)
    if unknown:
        raise ValueError(f"wells to exclude are in neither table: {unknown}")
    return (
        features_df[~features_df["sample_id"].astype(str).isin(excluded)],
        targets_df[~targets_df["sample_id"].astype(str).isin(excluded)],
    )


def load_run_matrix(
    config: FeatureToMcherryConfig, exclude_wells: Sequence[str] = ()
) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    """``X``, ``y``, feature names and per-row ``CELL_KEY`` metadata of a run.

    Follows the load, normalise, exclude and join steps of ``pipeline.run``.
    ``exclude_wells`` are removed from both tables after normalisation, so a DMSO
    reference still sees its own well.
    """
    target_path = Path(config.target_csv)
    if target_path.is_dir():
        targets_df = load_targets_from_directory(
            target_path, target_columns=config.target_columns
        )
    else:
        targets_df = load_targets(target_path, target_columns=config.target_columns)
    targets_df, target_columns = apply_dmso_normalization(
        targets_df,
        enabled=config.normalize_to_dmso,
        dmso_well=config.dmso_well,
        target_columns=config.target_columns,
        min_peak_fraction=config.dmso_gate_min_peak_fraction,
        absolute_floor=config.dmso_gate_absolute_floor,
    )
    feature_path = Path(config.feature_csv)
    feature_arguments = dict(
        id_column=config.id_column,
        sample_id_column=config.sample_id_column,
        timepoint_column=config.timepoint_column,
        z_index_column=config.z_index_column,
    )
    if feature_path.is_dir():
        features_df = load_features_from_directory(
            feature_path, pattern=config.feature_pattern, **feature_arguments
        )
    else:
        features_df = load_features(feature_path, **feature_arguments)
    features_df = features_df.drop(
        columns=[c for c in config.exclude_feature_columns if c in features_df.columns]
    )
    features_df, targets_df = drop_wells(features_df, targets_df, exclude_wells)
    X, y, _, feature_names, metadata = build_matrix_with_metadata(
        features_df,
        targets_df,
        target_columns=target_columns,
        group_by=config.group_by,
    )
    return X, y, feature_names, metadata
