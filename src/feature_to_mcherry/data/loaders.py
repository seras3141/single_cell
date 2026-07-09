"""Loaders for feature tables and mcherry_metrics target tables."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.utils.file_utils import ConfigurableFileHandler

from .contract import CELL_KEY, TARGET_COLUMNS

logger = logging.getLogger(__name__)

_PROVENANCE_COLUMNS = {
    "image_filename",
    "mask_filename",
    "processing_timestamp",
    "feature_extraction_version",
    "dataset_name",
}


def load_targets(
    csv_path: Path, target_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """Load the mcherry_metrics instance-metrics CSV, keeping only cell_key + targets.

    Parameters
    ----------
    csv_path : Path
        Path to an instance-metrics CSV produced by ``src.mcherry_metrics``.
    target_columns : list[str], optional
        Percentile columns to keep as regression targets. Defaults to
        ``TARGET_COLUMNS`` (``percentile_75``, ``percentile_90``, ``percentile_95``).

    Returns
    -------
    pd.DataFrame
        Columns: ``CELL_KEY + target_columns``. Rows with a NaN target are dropped.
    """
    target_columns = list(target_columns) if target_columns else list(TARGET_COLUMNS)
    df = pd.read_csv(csv_path)

    missing = [
        column for column in CELL_KEY + target_columns if column not in df.columns
    ]
    if missing:
        raise ValueError(
            f"Target CSV {csv_path} is missing required columns: {missing}"
        )

    df = df[CELL_KEY + target_columns].copy()

    n_before = len(df)
    df = df.dropna(subset=target_columns)
    n_dropped = n_before - len(df)
    if n_dropped:
        logger.warning(
            "Dropped %d/%d target rows with NaN target values from %s",
            n_dropped,
            n_before,
            csv_path,
        )

    return df.reset_index(drop=True)


def load_features(
    csv_path: Path,
    id_column: str,
    sample_id_column: Optional[str] = None,
    timepoint_column: Optional[str] = None,
    image_filename_column: str = "image_filename",
    file_handler: Optional[ConfigurableFileHandler] = None,
) -> pd.DataFrame:
    """Load a per-cell feature CSV, normalized to CELL_KEY + feature columns.

    Works generically across feature-extraction backends (regionprops, pyradiomics,
    incarta, or any future source) via the ``id_column``/``sample_id_column``/
    ``timepoint_column`` knobs, rather than backend-specific branches.

    Parameters
    ----------
    csv_path : Path
        Path to a per-cell feature CSV.
    id_column : str
        Name of the per-cell id column in this CSV (e.g. ``"instance_id"`` for the
        regionprops backend). Renamed to ``label_id`` internally.
    sample_id_column, timepoint_column : str, optional
        Column names already carrying ``sample_id``/``timepoint`` in this CSV. If
        ``None``, the corresponding value is derived from ``image_filename`` using
        :class:`ConfigurableFileHandler`, mirroring
        ``src.mcherry_metrics.io.loaders.extract_image_metadata``.
    image_filename_column : str
        Column holding the source image filename, used for derivation when
        ``sample_id_column``/``timepoint_column`` are ``None``.
    file_handler : ConfigurableFileHandler, optional
        Handler used for filename parsing. Defaults to a plain
        ``ConfigurableFileHandler()``.

    Returns
    -------
    pd.DataFrame
        Columns: ``CELL_KEY + <numeric feature columns>``.
    """
    df = pd.read_csv(csv_path)

    if id_column not in df.columns:
        raise ValueError(
            f"id_column {id_column!r} not found in feature CSV {csv_path}; "
            f"available columns: {list(df.columns)}"
        )
    df = df.rename(columns={id_column: "label_id"})

    if sample_id_column is not None:
        df = df.rename(columns={sample_id_column: "sample_id"})
    if timepoint_column is not None:
        df = df.rename(columns={timepoint_column: "timepoint"})

    needs_sample_id = sample_id_column is None
    needs_timepoint = timepoint_column is None
    if needs_sample_id or needs_timepoint:
        if image_filename_column not in df.columns:
            raise ValueError(
                f"Cannot derive sample_id/timepoint: {image_filename_column!r} column "
                f"not found in {csv_path}, and sample_id_column/timepoint_column were "
                "not both provided."
            )
        handler = file_handler or ConfigurableFileHandler()
        filenames = df[image_filename_column].astype(str)

        if needs_sample_id:
            df["sample_id"] = filenames.map(
                lambda name: handler.extract_sample_id(name) or ""
            )
        if needs_timepoint:

            def _timepoint(name: str) -> str:
                timepoint = handler.extract_time_point(name)
                return "" if timepoint == "unknown" else str(timepoint)

            df["timepoint"] = filenames.map(_timepoint)

    id_and_key_columns = {"label_id", "sample_id", "timepoint"}
    feature_columns = [
        column
        for column in df.columns
        if column not in _PROVENANCE_COLUMNS
        and column not in id_and_key_columns
        and pd.api.types.is_numeric_dtype(df[column])
    ]

    if not feature_columns:
        raise ValueError(
            f"No numeric feature columns detected in {csv_path} after excluding "
            "id/provenance columns."
        )

    logger.info(
        "Loaded %d cells, %d feature columns from %s",
        len(df),
        len(feature_columns),
        csv_path,
    )

    return df[CELL_KEY + feature_columns].copy()
