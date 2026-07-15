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


def load_targets_from_directory(
    directory: Path,
    target_columns: Optional[List[str]] = None,
    pattern: str = "*.csv",
) -> pd.DataFrame:
    """Load and concatenate a directory of per-(well, timepoint, z) target CSVs.

    Some ``mcherry_metrics`` runs write one CSV per (well, timepoint, z) slice under
    a ``split_data/`` directory, alongside (or instead of) a single combined
    ``instance_metrics.csv``. This globs ``directory`` for matching files, loads each
    with :func:`load_targets`, and concatenates the results into one combined target
    table.

    Parameters
    ----------
    directory : Path
        Directory containing the per-slice target CSVs.
    target_columns : list[str], optional
        Forwarded to :func:`load_targets` for every matched file.
    pattern : str
        Glob pattern selecting the target CSVs within ``directory``.

    Returns
    -------
    pd.DataFrame
        Columns: ``CELL_KEY + target_columns``, concatenated across all matched files.

    Raises
    ------
    ValueError
        If no files match ``pattern`` in ``directory``.
    """
    directory = Path(directory)
    csv_paths = sorted(directory.glob(pattern))
    if not csv_paths:
        raise ValueError(
            f"No files matching pattern {pattern!r} found in directory {directory}"
        )

    frames = [
        load_targets(csv_path, target_columns=target_columns) for csv_path in csv_paths
    ]
    combined = pd.concat(frames, ignore_index=True)

    logger.info(
        "Loaded %d target rows from %d files in %s (pattern %r)",
        len(combined),
        len(csv_paths),
        directory,
        pattern,
    )

    return combined


def load_features(
    csv_path: Path,
    id_column: str,
    sample_id_column: Optional[str] = None,
    timepoint_column: Optional[str] = None,
    z_index_column: Optional[str] = None,
    image_filename_column: str = "image_filename",
    file_handler: Optional[ConfigurableFileHandler] = None,
) -> pd.DataFrame:
    """Load a per-cell feature CSV, normalized to CELL_KEY + feature columns.

    Works generically across feature-extraction backends (regionprops, pyradiomics,
    incarta, or any future source) via the ``id_column``/``sample_id_column``/
    ``timepoint_column``/``z_index_column`` knobs, rather than backend-specific
    branches.

    Parameters
    ----------
    csv_path : Path
        Path to a per-cell feature CSV.
    id_column : str
        Name of the per-cell id column in this CSV (e.g. ``"instance_id"`` for the
        regionprops/incarta backends). Renamed to ``cell_id`` internally.
    sample_id_column, timepoint_column, z_index_column : str, optional
        Column names already carrying ``sample_id``/``timepoint``/``z_index`` in this
        CSV. Resolution order per field: (1) if given explicitly, rename that column;
        (2) else if a column already named ``sample_id``/``timepoint``/``z_index``
        exists (e.g. the incarta ``split_data`` layout, which already carries these
        per row), use it as-is; (3) else if ``image_filename_column`` is present,
        derive via :class:`ConfigurableFileHandler` (e.g. the regionprops layout, one
        CSV covering many source images); (4) else raise ``ValueError`` — z_index in
        particular is required, since it is part of the join key.
    image_filename_column : str
        Column holding the source image filename, used for derivation when present
        and the corresponding column arg is ``None``.
    file_handler : ConfigurableFileHandler, optional
        Handler used for filename parsing. Defaults to a plain
        ``ConfigurableFileHandler()``.

    Returns
    -------
    pd.DataFrame
        Columns: ``CELL_KEY + <numeric feature columns>``.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    if id_column not in df.columns:
        raise ValueError(
            f"id_column {id_column!r} not found in feature CSV {csv_path}; "
            f"available columns: {list(df.columns)}"
        )
    df = df.rename(columns={id_column: "cell_id"})

    for column_arg, canonical_name in (
        (sample_id_column, "sample_id"),
        (timepoint_column, "timepoint"),
        (z_index_column, "z_index"),
    ):
        if column_arg is None:
            continue
        if column_arg not in df.columns:
            raise ValueError(
                f"{canonical_name}_column {column_arg!r} not found in feature CSV "
                f"{csv_path}; available columns: {list(df.columns)}"
            )
        df = df.rename(columns={column_arg: canonical_name})

    # Resolution order per field: (1) explicit *_column arg, already applied above;
    # (2) a column already named sample_id/timepoint/z_index (e.g. incarta's
    # split_data layout, which already carries these per row); (3) derive from an
    # image_filename column via ConfigurableFileHandler (e.g. regionprops, one CSV
    # covering many source images); (4) unresolved -> raise.
    needs_sample_id = "sample_id" not in df.columns
    needs_timepoint = "timepoint" not in df.columns
    needs_z_index = "z_index" not in df.columns

    if needs_sample_id or needs_timepoint or needs_z_index:
        if image_filename_column not in df.columns:
            missing = [
                name
                for name, needed in (
                    ("sample_id", needs_sample_id),
                    ("timepoint", needs_timepoint),
                    ("z_index", needs_z_index),
                )
                if needed
            ]
            raise ValueError(
                f"Cannot resolve {missing} for {csv_path}: no explicit *_column "
                "argument was given, no already-named column exists, and "
                f"{image_filename_column!r} is not present to derive from. z_index "
                "in particular is required — it is part of the join key."
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
        if needs_z_index:
            unresolved = [
                name
                for name in filenames.unique()
                if handler.extract_z_index(name) is None
            ]
            if unresolved:
                raise ValueError(
                    f"Cannot derive z_index for {csv_path}: filename(s) "
                    f"{unresolved} do not match the '_z<N>' convention, and "
                    "z_index_column was not provided. z_index is required — it is "
                    "part of the join key."
                )
            df["z_index"] = filenames.map(
                lambda name: str(handler.extract_z_index(name))
            )

    id_and_key_columns = {"cell_id", "sample_id", "timepoint", "z_index"}
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


def load_features_from_directory(
    directory: Path,
    id_column: str,
    sample_id_column: Optional[str] = None,
    timepoint_column: Optional[str] = None,
    z_index_column: Optional[str] = None,
    image_filename_column: str = "image_filename",
    pattern: str = "*.csv",
    file_handler: Optional[ConfigurableFileHandler] = None,
) -> pd.DataFrame:
    """Load and concatenate a directory of per-(well, timepoint, z) feature CSVs.

    Some feature-extraction backends write one CSV per (well, timepoint, z) slice
    (e.g. the incarta ``split_data`` layout —
    ``pMF5V1_<well>_t<tp>_z<z>_BF_features.csv``), so a single :func:`load_features`
    call only covers one slice. This globs
    ``directory`` for matching files, loads each with :func:`load_features` (same
    column-resolution rules — an already-present ``sample_id``/``timepoint``/
    ``z_index`` column, an explicit ``*_column`` override, or filename derivation via
    ``image_filename_column``), and concatenates the results into one combined
    feature table.

    Parameters
    ----------
    directory : Path
        Directory containing the per-slice feature CSVs.
    id_column, sample_id_column, timepoint_column, z_index_column, image_filename_column
        Forwarded to :func:`load_features` for every matched file — see its
        docstring for the resolution order.
    pattern : str
        Glob pattern selecting the feature CSVs within ``directory``.
    file_handler : ConfigurableFileHandler, optional
        Handler used for filename parsing; passed through to :func:`load_features`.

    Returns
    -------
    pd.DataFrame
        Columns: ``CELL_KEY + <numeric feature columns>``, concatenated across all
        matched files.

    Raises
    ------
    ValueError
        If no files match ``pattern`` in ``directory``.
    """
    directory = Path(directory)
    csv_paths = sorted(directory.glob(pattern))
    if not csv_paths:
        raise ValueError(
            f"No files matching pattern {pattern!r} found in directory {directory}"
        )

    frames = [
        load_features(
            csv_path,
            id_column=id_column,
            sample_id_column=sample_id_column,
            timepoint_column=timepoint_column,
            z_index_column=z_index_column,
            image_filename_column=image_filename_column,
            file_handler=file_handler,
        )
        for csv_path in csv_paths
    ]
    combined = pd.concat(frames, ignore_index=True)

    logger.info(
        "Loaded %d cells from %d feature files in %s (pattern %r)",
        len(combined),
        len(csv_paths),
        directory,
        pattern,
    )

    return combined
