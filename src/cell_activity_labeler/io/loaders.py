"""CSV loading and normalization helpers for activity labeling."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

IMAGE_COLUMNS = ("image", "image_path")
CELL_ID_COLUMN = "cell_id"


def validate_metrics_input_dataframe(metrics_df: pd.DataFrame) -> None:
    """Validate that a metrics table has the minimum identity columns."""
    missing = []
    if not any(column in metrics_df.columns for column in IMAGE_COLUMNS):
        missing.append("image or image_path")
    if CELL_ID_COLUMN not in metrics_df.columns:
        raise ValueError(
            "Metrics dataframe is missing required column 'cell_id'. Legacy CSVs "
            "using 'label_id'/'label' are no longer supported; re-run "
            "mcherry_metrics extraction to emit 'cell_id' directly."
        )
    if missing:
        raise ValueError(
            "Metrics dataframe is missing required columns: " + ", ".join(missing)
        )


def normalize_metrics_dataframe(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Return a metrics table compatible with legacy and milestone-1 CSVs."""
    validate_metrics_input_dataframe(metrics_df)

    normalized = metrics_df.copy()

    if "image" not in normalized.columns and "image_path" in normalized.columns:
        normalized["image"] = normalized["image_path"].map(
            lambda value: Path(str(value)).name if pd.notna(value) else ""
        )
    if "image_path" not in normalized.columns and "image" in normalized.columns:
        normalized["image_path"] = normalized["image"]

    if "sample" not in normalized.columns and "sample_id" in normalized.columns:
        normalized["sample"] = normalized["sample_id"]
    if "sample_id" not in normalized.columns and "sample" in normalized.columns:
        normalized["sample_id"] = normalized["sample"]

    if "time" not in normalized.columns and "timepoint" in normalized.columns:
        normalized["time"] = normalized["timepoint"]
    if "timepoint" not in normalized.columns and "time" in normalized.columns:
        normalized["timepoint"] = normalized["time"]

    if "ID" not in normalized.columns:
        source_series = None
        if "image_path" in normalized.columns:
            source_series = normalized["image_path"]
        elif "image" in normalized.columns:
            source_series = normalized["image"]
        if source_series is not None:
            normalized["ID"] = source_series.map(
                lambda value: (
                    Path(str(value)).stem.replace("_mCherry", "")
                    if pd.notna(value)
                    else ""
                )
            )

    if "label_path" not in normalized.columns:
        normalized["label_path"] = pd.NA

    return normalized


def load_metrics_csv(source: Path) -> pd.DataFrame:
    """Load a metrics CSV and normalize it for the labeling pipeline."""
    return normalize_metrics_dataframe(pd.read_csv(source))


__all__ = [
    "load_metrics_csv",
    "normalize_metrics_dataframe",
    "validate_metrics_input_dataframe",
]
