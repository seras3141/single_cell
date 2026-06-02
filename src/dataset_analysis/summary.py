"""Summary tables and JSON export for dataset-analysis notebooks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Sequence, Union

import numpy as np
import pandas as pd


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (set, tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if pd.isna(value) if not isinstance(value, (list, tuple, set, dict)) else False:
        return None
    return value


def _observed_groups(inventory: pd.DataFrame) -> pd.DataFrame:
    if inventory.empty:
        return pd.DataFrame(columns=["plate_id", "time_point", "well_id"])
    return inventory.drop_duplicates(["plate_id", "time_point", "well_id"])


def build_dataset_summary(
    inventory: pd.DataFrame,
    issues: pd.DataFrame,
    expected_channels: Sequence[str],
    expected_z_indices: Sequence[int],
    projection_z_index: int = 0,
) -> Dict[str, Any]:
    """Build a machine-readable summary for downstream notebooks."""
    observed = _observed_groups(inventory)
    expected_files = (
        len(observed) * len(expected_channels) * (len(expected_z_indices) + 1)
    )
    issue_counts_by_severity = (
        issues["severity"].value_counts().to_dict() if not issues.empty else {}
    )
    issue_counts_by_type = (
        issues["issue_type"].value_counts().to_dict() if not issues.empty else {}
    )

    if inventory.empty:
        z_indices = []
        channels_present = []
        plate_ids = []
        time_points = []
    else:
        z_indices = sorted(inventory["z_index"].dropna().astype(int).unique().tolist())
        channels_present = sorted(
            inventory["channel"].dropna().astype(str).unique().tolist()
        )
        plate_ids = sorted(inventory["plate_id"].dropna().astype(str).unique().tolist())
        time_points = sorted(
            inventory["time_point"].dropna().astype(int).unique().tolist()
        )

    control_counts = {}
    drug_counts = {}
    if not observed.empty:
        if "control" in observed:
            control_counts = (
                observed.dropna(subset=["control"])["control"].value_counts().to_dict()
            )
        if "drug" in observed:
            drug_counts = (
                observed.dropna(subset=["drug"])["drug"].value_counts().to_dict()
            )

    summary = {
        "total_image_files": len(inventory),
        "unique_plates": plate_ids,
        "unique_time_points": time_points,
        "observed_plate_time_wells": len(observed),
        "channels_present": channels_present,
        "expected_channels": list(expected_channels),
        "z_indices_present": z_indices,
        "expected_core_z_indices": list(expected_z_indices),
        "expected_projection_z_index": projection_z_index,
        "total_dataset_size_gb": (
            float(inventory["file_size_mb"].sum() / 1024)
            if not inventory.empty
            else 0.0
        ),
        "expected_files_for_observed_wells": expected_files,
        "actual_files": len(inventory),
        "issue_counts_by_severity": issue_counts_by_severity,
        "issue_counts_by_type": issue_counts_by_type,
        "control_counts_observed_wells": control_counts,
        "drug_counts_observed_wells": drug_counts,
    }
    return _json_safe(summary)


def build_summary_table(
    inventory: pd.DataFrame,
    issues: pd.DataFrame,
    expected_channels: Sequence[str],
    expected_z_indices: Sequence[int],
    projection_z_index: int = 0,
) -> pd.DataFrame:
    """Build a compact human-readable summary table."""
    summary = build_dataset_summary(
        inventory,
        issues,
        expected_channels,
        expected_z_indices,
        projection_z_index=projection_z_index,
    )

    if inventory.empty:
        z_summary = "none"
        wells_by_plate = "none"
    else:
        per_group_z = inventory.groupby(["plate_id", "time_point", "well_id"])[
            "z_index"
        ].nunique()
        z_summary = (
            f"min {per_group_z.min()}, median {per_group_z.median():.1f}, "
            f"max {per_group_z.max()}"
        )
        wells_by_plate = (
            inventory.drop_duplicates(["plate_id", "time_point", "well_id"])
            .groupby(["plate_id", "time_point"])["well_id"]
            .nunique()
            .rename("wells")
            .reset_index()
            .assign(
                label=lambda df: df["plate_id"] + " t" + df["time_point"].astype(str)
            )
        )
        wells_by_plate = ", ".join(
            f"{row.label}: {int(row.wells)}" for row in wells_by_plate.itertuples()
        )

    rows = [
        ("Total image files", summary["total_image_files"]),
        ("Unique plates", ", ".join(summary["unique_plates"]) or "none"),
        (
            "Unique time points",
            ", ".join(map(str, summary["unique_time_points"])) or "none",
        ),
        ("Observed wells per plate/time", wells_by_plate),
        ("Channels present", ", ".join(summary["channels_present"]) or "none"),
        ("Z-slices per observed well", z_summary),
        ("Total dataset size (GB)", f"{summary['total_dataset_size_gb']:.3f}"),
        (
            "Expected vs actual files",
            f"{summary['expected_files_for_observed_wells']} expected / "
            f"{summary['actual_files']} actual",
        ),
        ("Issue counts by severity", summary["issue_counts_by_severity"] or "none"),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def write_summary_json(summary: Dict[str, Any], output_path: Union[str, Path]) -> Path:
    """Write a dataset summary JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(summary), handle, indent=2, sort_keys=True)
    return path
