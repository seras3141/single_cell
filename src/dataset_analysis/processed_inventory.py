"""Processed output inventory and gap detection for pipeline stages."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.utils.file_utils import EXPERIMENT_WAVELENGTH_MAPPINGS

STAGE_ORDER = [
    "prepare-split",
    "prepare-3d",
    "prepare-blur",
    "segment-2d",
    "segment-3d",
    "mcherry",
]

_STAGE_DIRS = {
    "prepare-split": "split_data",
    "prepare-3d": "3d_data",
    "prepare-blur": "blur_heatmaps",
    "segment-2d": "inference/cellpose_sam/test/masks",
    "segment-3d": "inference_tracked/cellpose_sam/test/final",
    "mcherry": "mcherry_metrics/cellpose_sam",
}


def _check_stage_dir(processed_dir: Path, stage: str) -> Path:
    stage_dir = processed_dir / _STAGE_DIRS[stage]
    if not stage_dir.exists():
        warnings.warn(
            f"Stage directory for '{stage}' does not exist: {stage_dir}. "
            "All files for this stage will be marked as missing."
        )
    return stage_dir


def _file_size_mb(path: Path) -> Optional[float]:
    if path.exists():
        return path.stat().st_size / 1e6
    return None


def build_processed_inventory(
    raw_inventory: pd.DataFrame,
    processed_dir: Path,
    experiment_name: str,
) -> pd.DataFrame:
    """Build a per-file inventory of expected processed outputs and check existence.

    Args:
        raw_inventory: DataFrame loaded from dataset_inventory.csv (ground truth).
        processed_dir: Root directory of processed outputs for this experiment.
        experiment_name: Key into EXPERIMENT_WAVELENGTH_MAPPINGS for channel resolution.

    Returns:
        DataFrame with columns: stage, well_id, time_point, z_index, channel,
        expected_path, found, file_size_mb.
    """
    channel_map = EXPERIMENT_WAVELENGTH_MAPPINGS.get(experiment_name)
    if channel_map is None:
        raise ValueError(
            f"Unknown experiment name '{experiment_name}'. "
            f"Valid names: {sorted(EXPERIMENT_WAVELENGTH_MAPPINGS)}"
        )

    processed_dir = Path(processed_dir)
    rows: List[Dict[str, Any]] = []

    # --- prepare-split: one file per (well, tp, z, channel) ---
    _check_stage_dir(processed_dir, "prepare-split")
    split_dir = processed_dir / _STAGE_DIRS["prepare-split"]
    for _, row in raw_inventory.iterrows():
        channel = channel_map[int(row.wavelength)]
        expected = split_dir / f"{row.plate_id}_{row.well_id}_t{row.time_point}_z{row.z_index}_{channel}.tif"
        rows.append({
            "stage": "prepare-split",
            "well_id": row.well_id,
            "time_point": row.time_point,
            "z_index": row.z_index,
            "channel": channel,
            "expected_path": str(expected),
            "found": expected.exists(),
            "file_size_mb": _file_size_mb(expected),
        })

    # --- prepare-3d: one BF_3d.tif per (well, tp) ---
    _check_stage_dir(processed_dir, "prepare-3d")
    d3_dir = processed_dir / _STAGE_DIRS["prepare-3d"]
    for _, row in raw_inventory[["plate_id", "well_id", "time_point"]].drop_duplicates().iterrows():
        expected = d3_dir / f"{row.plate_id}_{row.well_id}_t{row.time_point}_BF_3d.tif"
        rows.append({
            "stage": "prepare-3d",
            "well_id": row.well_id,
            "time_point": row.time_point,
            "z_index": None,
            "channel": "BF",
            "expected_path": str(expected),
            "found": expected.exists(),
            "file_size_mb": _file_size_mb(expected),
        })

    # --- prepare-blur: one blur_heatmap.tif per (well, tp) ---
    _check_stage_dir(processed_dir, "prepare-blur")
    blur_dir = processed_dir / _STAGE_DIRS["prepare-blur"]
    for _, row in raw_inventory[["plate_id", "well_id", "time_point"]].drop_duplicates().iterrows():
        expected = blur_dir / f"{row.plate_id}_{row.well_id}_t{row.time_point}_BF_3d_blur_heatmap.tif"
        rows.append({
            "stage": "prepare-blur",
            "well_id": row.well_id,
            "time_point": row.time_point,
            "z_index": None,
            "channel": "BF",
            "expected_path": str(expected),
            "found": expected.exists(),
            "file_size_mb": _file_size_mb(expected),
        })

    # --- segment-2d: one pred_mask.zarr per (well, tp, z) ---
    _check_stage_dir(processed_dir, "segment-2d")
    masks_dir = processed_dir / _STAGE_DIRS["segment-2d"]
    for _, row in raw_inventory[["plate_id", "well_id", "time_point", "z_index"]].drop_duplicates().iterrows():
        expected = masks_dir / f"{row.plate_id}_{row.well_id}_t{row.time_point}_z{row.z_index}_pred_mask.zarr"
        rows.append({
            "stage": "segment-2d",
            "well_id": row.well_id,
            "time_point": row.time_point,
            "z_index": row.z_index,
            "channel": None,
            "expected_path": str(expected),
            "found": expected.exists(),
            "file_size_mb": _file_size_mb(expected),
        })

    # --- segment-3d: one pred_mask_3d.zarr per (well, tp) ---
    _check_stage_dir(processed_dir, "segment-3d")
    tracked_dir = processed_dir / _STAGE_DIRS["segment-3d"]
    for _, row in raw_inventory[["plate_id", "well_id", "time_point"]].drop_duplicates().iterrows():
        expected = tracked_dir / f"{row.plate_id}_{row.well_id}_t{row.time_point}_pred_mask_3d.zarr"
        rows.append({
            "stage": "segment-3d",
            "well_id": row.well_id,
            "time_point": row.time_point,
            "z_index": None,
            "channel": None,
            "expected_path": str(expected),
            "found": expected.exists(),
            "file_size_mb": _file_size_mb(expected),
        })

    # --- mcherry: singleton instance_metrics.csv ---
    _check_stage_dir(processed_dir, "mcherry")
    mcherry_dir = processed_dir / _STAGE_DIRS["mcherry"]
    expected = mcherry_dir / "instance_metrics.csv"
    rows.append({
        "stage": "mcherry",
        "well_id": None,
        "time_point": None,
        "z_index": None,
        "channel": None,
        "expected_path": str(expected),
        "found": expected.exists(),
        "file_size_mb": _file_size_mb(expected),
    })

    return pd.DataFrame(rows)


def build_processed_summary(inventory: pd.DataFrame) -> Dict[str, Any]:
    """Aggregate inventory into a per-stage summary with missing entry details.

    Returns:
        Dict mapping stage name to {expected, found, missing, missing_entries}.
    """
    summary: Dict[str, Any] = {}
    for stage in STAGE_ORDER:
        group = inventory[inventory["stage"] == stage]
        if group.empty:
            continue
        missing = group[~group["found"]]
        missing_entries = []
        for _, row in missing.iterrows():
            entry: Dict[str, Any] = {}
            if row.time_point is not None and not (isinstance(row.time_point, float) and pd.isna(row.time_point)):
                entry["time_point"] = int(row.time_point)
            if row.well_id is not None and not (isinstance(row.well_id, float) and pd.isna(row.well_id)):
                entry["well_id"] = row.well_id
            if row.z_index is not None and not (isinstance(row.z_index, float) and pd.isna(row.z_index)):
                entry["z_index"] = int(row.z_index)
            missing_entries.append(entry)
        summary[stage] = {
            "expected": len(group),
            "found": int(group["found"].sum()),
            "missing": int((~group["found"]).sum()),
            "missing_entries": missing_entries,
        }
    return summary


def print_summary_table(summary: Dict[str, Any]) -> None:
    """Print a human-readable per-stage completeness table."""
    header = f"{'Stage':<16} {'Expected':>10} {'Found':>10} {'Missing':>10}"
    print(header)
    print("-" * len(header))
    for stage in STAGE_ORDER:
        if stage not in summary:
            continue
        s = summary[stage]
        gap_marker = "  <- gap" if s["missing"] > 0 else ""
        print(
            f"{stage:<16} {s['expected']:>10} {s['found']:>10} {s['missing']:>10}{gap_marker}"
        )
