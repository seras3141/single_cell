"""MF5v1 plate layout loading and per-well annotation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import pandas as pd

ROWS = tuple("ABCDEFGHIJKLMNOP")


def load_plate_layout(layout_path: Union[str, Path]) -> Dict[str, Any]:
    """Load a machine-readable plate layout JSON file."""
    path = Path(layout_path)
    if not path.exists():
        raise FileNotFoundError(f"Plate layout file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        layout = json.load(handle)

    required_keys = {"dimensions", "row_assignments", "drugs", "controls"}
    missing = required_keys - set(layout)
    if missing:
        raise ValueError(f"Plate layout is missing required keys: {sorted(missing)}")

    return layout


def _empty_annotation(row: str, col: int, quadrant: int) -> Dict[str, Any]:
    return {
        "well_id": f"{row}{col:02d}",
        "row": row,
        "col": col,
        "content": "empty",
        "drug": None,
        "alias": None,
        "concentration_uM": None,
        "replicate": None,
        "quadrant": quadrant,
        "control": None,
        "control_type": None,
        "class": None,
        "target": None,
        "cmax_uM": None,
    }


def _drug_annotation(
    row: str,
    col: int,
    quadrant: int,
    row_info: Mapping[str, Any],
    layout: Mapping[str, Any],
) -> Dict[str, Any]:
    columns_per_quadrant = int(
        layout.get("quadrants", {}).get("columns_per_quadrant", 6)
    )
    col_offset = (col - 1) % columns_per_quadrant + 1
    conc_key = layout["column_pattern_within_quadrant"].get(f"offset_{col_offset}")

    if conc_key == "empty":
        return _empty_annotation(row, col, quadrant)

    drug_name = row_info["drug"]
    drug_info = layout["drugs"][drug_name]
    concentrations = drug_info.get("concentrations_uM", {})

    return {
        "well_id": f"{row}{col:02d}",
        "row": row,
        "col": col,
        "content": "drug",
        "drug": drug_name,
        "alias": drug_info.get("alias"),
        "concentration_uM": concentrations.get(conc_key),
        "replicate": row_info.get("replicate"),
        "quadrant": quadrant,
        "control": None,
        "control_type": None,
        "class": drug_info.get("class"),
        "target": drug_info.get("target"),
        "cmax_uM": drug_info.get("cmax_uM"),
    }


def _control_annotation(
    row: str,
    col: int,
    quadrant: int,
    row_info: Mapping[str, Any],
    layout: Mapping[str, Any],
) -> Dict[str, Any]:
    columns_per_quadrant = int(
        layout.get("quadrants", {}).get("columns_per_quadrant", 6)
    )
    col_offset = (col - 1) % columns_per_quadrant + 1

    if row_info.get("drug"):
        control_name = row_info["drug"]
        if col_offset == 1:
            return _empty_annotation(row, col, quadrant)
    else:
        control_name = layout["control_column_pattern_within_quadrant"].get(
            f"offset_{col_offset}", "empty"
        )
        if control_name == "empty":
            return _empty_annotation(row, col, quadrant)

    control_info = layout.get("controls", {}).get(control_name, {})
    return {
        "well_id": f"{row}{col:02d}",
        "row": row,
        "col": col,
        "content": "control",
        "drug": None,
        "alias": None,
        "concentration_uM": None,
        "replicate": row_info.get("replicate"),
        "quadrant": quadrant,
        "control": control_name,
        "control_type": control_info.get("type", row_info.get("control_type")),
        "class": None,
        "target": None,
        "cmax_uM": None,
    }


def get_well_annotation(
    row: str, col: int, layout: Mapping[str, Any]
) -> Dict[str, Any]:
    """Return the MF5v1 annotation for one well."""
    normalized_row = row.upper()
    normalized_col = int(col)

    if normalized_row not in layout["row_assignments"]:
        raise ValueError(f"Unknown plate row: {row}")

    max_col = int(layout.get("dimensions", {}).get("columns", 24))
    if normalized_col < 1 or normalized_col > max_col:
        raise ValueError(f"Column must be between 1 and {max_col}: {col}")

    columns_per_quadrant = int(
        layout.get("quadrants", {}).get("columns_per_quadrant", 6)
    )
    quadrant = (normalized_col - 1) // columns_per_quadrant + 1
    row_info = layout["row_assignments"][normalized_row]
    content = row_info.get("content")

    if content == "empty":
        return _empty_annotation(normalized_row, normalized_col, quadrant)
    if content == "drug":
        return _drug_annotation(
            normalized_row, normalized_col, quadrant, row_info, layout
        )
    if content == "control":
        return _control_annotation(
            normalized_row, normalized_col, quadrant, row_info, layout
        )

    annotation = _empty_annotation(normalized_row, normalized_col, quadrant)
    annotation["content"] = content or "unknown"
    return annotation


def build_plate_annotation_dataframe(
    layout_path: Union[str, Path],
    rows: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Build a 384-well MF5v1 annotation table from the layout JSON."""
    layout = load_plate_layout(layout_path)
    row_labels = rows or list(layout.get("dimensions", {}).get("row_labels", ROWS))
    column_count = int(layout.get("dimensions", {}).get("columns", 24))

    records = []
    for row in row_labels:
        for col in range(1, column_count + 1):
            records.append(get_well_annotation(row, col, layout))

    return pd.DataFrame.from_records(records)
