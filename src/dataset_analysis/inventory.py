"""Raw TIFF discovery and inventory construction for dataset summary notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd

from src.dataset_analysis.layout import build_plate_annotation_dataframe
from src.utils.file_utils import (
    ConfigurableFileHandler,
    DEFAULT_WAVELENGTH_MAPPINGS,
    EXPERIMENT_WAVELENGTH_MAPPINGS,
)

TIFF_SUFFIXES = {".tif", ".tiff"}

BASE_INVENTORY_COLUMNS = [
    "plate_id",
    "time_point",
    "well_id",
    "row",
    "col",
    "z_index",
    "wavelength",
    "channel",
    "file_path",
    "file_size_mb",
]


def normalize_plate_id(plate_number: Optional[str]) -> str:
    """Normalize a plate identifier to the downstream `p<plate>` convention."""
    if plate_number is None or plate_number == "" or plate_number == "unknown":
        return "unknown"
    plate = str(plate_number)
    if plate.startswith("p"):
        return plate
    return f"p{plate}"


def load_expected_channels(
    wavelength_mappings: Optional[Dict[int, str]] = None,
) -> List[str]:
    """Return expected channel names in wavelength-index order.

    Args:
        wavelength_mappings: Mapping to use. Defaults to DEFAULT_WAVELENGTH_MAPPINGS (Ew2 convention).
            Use EXPERIMENT_WAVELENGTH_MAPPINGS[experiment_name] for other experiments.
    """
    mappings = wavelength_mappings if wavelength_mappings is not None else DEFAULT_WAVELENGTH_MAPPINGS
    return [mappings[index] for index in sorted(mappings)]


def discover_image_files(
    data_root: Union[str, Path],
    file_handler: Optional[ConfigurableFileHandler] = None,
) -> List[Path]:
    """Find raw TIFF files that match the configured image filename pattern."""
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"Raw TIFF directory not found: {root}")

    handler = file_handler or ConfigurableFileHandler()
    candidates = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in TIFF_SUFFIXES
    )
    return [path for path in candidates if handler.extract_values(str(path), "image")]


def parse_image_metadata(
    file_path: Union[str, Path],
    file_handler: ConfigurableFileHandler,
    plate_number: Optional[str] = None,
) -> Dict[str, Any]:
    """Parse one raw image filename into normalized inventory metadata."""
    path = Path(file_path)
    values = file_handler.extract_values(str(path), "image")
    if not values:
        raise ValueError(f"Could not parse image filename: {path}")

    extracted_plate = plate_number or file_handler.extract_plate_number(str(path))
    wavelength = int(values["wavelength"])
    channel = file_handler.get_channel_name(wavelength)
    row = values["row"].upper()
    col = int(values["col"])

    return {
        "plate_id": normalize_plate_id(extracted_plate),
        "time_point": int(values["time"]),
        "well_id": f"{row}{col:02d}",
        "row": row,
        "col": col,
        "z_index": int(values["z"]),
        "wavelength": wavelength,
        "channel": channel,
        "file_path": str(path),
        "file_size_mb": path.stat().st_size / (1024 * 1024),
    }


def _empty_inventory_dataframe(annotation_columns: Sequence[str]) -> pd.DataFrame:
    columns = list(BASE_INVENTORY_COLUMNS) + [
        col for col in annotation_columns if col != "well_id"
    ]
    return pd.DataFrame(columns=columns)


def build_dataset_inventory(
    data_root: Union[str, Path],
    layout_path: Union[str, Path],
    plate_number: Optional[str] = None,
    wavelength_mappings: Optional[Dict[int, str]] = None,
) -> pd.DataFrame:
    """Build a per-image inventory joined to MF5v1 well annotations.

    Args:
        data_root: Root directory containing raw TIFF files.
        layout_path: Path to the plate layout JSON file.
        plate_number: Default plate number if not extractable from filename.
        wavelength_mappings: Wavelength-to-channel mapping for this experiment.
            Defaults to DEFAULT_WAVELENGTH_MAPPINGS (Ew2 convention).
            Use EXPERIMENT_WAVELENGTH_MAPPINGS[experiment_name] for HD/SA experiments.
    """
    handler = ConfigurableFileHandler(
        plate_number=plate_number,
        wavelength_mappings=wavelength_mappings,
    )
    annotations = build_plate_annotation_dataframe(layout_path)
    annotation_columns = [
        col for col in annotations.columns if col not in {"row", "col"}
    ]
    image_files = discover_image_files(data_root, handler)

    if not image_files:
        return _empty_inventory_dataframe(annotation_columns)

    records = [
        parse_image_metadata(path, handler, plate_number=plate_number)
        for path in image_files
    ]
    inventory = pd.DataFrame.from_records(records)
    inventory = inventory.merge(
        annotations[annotation_columns],
        on="well_id",
        how="left",
    )
    inventory["content"] = inventory["content"].fillna("unmapped")

    return inventory
