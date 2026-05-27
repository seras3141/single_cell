"""Loaders and metadata helpers for mCherry metrics extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.utils.file_utils import ConfigurableFileHandler


def ensure_2d(array: np.ndarray) -> np.ndarray:
    """Return a 2D view of an input array.

    Parameters
    ----------
    array : np.ndarray
        Input image or label array.

    Returns
    -------
    np.ndarray
        The input array if already 2D, otherwise the first plane.
    """
    if array.ndim == 2:
        return array
    return array[0]


def find_label_from_mcherry_path(
    img_path: Path,
    mcherry_suffix: str = "_mCherry",
    mask_suffix: str = "_Cells",
    label_dir: Path | None = None,
) -> Path | None:
    """Resolve the expected label path for an mCherry image."""
    target_name = img_path.name.replace(mcherry_suffix, mask_suffix)
    candidate = (label_dir or img_path.parent) / target_name
    if candidate.exists():
        return candidate
    return None


def discover_mcherry_images(mcherry_dir: Path, pattern: str) -> list[Path]:
    """Discover mCherry images for a batch run."""
    return sorted(mcherry_dir.glob(pattern))


def resolve_label_paths(
    image_paths: list[Path],
    label_dir: Path | None,
    label_suffix: str,
) -> list[Path | None]:
    """Resolve label paths for a batch of images."""
    return [
        find_label_from_mcherry_path(
            img_path=image_path,
            mask_suffix=label_suffix,
            label_dir=label_dir,
        )
        for image_path in image_paths
    ]


def build_file_handler(
    wavelength_mappings: dict[int, str] | None = None,
    plate_number: str | None = None,
) -> ConfigurableFileHandler:
    """Build a file handler for metadata extraction."""
    return ConfigurableFileHandler(
        wavelength_mappings=wavelength_mappings,
        plate_number=plate_number,
    )


def extract_image_metadata(
    image_path: Path,
    file_handler: ConfigurableFileHandler,
) -> dict[str, Any]:
    """Extract sample metadata from an image path."""
    image_name = image_path.name
    sample_id = file_handler.extract_sample_id(image_name) or ""
    z_index = file_handler.extract_z_index(image_name)
    timepoint = file_handler.extract_time_point(image_name)

    try:
        unique_id = file_handler.extract_unique_id(image_name)
    except ValueError:
        unique_id = sample_id

    return {
        "sample_id": sample_id,
        "z_index": -1 if z_index is None else int(z_index),
        "timepoint": "" if timepoint == "unknown" else str(timepoint),
        "unique_id": unique_id,
    }


def should_keep_image(
    image_path: Path,
    file_handler: ConfigurableFileHandler,
    exclude_z0: bool,
) -> bool:
    """Determine whether an image should be processed."""
    if not exclude_z0:
        return True

    z_index = file_handler.extract_z_index(image_path.name)
    if z_index is None:
        return True
    return int(z_index) != 0