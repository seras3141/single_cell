"""Activity image generation from labeled instance tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff


def create_activity_labeled_image(
    label_data: np.ndarray,
    classification_data: pd.DataFrame,
    label_dict: dict[str, int] | None = None,
) -> np.ndarray:
    """Render an activity label image for a single source image."""
    required_cols = {"cell_id", "is_active", "image"}
    if not required_cols.issubset(classification_data.columns):
        raise ValueError(f"Classification data must contain columns: {required_cols}")

    if classification_data["image"].nunique() > 1:
        raise ValueError(
            "Classification data contains multiple images. Filter to a single image first."
        )

    activity_labels = np.zeros_like(label_data, dtype=np.int32)
    activity_dict = dict(
        zip(classification_data["cell_id"], classification_data["is_active"])
    )

    for cell_id in np.unique(label_data):
        if cell_id == 0:
            continue
        mask = label_data == cell_id
        is_active = bool(activity_dict.get(cell_id, False))
        if label_dict is not None:
            activity_labels[mask] = (
                label_dict["active"] if is_active else label_dict["dead"]
            )
        else:
            activity_labels[mask] = int(cell_id) if is_active else -int(cell_id)

    return activity_labels


def save_activity_images(
    labeled_df: pd.DataFrame,
    output_dir: Path,
    label_dict: dict[str, int] | None = None,
) -> list[Path]:
    """Write activity TIFFs for all images that have a label path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for _, image_df in labeled_df.groupby("image", dropna=False):
        label_path_value = image_df["label_path"].iloc[0]
        if pd.isna(label_path_value):
            continue

        label_path = Path(str(label_path_value))
        if not label_path.exists():
            continue

        label_data = tiff.imread(str(label_path)).astype(np.int32, copy=False)
        image_path_value = image_df["image_path"].iloc[0]
        image_path = Path(str(image_path_value)) if pd.notna(image_path_value) else None
        if image_path is not None:
            output_path = _derive_activity_output_path(
                image_path=image_path,
                output_dir=output_dir,
                binary_output=label_dict is not None,
            )
        else:
            output_path = output_dir / f"{image_df['image'].iloc[0]}_activity.tif"

        activity_labels = create_activity_labeled_image(
            label_data,
            image_df,
            label_dict=label_dict,
        )
        tiff.imwrite(str(output_path), activity_labels.astype(np.int32))
        saved_paths.append(output_path)

    return saved_paths


def _derive_activity_output_path(
    image_path: Path, output_dir: Path, binary_output: bool
) -> Path:
    suffix = "_activity_bin" if binary_output else "_activity"
    filename = image_path.name.replace("_mCherry", suffix)
    return output_dir / filename


__all__ = ["create_activity_labeled_image", "save_activity_images"]