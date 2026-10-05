"""Per-cell PyRadiomics features for 2D label masks."""

from __future__ import annotations

import functools
import importlib
import logging
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage

from src.utils.config_schemas import PyradiomicsConfig

logger = logging.getLogger(__name__)

_EXTRACTOR_CACHE: Dict[Tuple[Any, ...], Any] = {}

_DIAGNOSTICS_PREFIX = "diagnostics_"
_FEATURE_PREFIX = "original_"
_CROP_PAD = 2


@functools.lru_cache(maxsize=1)
def _resolve_backend() -> Tuple[Any, Any, Any]:
    # Lazy: sdist C build may be absent; tests patch this.
    try:
        featureextractor = importlib.import_module("radiomics.featureextractor")
        imageoperations = importlib.import_module("radiomics.imageoperations")
        sitk = importlib.import_module("SimpleITK")
    except ImportError as exc:
        raise ImportError(
            "The pyradiomics backend needs the pyradiomics-cuda package (radiomics) "
            "and SimpleITK; it builds from source, so run uv sync with a C compiler"
        ) from exc
    # PyRadiomics logs every ROI at INFO.
    logging.getLogger("radiomics").setLevel(logging.ERROR)
    return featureextractor, imageoperations, sitk


def _extractor_key(cfg: PyradiomicsConfig) -> Tuple[Any, ...]:
    return (
        float(cfg.bin_width),
        bool(cfg.force_2d),
        tuple(cfg.feature_classes),
        bool(cfg.include_diagnostics),
    )


def build_extractor(cfg: PyradiomicsConfig, featureextractor: Any) -> Any:
    extractor = featureextractor.RadiomicsFeatureExtractor(
        binWidth=cfg.bin_width,
        force2D=cfg.force_2d,
        normalize=False,  # done once per image in get_radiomics_features
        additionalInfo=cfg.include_diagnostics,
    )
    extractor.disableAllFeatures()
    for feature_class in cfg.feature_classes:
        extractor.enableFeatureClassByName(feature_class)
    return extractor


def _get_extractor(cfg: PyradiomicsConfig, featureextractor: Any) -> Any:
    key = _extractor_key(cfg)
    extractor = _EXTRACTOR_CACHE.get(key)
    if extractor is None:
        extractor = build_extractor(cfg, featureextractor)
        _EXTRACTOR_CACHE[key] = extractor
    return extractor


def _border_labels(mask: np.ndarray) -> Set[int]:
    ring = np.concatenate([mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]])
    return {int(label) for label in np.unique(ring) if label != 0}


def _to_feature(value: Any) -> float:
    # Mixed-type columns fail the Parquet write.
    try:
        return float(np.asarray(value, dtype=np.float64).item())
    except (TypeError, ValueError):
        return float("nan")


def _to_value(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _label_counts(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Foreground labels and pixel counts; bincount only while labels are dense."""
    if mask.size == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    low, high = int(mask.min()), int(mask.max())
    if low < 0 or high > np.iinfo(np.uint32).max:
        raise ValueError(f"Labels must fit uint32, got {low}..{high}")
    if high <= mask.size:
        counts = np.bincount(mask.ravel().astype(np.int64, copy=False))
        labels = np.flatnonzero(counts)
        counts = counts[labels]
    else:
        labels, counts = np.unique(mask, return_counts=True)
    foreground = labels != 0
    return labels[foreground], counts[foreground]


def _bounding_boxes(
    mask: np.ndarray, labels: List[int], all_foreground_labels: np.ndarray
) -> Dict[int, Tuple[slice, ...]]:
    """Find boxes directly for dense IDs and renumber sparse IDs first."""
    if max(labels) <= mask.size:
        boxes = ndimage.find_objects(mask, max_label=max(labels))
        return {label: boxes[label - 1] for label in labels}
    dense = np.where(mask != 0, np.searchsorted(all_foreground_labels, mask) + 1, 0)
    boxes = ndimage.find_objects(dense)
    return {
        label: boxes[int(np.searchsorted(all_foreground_labels, label))]
        for label in labels
    }


def get_radiomics_features(
    mask: np.ndarray, image: np.ndarray, cfg: PyradiomicsConfig
) -> pd.DataFrame:
    """One row per label; skip counts go in ``df.attrs``."""
    if not (np.issubdtype(mask.dtype, np.integer) or mask.dtype == bool):
        raise ValueError(f"Mask must be an integer label image, got {mask.dtype}")
    labels, counts = _label_counts(mask)
    empty = pd.DataFrame(columns=["cell_id"])
    empty.attrs.update(n_skipped_small=0, n_label_errors=0)
    if labels.size == 0:
        return empty

    keep = [int(label) for label in labels[counts >= cfg.min_pixels]]
    n_skipped = int(labels.size - len(keep))
    if n_skipped:
        logger.debug("Skipped %d labels below min_pixels=%d", n_skipped, cfg.min_pixels)
    if not keep:
        empty.attrs["n_skipped_small"] = n_skipped
        return empty

    featureextractor, imageoperations, sitk = _resolve_backend()
    extractor = _get_extractor(cfg, featureextractor)
    pixels = np.asarray(image, dtype=np.float32)
    if cfg.normalize:
        # Before cropping: per-crop statistics would differ.
        pixels = sitk.GetArrayFromImage(
            imageoperations.normalizeImage(
                sitk.GetImageFromArray(pixels), normalizeScale=cfg.normalize_scale
            )
        )
    labels_u32 = np.asarray(mask, dtype=np.uint32)  # uint16 wraps labels > 65,535
    boxes = _bounding_boxes(labels_u32, keep, labels)
    border = _border_labels(mask)

    rows = []
    n_label_errors = 0
    for label in keep:
        crop = tuple(
            slice(max(axis.start - _CROP_PAD, 0), min(axis.stop + _CROP_PAD, size))
            for axis, size in zip(boxes[label], mask.shape)
        )
        try:
            result = extractor.execute(
                sitk.GetImageFromArray(pixels[crop]),
                sitk.GetImageFromArray(labels_u32[crop]),
                label=label,
            )
        except ValueError as exc:  # PyRadiomics ROI checks reject thin fragments
            n_label_errors += 1
            logger.warning("PyRadiomics rejected label %d: %s", label, exc)
            continue
        row: Dict[str, Any] = {"cell_id": label, "touches_border": label in border}
        for name, value in result.items():
            if name.startswith(_FEATURE_PREFIX):
                row[name] = _to_feature(value)
            elif cfg.include_diagnostics and name.startswith(_DIAGNOSTICS_PREFIX):
                row[name] = _to_value(value)
        rows.append(row)

    df = pd.DataFrame(rows) if rows else empty.copy()
    df.attrs.update(n_skipped_small=n_skipped, n_label_errors=n_label_errors)
    return df
