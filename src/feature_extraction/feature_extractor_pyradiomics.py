"""Per-cell PyRadiomics features for 2D label masks."""

from __future__ import annotations

import functools
import importlib
import logging
from importlib import metadata
from typing import Any, Dict, Set, Tuple

import numpy as np
import pandas as pd

from src.utils.config_schemas import PyradiomicsConfig

logger = logging.getLogger(__name__)

_EXTRACTOR_CACHE: Dict[Tuple[Any, ...], Any] = {}

_DIAGNOSTICS_PREFIX = "diagnostics_"
_FEATURE_PREFIX = "original_"


@functools.lru_cache(maxsize=2)
def _resolve_backend(require_cuda: bool) -> Tuple[Any, Any]:
    # Lazy: .venv lacks radiomics; tests patch this.
    if require_cuda:
        try:
            metadata.distribution("pyradiomics-cuda")
        except metadata.PackageNotFoundError as exc:
            raise ImportError(
                "feature_extraction.pyradiomics.require_cuda is set but the "
                "pyradiomics-cuda package is not installed in this environment"
            ) from exc
    try:
        featureextractor = importlib.import_module("radiomics.featureextractor")
        sitk = importlib.import_module("SimpleITK")
    except ImportError as exc:
        raise ImportError(
            "The pyradiomics backend needs PyRadiomics (pyradiomics-cuda) and "
            "SimpleITK; run it in the .venv-pyradiomics environment"
        ) from exc
    # PyRadiomics logs every ROI at INFO.
    logging.getLogger("radiomics").setLevel(logging.ERROR)
    return featureextractor, sitk


def _extractor_key(cfg: PyradiomicsConfig) -> Tuple[Any, ...]:
    return (
        float(cfg.bin_width),
        bool(cfg.force_2d),
        bool(cfg.normalize),
        float(cfg.normalize_scale),
        tuple(cfg.feature_classes),
    )


def build_extractor(cfg: PyradiomicsConfig, featureextractor: Any) -> Any:
    extractor = featureextractor.RadiomicsFeatureExtractor(
        binWidth=cfg.bin_width,
        force2D=cfg.force_2d,
        normalize=cfg.normalize,
        normalizeScale=cfg.normalize_scale,
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


def get_radiomics_features(
    mask: np.ndarray, image: np.ndarray, cfg: PyradiomicsConfig
) -> pd.DataFrame:
    """One row per label; skip counts go in ``df.attrs``."""
    labels = np.unique(mask)
    labels = labels[labels != 0]
    empty = pd.DataFrame(columns=["cell_id"])
    empty.attrs.update(n_skipped_small=0, n_label_errors=0)
    if labels.size == 0:
        return empty

    counts = np.bincount(mask.ravel().astype(np.int64, copy=False))
    keep = [int(label) for label in labels if counts[label] >= cfg.min_pixels]
    n_skipped = int(labels.size - len(keep))
    if n_skipped:
        logger.debug("Skipped %d labels below min_pixels=%d", n_skipped, cfg.min_pixels)
    if not keep:
        empty.attrs["n_skipped_small"] = n_skipped
        return empty

    featureextractor, sitk = _resolve_backend(cfg.require_cuda)
    extractor = _get_extractor(cfg, featureextractor)
    # uint16 would wrap labels above 65,535.
    image_itk = sitk.GetImageFromArray(np.asarray(image, dtype=np.float32))
    mask_itk = sitk.GetImageFromArray(np.asarray(mask, dtype=np.uint32))
    border = _border_labels(mask)

    rows = []
    n_label_errors = 0
    for label in keep:
        try:
            result = extractor.execute(image_itk, mask_itk, label=label)
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
