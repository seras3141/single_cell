"""Per-cell PyRadiomics features for 2D label masks (the ``pyradiomics`` backend).

Adapted from the external GPU radiomics delivery. PyRadiomics and SimpleITK are
imported lazily through :func:`_resolve_backend`, so this module imports cleanly in
environments without them (the primary ``.venv``); extraction itself runs in the
separate ``.venv-pyradiomics`` env, where ``pyradiomics-cuda`` provides the
``radiomics`` import path. :func:`_resolve_backend` is also the one seam tests replace.
"""

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

#: Per-process cache of configured extractors, keyed on the extractor settings.
_EXTRACTOR_CACHE: Dict[Tuple[Any, ...], Any] = {}

_DIAGNOSTICS_PREFIX = "diagnostics_"
_FEATURE_PREFIX = "original_"


@functools.lru_cache(maxsize=2)
def _resolve_backend(require_cuda: bool) -> Tuple[Any, Any]:
    """Import and return ``(radiomics.featureextractor, SimpleITK)``, once per process.

    Raises:
        ImportError: PyRadiomics or SimpleITK is missing, or ``require_cuda`` is set
            and the ``pyradiomics-cuda`` distribution is not installed. The pipeline
            re-raises ``ImportError`` on the first file instead of recording it per
            file.
    """
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
    # PyRadiomics logs every ROI at INFO; keep only errors, on its own logger.
    logging.getLogger("radiomics").setLevel(logging.ERROR)
    return featureextractor, sitk


def _extractor_key(cfg: PyradiomicsConfig) -> Tuple[Any, ...]:
    """Cache key: only the settings that change the configured extractor."""
    return (
        float(cfg.bin_width),
        bool(cfg.force_2d),
        bool(cfg.normalize),
        float(cfg.normalize_scale),
        tuple(cfg.feature_classes),
    )


def build_extractor(cfg: PyradiomicsConfig, featureextractor: Any) -> Any:
    """Create a ``RadiomicsFeatureExtractor`` with only ``cfg.feature_classes`` on."""
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
    """Labels present in the outer 1-pixel ring of ``mask``."""
    ring = np.concatenate([mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]])
    return {int(label) for label in np.unique(ring) if label != 0}


def _to_feature(value: Any) -> float:
    """A feature value as float; NaN if PyRadiomics returned anything non-scalar.

    Keeps every ``original_*`` column float, so one odd value cannot turn a
    column into mixed types (which Parquet rejects at write time).
    """
    try:
        return float(np.asarray(value, dtype=np.float64).item())
    except (TypeError, ValueError):
        return float("nan")


def _to_value(value: Any) -> Any:
    """PyRadiomics returns 0-d numpy arrays; unwrap them, stringify non-scalars."""
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (bool, int, float, str)):
        return value
    return str(value)  # tuples/dicts from diagnostics_*, so Parquet can store them


def get_radiomics_features(
    mask: np.ndarray, image: np.ndarray, cfg: PyradiomicsConfig
) -> pd.DataFrame:
    """Extract PyRadiomics features for every label of a 2D mask.

    Args:
        mask: 2D integer label mask (0 = background).
        image: 2D intensity image, same shape as ``mask``.
        cfg: Backend settings.

    Returns:
        One row per extracted label: ``cell_id``, ``touches_border``, then the
        ``original_*`` features (plus ``diagnostics_*`` when
        ``cfg.include_diagnostics``). Labels with fewer than ``cfg.min_pixels``
        pixels are skipped, and labels PyRadiomics rejects (its own ROI checks
        raise ``ValueError``, e.g. a 1-pixel-wide fragment) are skipped too, so
        one bad label does not discard the image. Both counts are in
        ``df.attrs`` (``n_skipped_small``, ``n_label_errors``).
    """
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
    # One conversion per image. uint32 keeps label ids above 65,535 intact.
    image_itk = sitk.GetImageFromArray(np.asarray(image, dtype=np.float32))
    mask_itk = sitk.GetImageFromArray(np.asarray(mask, dtype=np.uint32))
    border = _border_labels(mask)

    rows = []
    n_label_errors = 0
    for label in keep:
        try:
            result = extractor.execute(image_itk, mask_itk, label=label)
        except ValueError as exc:
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
