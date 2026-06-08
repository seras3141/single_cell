"""Image preprocessing helpers for metric extraction."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage import filters, morphology

from ..config import ExtractionConfig


class ImagePreprocessor:
    """Apply optional preprocessing before metric extraction.

    Parameters
    ----------
    config : ExtractionConfig
        Extraction configuration containing preprocessing options.
    """

    def __init__(self, config: ExtractionConfig):
        self.config = config

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess an image according to the extraction configuration.

        Parameters
        ----------
        image : np.ndarray
            Input intensity image.

        Returns
        -------
        np.ndarray
            Preprocessed image.
        """
        result = image.astype(np.float64, copy=False)

        if self.config.normalize_before_extraction:
            result = self._normalize(result)

        if self.config.gaussian_sigma > 0:
            result = ndi.gaussian_filter(result, sigma=self.config.gaussian_sigma)

        if self.config.median_footprint > 0:
            result = filters.median(
                result,
                footprint=morphology.disk(self.config.median_footprint),
            )

        if self.config.background_subtract_radius > 0:
            background = morphology.opening(
                result,
                morphology.disk(self.config.background_subtract_radius),
            )
            result = result - background

        return result

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        if self.config.normalize_mode == "minmax":
            min_value = float(np.nanmin(image))
            max_value = float(np.nanmax(image))
            if max_value > min_value:
                return (image - min_value) / (max_value - min_value)
            return image

        p1, p99 = np.nanpercentile(image, (1, 99))
        if p99 > p1:
            return (image - p1) / (p99 - p1)
        return image