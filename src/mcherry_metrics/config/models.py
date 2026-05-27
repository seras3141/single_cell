"""Typed configuration for mCherry metrics extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


NormalizeMode = Literal["minmax", "percentile"]


@dataclass(frozen=True)
class ExtractionConfig:
    """Configuration for per-instance intensity extraction.

    Parameters
    ----------
    percentiles : list[int]
        Additional percentile values to compute. The stable CSV contract always
        includes the 75th, 90th, and 95th percentiles.
    normalize_before_extraction : bool
        Whether to normalize the image before computing instance statistics.
    normalize_mode : {"minmax", "percentile"}
        Normalization mode used when normalization is enabled.
    gaussian_sigma : float
        Gaussian blur sigma. Zero disables the blur step.
    median_footprint : int
        Radius of the median filter footprint. Zero disables it.
    background_subtract_radius : int
        Radius for morphological opening background subtraction. Zero disables
        it.
    n_jobs : int
        Number of parallel workers used for batch extraction.
    exclude_z0 : bool
        Whether to exclude z0 images based on filename metadata.
    min_area_px : int
        Minimum region area to keep in the output table.
    write_analytics : bool
        Whether a batch run should emit analytics outputs.
    """

    percentiles: list[int] = field(default_factory=lambda: [75, 90, 95])
    normalize_before_extraction: bool = False
    normalize_mode: NormalizeMode = "minmax"
    gaussian_sigma: float = 0.0
    median_footprint: int = 0
    background_subtract_radius: int = 0
    n_jobs: int = 1
    exclude_z0: bool = True
    min_area_px: int = 10
    write_analytics: bool = True

    def __post_init__(self) -> None:
        """Validate extraction parameters."""
        if not self.percentiles:
            raise ValueError("percentiles must contain at least one value")

        if any(percentile < 0 or percentile > 100 for percentile in self.percentiles):
            raise ValueError("percentiles must be between 0 and 100")

        if self.gaussian_sigma < 0:
            raise ValueError("gaussian_sigma must be non-negative")

        if self.median_footprint < 0:
            raise ValueError("median_footprint must be non-negative")

        if self.background_subtract_radius < 0:
            raise ValueError("background_subtract_radius must be non-negative")

        if self.min_area_px < 1:
            raise ValueError("min_area_px must be at least 1")

        if self.n_jobs == 0:
            raise ValueError("n_jobs must be non-zero")

    @property
    def effective_percentiles(self) -> list[int]:
        """Return the contract percentiles plus any caller-specified extras."""
        return sorted({75, 90, 95, *self.percentiles})