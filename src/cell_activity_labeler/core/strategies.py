"""Threshold selection strategies for instance-level labeling."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..config import Method, ThresholdParams
from .thresholding import ThresholdComputer

SUPPORTED_LABELING_METHODS = (
    "otsu",
    "yen",
    "li",
    "triangle",
    "percentile",
    "manual",
)


class LabelingStrategy(ABC):
    """Compute a scalar threshold for instance-level metric values."""

    method_name: Method

    def __init__(self, params: ThresholdParams | None = None) -> None:
        self.params = params or ThresholdParams()

    @staticmethod
    def _prepare_values(values: np.ndarray) -> np.ndarray:
        prepared = np.asarray(values, dtype=float).ravel()
        return prepared[~np.isnan(prepared)]

    @abstractmethod
    def compute_threshold(self, values: np.ndarray) -> float:
        """Return a scalar threshold for the provided metric values."""


class _ThresholdComputerStrategy(LabelingStrategy):
    """Adapt the existing threshold computer to instance metrics."""

    method_name: Method

    def compute_threshold(self, values: np.ndarray) -> float:
        prepared = self._prepare_values(values)
        if prepared.size == 0:
            return 0.0
        if prepared.size == 1 or np.unique(prepared).size == 1:
            return float(prepared[0])

        computer = ThresholdComputer(self.method_name, self.params)
        try:
            return float(computer.compute(prepared))
        except Exception:
            return float(np.median(prepared))


class OtsuStrategy(_ThresholdComputerStrategy):
    method_name = "otsu"


class YenStrategy(_ThresholdComputerStrategy):
    method_name = "yen"


class LiStrategy(_ThresholdComputerStrategy):
    method_name = "li"


class TriangleStrategy(_ThresholdComputerStrategy):
    method_name = "triangle"


class PercentileStrategy(_ThresholdComputerStrategy):
    method_name = "percentile"


class ManualStrategy(_ThresholdComputerStrategy):
    method_name = "manual"


def get_labeling_strategy(
    method: Method, params: ThresholdParams | None = None
) -> LabelingStrategy:
    """Return the configured strategy for instance labeling."""
    strategies: dict[str, type[LabelingStrategy]] = {
        "otsu": OtsuStrategy,
        "yen": YenStrategy,
        "li": LiStrategy,
        "triangle": TriangleStrategy,
        "percentile": PercentileStrategy,
        "manual": ManualStrategy,
    }
    if method not in strategies:
        raise ValueError(
            f"Method '{method}' is not supported for instance labeling. "
            f"Supported methods: {', '.join(SUPPORTED_LABELING_METHODS)}"
        )
    return strategies[method](params=params)


__all__ = [
    "SUPPORTED_LABELING_METHODS",
    "LabelingStrategy",
    "OtsuStrategy",
    "YenStrategy",
    "LiStrategy",
    "TriangleStrategy",
    "PercentileStrategy",
    "ManualStrategy",
    "get_labeling_strategy",
]