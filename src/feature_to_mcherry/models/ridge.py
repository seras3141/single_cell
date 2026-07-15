"""Ridge regression mean baseline."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class RidgeMeanBaseline:
    """Multi-output Ridge regression predicting the conditional mean of each target.

    A single ``Ridge`` model is fit across all target columns at once (scikit-learn's
    ``Ridge`` natively supports 2D ``y``), rather than one independent model per
    column, since the targets share the same underlying feature representation.

    This predicts a mean, not a quantile — it is a feature-quality sanity floor, not a
    competitor to the pinball-loss quantile model.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=alpha)),
            ]
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeMeanBaseline":
        """Fit the scaler and Ridge model on training data.

        The scaler is fit only on ``X`` passed here — callers must pass training-fold
        data only, never the full dataset, to keep cross-validation leakage-safe.
        """
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the conditional mean of each target; shape (n_samples, n_targets)."""
        return np.asarray(self.pipeline.predict(X))
