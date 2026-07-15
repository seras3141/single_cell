"""Linear quantile regression baseline."""

from __future__ import annotations

from typing import List

import numpy as np
from sklearn.linear_model import QuantileRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class LinearQuantileRegressor:
    """Independent linear quantile regression model per tau, trained with pinball loss.

    Fits ``sklearn.linear_model.QuantileRegressor`` once per tau. Because each tau is
    fit independently, quantile crossing (e.g. predicted p75 > p90) can occur — this
    class does not enforce monotonicity. Crossing is measured by
    ``evaluation.metrics.quantile_crossing_rate``; an optional post-hoc sort is applied
    in ``pipeline.py`` (controlled by the ``sort_quantiles`` config flag), not here.
    """

    def __init__(
        self, taus: List[float], alpha: float = 0.0, solver: str = "highs"
    ) -> None:
        self.taus = list(taus)
        self.alpha = alpha
        self.solver = solver
        self.pipelines: List[Pipeline] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearQuantileRegressor":
        """Fit one quantile-regression pipeline per tau.

        ``y`` must have one column per tau, in the same order as ``self.taus``. The
        scaler for each pipeline is fit only on ``X``/``y`` passed here — callers must
        pass training-fold data only, never the full dataset.
        """
        if y.shape[1] != len(self.taus):
            raise ValueError(
                f"y has {y.shape[1]} target columns but {len(self.taus)} taus were "
                "configured; each tau must correspond 1:1 to a target column."
            )

        self.pipelines = []
        for i, tau in enumerate(self.taus):
            pipeline = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "quantile",
                        QuantileRegressor(
                            quantile=tau, alpha=self.alpha, solver=self.solver
                        ),
                    ),
                ]
            )
            pipeline.fit(X, y[:, i])
            self.pipelines.append(pipeline)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict each tau's quantile.

        Returns shape (n_samples, n_taus), columns ordered as ``self.taus``.
        """
        if not self.pipelines:
            raise RuntimeError("LinearQuantileRegressor.predict called before fit().")
        return np.column_stack([pipeline.predict(X) for pipeline in self.pipelines])
