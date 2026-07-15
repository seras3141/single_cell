"""Tests for feature_to_mcherry.models.linear_quantile."""

from __future__ import annotations

import numpy as np
import pytest

from src.feature_to_mcherry.evaluation.metrics import pinball_loss
from src.feature_to_mcherry.models.linear_quantile import LinearQuantileRegressor
from src.feature_to_mcherry.pipeline import _sort_quantiles


def _skewed_data(n: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 10, size=(n, 2))
    base = X[:, 0] * 2 + X[:, 1]
    noise = rng.exponential(scale=2.0, size=n)  # right-skewed noise
    y = base + noise
    return X, y


def test_fit_predict_shape() -> None:
    X, y_col = _skewed_data()
    y = np.column_stack([y_col, y_col + 1, y_col + 2])
    model = LinearQuantileRegressor(taus=[0.75, 0.90, 0.95], alpha=0.0).fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(X), 3)


def test_predict_before_fit_raises() -> None:
    model = LinearQuantileRegressor(taus=[0.5])
    with pytest.raises(RuntimeError, match="before fit"):
        model.predict(np.zeros((2, 1)))


def test_fit_raises_on_tau_target_column_mismatch() -> None:
    X = np.zeros((5, 1))
    y = np.zeros((5, 2))
    model = LinearQuantileRegressor(taus=[0.5, 0.75, 0.9])
    with pytest.raises(ValueError, match="taus"):
        model.fit(X, y)


def test_higher_tau_yields_higher_predictions_on_skewed_target() -> None:
    X, y_col = _skewed_data()
    y = np.column_stack([y_col, y_col])
    model = LinearQuantileRegressor(taus=[0.1, 0.9], alpha=0.0).fit(X, y)
    preds = model.predict(X)
    assert preds[:, 1].mean() > preds[:, 0].mean()


def test_pinball_loss_is_minimized_at_the_correct_tau() -> None:
    X, y = _skewed_data()
    tau = 0.9
    model = LinearQuantileRegressor(taus=[tau], alpha=0.0).fit(X, y.reshape(-1, 1))
    preds = model.predict(X)[:, 0]

    correct_loss = pinball_loss(y, preds, tau)
    mismatched_loss = pinball_loss(y, preds, 0.1)
    assert correct_loss < mismatched_loss


def test_sort_quantiles_enforces_monotonic_rows() -> None:
    preds = np.array([[5.0, 2.0, 8.0], [1.0, 1.0, 0.5]])
    sorted_preds = _sort_quantiles(preds)
    assert np.all(np.diff(sorted_preds, axis=1) >= 0)
