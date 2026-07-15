"""Tests for feature_to_mcherry.models.ridge."""

from __future__ import annotations

import numpy as np
import pytest

from src.feature_to_mcherry.models.ridge import RidgeMeanBaseline


def test_ridge_multioutput_shape() -> None:
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]])
    model = RidgeMeanBaseline(alpha=0.1).fit(X, y)
    preds = model.predict(X)
    assert preds.shape == y.shape


def test_ridge_scaler_is_fit_only_on_training_data() -> None:
    X_train = np.array([[1.0], [2.0], [3.0]])
    y_train = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]])

    model = RidgeMeanBaseline(alpha=0.0).fit(X_train, y_train)
    scaler = model.pipeline.named_steps["scaler"]

    assert scaler.mean_[0] == pytest.approx(X_train.mean())


def test_ridge_beats_constant_mean_baseline_on_linear_data() -> None:
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 10, size=(200, 3))
    coefs = np.array([2.0, -1.0, 0.5])
    noise = rng.normal(0, 0.1, size=200)
    y_col = X @ coefs + noise
    y = np.column_stack([y_col, 2 * y_col, 3 * y_col])

    n_train = 150
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    model = RidgeMeanBaseline(alpha=1.0).fit(X_train, y_train)
    preds = model.predict(X_test)

    ridge_mae = np.mean(np.abs(preds - y_test))
    constant_mae = np.mean(np.abs(y_test - y_train.mean(axis=0)))

    assert ridge_mae < constant_mae
