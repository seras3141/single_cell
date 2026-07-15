"""Tests for feature_to_mcherry.evaluation.metrics."""

from __future__ import annotations

import numpy as np
import pytest

from src.feature_to_mcherry.evaluation.metrics import (
    per_target_regression_metrics,
    pinball_loss,
    quantile_crossing_rate,
)


def test_pinball_loss_zero_for_perfect_predictions() -> None:
    y = np.array([1.0, 2.0, 3.0])
    assert pinball_loss(y, y, tau=0.75) == 0.0


def test_pinball_loss_hand_computed() -> None:
    y_true = np.array([10.0])
    y_pred = np.array([8.0])  # under-prediction: diff = 2.0
    tau = 0.9
    # loss = max(tau*diff, (tau-1)*diff) = max(0.9*2.0, -0.1*2.0) = max(1.8, -0.2) = 1.8
    assert pinball_loss(y_true, y_pred, tau) == pytest.approx(1.8)


def test_quantile_crossing_rate_detects_violations() -> None:
    # row 0: 5 > 2 -> crossing; row 1: monotone non-decreasing -> no crossing
    y_pred = np.array([[5.0, 2.0, 8.0], [1.0, 2.0, 3.0]])
    assert quantile_crossing_rate(y_pred) == pytest.approx(0.5)


def test_quantile_crossing_rate_zero_when_all_monotone() -> None:
    y_pred = np.array([[1.0, 2.0, 3.0], [4.0, 4.0, 5.0]])
    assert quantile_crossing_rate(y_pred) == 0.0


def test_quantile_crossing_rate_single_column_is_zero() -> None:
    y_pred = np.array([[1.0], [2.0]])
    assert quantile_crossing_rate(y_pred) == 0.0


def test_per_target_regression_metrics_hand_computed() -> None:
    y_true = np.array([[1.0], [2.0], [3.0]])
    y_pred = np.array([[1.0], [2.0], [3.0]])  # perfect predictions

    results = per_target_regression_metrics(
        y_true, y_pred, taus=[0.75], target_names=["percentile_75"]
    )

    assert len(results) == 1
    assert results[0]["target"] == "percentile_75"
    assert results[0]["tau"] == 0.75
    assert results[0]["mae"] == pytest.approx(0.0)
    assert results[0]["r2"] == pytest.approx(1.0)
    assert results[0]["pinball_loss"] == pytest.approx(0.0)
