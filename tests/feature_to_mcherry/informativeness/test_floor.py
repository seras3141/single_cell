"""Tests for feature_to_mcherry.informativeness.floor."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.feature_to_mcherry.evaluation.cv import grouped_kfold_indices
from src.feature_to_mcherry.informativeness.floor import compute_floor


def _synthetic_signal_data(seed: int = 0, n_per_group: int = 40, n_groups: int = 6):
    rng = np.random.default_rng(seed)
    groups = np.repeat([f"A{i:02d}" for i in range(n_groups)], n_per_group)
    n = len(groups)

    f1 = rng.uniform(0, 10, size=n)
    f2 = rng.uniform(0, 10, size=n)
    noise = rng.normal(0, 0.2, size=n)
    target = 2 * f1 + 3 * f2 + noise

    X = np.column_stack([f1, f2])
    y = np.column_stack([target, target + 5.0])
    return X, y, groups


def _synthetic_noise_data(seed: int = 1, n_per_group: int = 40, n_groups: int = 6):
    rng = np.random.default_rng(seed)
    groups = np.repeat([f"A{i:02d}" for i in range(n_groups)], n_per_group)
    n = len(groups)

    X = rng.uniform(0, 10, size=(n, 2))
    y = rng.uniform(0, 10, size=(n, 2))  # unrelated to X
    return X, y, groups


TARGET_NAMES = ["percentile_75", "percentile_90"]
TAUS = [0.75, 0.90]


def test_compute_floor_high_r2_on_known_function_of_features() -> None:
    X, y, groups = _synthetic_signal_data()

    results = compute_floor(
        X,
        y,
        groups,
        TARGET_NAMES,
        TAUS,
        group_by="sample_id",
        n_splits=3,
        ridge_alpha=1.0,
        nonlinear_backend="sklearn",
    )

    for metrics in results["linear"].pooled_metrics:
        assert metrics["r2"] > 0.8, metrics
    for metrics in results["nonlinear"].pooled_metrics:
        assert math.isfinite(metrics["r2"])


def test_compute_floor_near_zero_r2_on_pure_noise_features() -> None:
    X, y, groups = _synthetic_noise_data()

    results = compute_floor(
        X,
        y,
        groups,
        TARGET_NAMES,
        TAUS,
        group_by="sample_id",
        n_splits=3,
        ridge_alpha=1.0,
        nonlinear_backend="sklearn",
    )

    for metrics in results["linear"].pooled_metrics:
        assert metrics["r2"] < 0.3, metrics


def test_compute_floor_sklearn_fallback_backend_used_is_sklearn() -> None:
    X, y, groups = _synthetic_signal_data()

    results = compute_floor(
        X,
        y,
        groups,
        TARGET_NAMES,
        TAUS,
        group_by="sample_id",
        n_splits=3,
        ridge_alpha=1.0,
        nonlinear_backend="sklearn",
    )

    assert results["nonlinear"].backend == "sklearn"


def test_compute_floor_lightgbm_backend_when_available() -> None:
    pytest.importorskip("lightgbm")
    X, y, groups = _synthetic_signal_data()

    results = compute_floor(
        X,
        y,
        groups,
        TARGET_NAMES,
        TAUS,
        group_by="sample_id",
        n_splits=3,
        ridge_alpha=1.0,
        nonlinear_backend="lightgbm",
    )

    assert results["nonlinear"].backend == "lightgbm"
    for metrics in results["nonlinear"].pooled_metrics:
        assert math.isfinite(metrics["r2"])


def test_compute_floor_lightgbm_requested_but_unavailable_raises(monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "lightgbm":
            raise ImportError("simulated missing lightgbm")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    from src.feature_to_mcherry.informativeness.floor import _make_nonlinear_regressor

    with pytest.raises(ValueError, match="lightgbm"):
        _make_nonlinear_regressor("lightgbm")


def test_compute_floor_auto_backend_falls_back_to_sklearn_when_lightgbm_missing(
    monkeypatch,
) -> None:
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "lightgbm":
            raise ImportError("simulated missing lightgbm")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    from src.feature_to_mcherry.informativeness.floor import _make_nonlinear_regressor

    _, backend_used = _make_nonlinear_regressor("auto")
    assert backend_used == "sklearn"


def test_grouped_kfold_indices_never_splits_a_group_across_train_and_val() -> None:
    X, y, groups = _synthetic_signal_data()

    for train_idx, val_idx in grouped_kfold_indices(
        X, groups, n_splits=3, group_by="sample_id"
    ):
        train_groups = set(groups[train_idx].tolist())
        val_groups = set(groups[val_idx].tolist())
        assert train_groups.isdisjoint(val_groups)
