"""Tests for feature_to_mcherry.informativeness.univariate."""

from __future__ import annotations

import numpy as np

from src.feature_to_mcherry.informativeness.univariate import (
    compute_univariate_associations,
    top_associations,
)


def _synthetic_data(seed: int = 0, n_per_group: int = 30):
    rng = np.random.default_rng(seed)
    groups = np.repeat(["A01", "A02", "A03", "A04"], n_per_group)
    n = len(groups)

    monotone_feature = rng.uniform(0, 10, size=n)
    noise_feature = rng.uniform(0, 10, size=n)
    target = monotone_feature + rng.normal(0, 0.1, size=n)

    X = np.column_stack([monotone_feature, noise_feature])
    y = target.reshape(-1, 1)
    return X, y, groups


def test_compute_univariate_associations_detects_injected_monotone_relationship() -> (
    None
):
    X, y, groups = _synthetic_data()
    feature_names = ["monotone_feature", "noise_feature"]
    target_names = ["percentile_75"]

    result = compute_univariate_associations(X, y, groups, feature_names, target_names)

    pooled = result[result["scope"] == "pooled"]
    monotone_rho = pooled[pooled["feature"] == "monotone_feature"]["rho"].iloc[0]
    noise_rho = pooled[pooled["feature"] == "noise_feature"]["rho"].iloc[0]

    assert monotone_rho > 0.9
    assert abs(noise_rho) < 0.3


def test_compute_univariate_associations_includes_per_group_scope() -> None:
    X, y, groups = _synthetic_data()
    feature_names = ["monotone_feature", "noise_feature"]
    target_names = ["percentile_75"]

    result = compute_univariate_associations(X, y, groups, feature_names, target_names)

    per_group = result[result["scope"] == "per_group"]
    assert set(per_group["group_id"]) == {"A01", "A02", "A03", "A04"}
    # Each feature x target x group combination is present.
    assert len(per_group) == 2 * 1 * 4

    pooled = result[result["scope"] == "pooled"]
    assert len(pooled) == 2 * 1
    assert (pooled["group_id"] == "").all()


def test_top_associations_returns_top_k_by_absolute_rho() -> None:
    X, y, groups = _synthetic_data()
    feature_names = ["monotone_feature", "noise_feature"]
    target_names = ["percentile_75"]
    result = compute_univariate_associations(X, y, groups, feature_names, target_names)

    top = top_associations(result, target="percentile_75", top_k=1)

    assert len(top) == 1
    assert top.iloc[0]["feature"] == "monotone_feature"
