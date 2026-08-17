"""Tests for feature_to_mcherry.informativeness.config validation."""

from __future__ import annotations

import pytest

from src.feature_to_mcherry.informativeness.config import InformativenessConfig


def _valid_kwargs(**overrides):
    kwargs = dict(feature_csv="features.csv", target_csv="targets.csv")
    kwargs.update(overrides)
    return kwargs


def test_well_timepoint_defaults_are_valid() -> None:
    config = InformativenessConfig(**_valid_kwargs())
    assert config.well_timepoint_scatter_top_k == 3
    assert config.well_timepoint_scatter_max_wells is None
    assert config.well_timepoint_scatter_max_points_per_well == 2000
    assert config.well_timepoint_scatter_seed == 0
    assert config.well_timepoint_colormap == "viridis"


def test_well_timepoint_top_k_rejects_zero() -> None:
    with pytest.raises(ValueError, match="well_timepoint_scatter_top_k"):
        InformativenessConfig(**_valid_kwargs(well_timepoint_scatter_top_k=0))


def test_well_timepoint_top_k_accepts_one() -> None:
    config = InformativenessConfig(**_valid_kwargs(well_timepoint_scatter_top_k=1))
    assert config.well_timepoint_scatter_top_k == 1


def test_well_timepoint_max_wells_none_is_allowed() -> None:
    config = InformativenessConfig(
        **_valid_kwargs(well_timepoint_scatter_max_wells=None)
    )
    assert config.well_timepoint_scatter_max_wells is None


def test_well_timepoint_max_wells_rejects_zero() -> None:
    with pytest.raises(ValueError, match="well_timepoint_scatter_max_wells"):
        InformativenessConfig(**_valid_kwargs(well_timepoint_scatter_max_wells=0))


def test_well_timepoint_max_wells_accepts_positive() -> None:
    config = InformativenessConfig(**_valid_kwargs(well_timepoint_scatter_max_wells=4))
    assert config.well_timepoint_scatter_max_wells == 4


def test_well_timepoint_max_points_per_well_rejects_zero() -> None:
    with pytest.raises(ValueError, match="well_timepoint_scatter_max_points_per_well"):
        InformativenessConfig(
            **_valid_kwargs(well_timepoint_scatter_max_points_per_well=0)
        )


def test_well_timepoint_max_points_per_well_accepts_one() -> None:
    config = InformativenessConfig(
        **_valid_kwargs(well_timepoint_scatter_max_points_per_well=1)
    )
    assert config.well_timepoint_scatter_max_points_per_well == 1
