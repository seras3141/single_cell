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


def test_normalize_to_dmso_requires_dmso_well() -> None:
    with pytest.raises(ValueError, match="dmso_well"):
        InformativenessConfig(**_valid_kwargs(normalize_to_dmso=True))


def test_normalize_to_dmso_with_well_is_valid() -> None:
    config = InformativenessConfig(
        **_valid_kwargs(normalize_to_dmso=True, dmso_well="M11")
    )
    assert config.normalize_to_dmso is True
    assert config.dmso_well == "M11"


# --- DMSO confidence gate (Step 3) -----------------------------------------------


def test_gate_defaults_to_off() -> None:
    config = InformativenessConfig(**_valid_kwargs())
    assert config.dmso_gate_min_peak_fraction is None
    assert config.dmso_gate_absolute_floor is None


def test_gate_without_normalization_raises() -> None:
    """Silently ignoring the gate would let a config claim a filter it never applied."""
    with pytest.raises(ValueError, match="only applies on the normalization path"):
        InformativenessConfig(**_valid_kwargs(dmso_gate_min_peak_fraction=0.1))


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.5, 10.0])
def test_gate_fraction_must_be_a_fraction(fraction: float) -> None:
    with pytest.raises(ValueError, match="dmso_gate_min_peak_fraction"):
        InformativenessConfig(
            **_valid_kwargs(
                normalize_to_dmso=True,
                dmso_well="M11",
                dmso_gate_min_peak_fraction=fraction,
            )
        )


def test_gate_floor_must_be_positive() -> None:
    with pytest.raises(ValueError, match="dmso_gate_absolute_floor"):
        InformativenessConfig(
            **_valid_kwargs(
                normalize_to_dmso=True,
                dmso_well="M11",
                dmso_gate_absolute_floor=0,
            )
        )


def test_gate_recommended_settings_are_valid() -> None:
    config = InformativenessConfig(
        **_valid_kwargs(
            normalize_to_dmso=True,
            dmso_well="M11",
            dmso_gate_min_peak_fraction=0.10,
            dmso_gate_absolute_floor=30,
        )
    )
    assert config.dmso_gate_min_peak_fraction == 0.10
    assert config.dmso_gate_absolute_floor == 30
