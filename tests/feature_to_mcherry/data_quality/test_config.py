"""Tests for feature_to_mcherry.data_quality.config."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.feature_to_mcherry.data_quality.config import (
    DataQualityConfig,
    SourceConfig,
    load_config,
)


def test_source_config_requires_label_feature_csv_target_csv() -> None:
    with pytest.raises(ValueError, match="label"):
        SourceConfig(label="", feature_csv="f.csv", target_csv="t.csv")
    with pytest.raises(ValueError, match="feature_csv"):
        SourceConfig(label="A", feature_csv="", target_csv="t.csv")
    with pytest.raises(ValueError, match="target_csv"):
        SourceConfig(label="A", feature_csv="f.csv", target_csv="")


def test_data_quality_config_requires_at_least_one_source() -> None:
    with pytest.raises(ValueError, match="sources"):
        DataQualityConfig(sources=[])


def test_data_quality_config_validates_quantile_bounds() -> None:
    source = SourceConfig(label="A", feature_csv="f.csv", target_csv="t.csv")
    with pytest.raises(ValueError, match="extreme_quantile_lo"):
        DataQualityConfig(sources=[source], extreme_quantile_lo=0.6)
    with pytest.raises(ValueError, match="extreme_quantile_hi"):
        DataQualityConfig(sources=[source], extreme_quantile_hi=0.4)


def test_load_config_round_trips_multi_source_yaml(tmp_path: Path) -> None:
    yaml_text = """
data_quality:
  sources:
    - label: Src1
      feature_csv: /a/features
      target_csv: /a/targets.csv
      id_column: cell_id
    - label: Src2
      feature_csv: /b/features
      target_csv: /b/targets.csv
  extreme_quantile_lo: 0.01
  extreme_quantile_hi: 0.99
  feature_column_groups:
    Shape: [area, perimeter]
  flag_timepoints: [11]
  output_dir: results/data_quality/test
"""
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(yaml_text)

    config = load_config(yaml_path=yaml_path)

    assert len(config.sources) == 2
    assert config.sources[0].label == "Src1"
    assert config.sources[0].id_column == "cell_id"
    assert config.sources[1].id_column == "instance_id"  # default
    assert config.extreme_quantile_lo == 0.01
    assert config.extreme_quantile_hi == 0.99
    assert config.feature_column_groups == {"Shape": ["area", "perimeter"]}
    assert config.flag_timepoints == [11]
    assert config.output_dir == "results/data_quality/test"


def test_load_config_applies_dot_notation_overrides(tmp_path: Path) -> None:
    yaml_text = """
data_quality:
  sources:
    - label: Src1
      feature_csv: /a/features
      target_csv: /a/targets.csv
"""
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(yaml_text)

    config = load_config(yaml_path=yaml_path, overrides=["extreme_quantile_hi=0.995"])

    assert config.extreme_quantile_hi == 0.995


def test_data_quality_config_defaults_feature_column_groups_to_none() -> None:
    source = SourceConfig(label="A", feature_csv="f.csv", target_csv="t.csv")
    config = DataQualityConfig(sources=[source])

    assert config.feature_column_groups is None
