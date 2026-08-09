"""Tests for feature_to_mcherry.report_figures.config."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.feature_to_mcherry.report_figures.config import (
    ReportFiguresConfig,
    load_config,
)


def test_defaults_load() -> None:
    config = ReportFiguresConfig()
    assert config.experiments == ["HD1509", "SA110", "HD1883", "Ew2-1", "Ew2-2"]
    assert config.scportrait_experiments == ["HD1509", "SA110", "Ew2-2"]
    assert config.a5_experiments == ["HD1509", "SA110"]
    assert config.a5_area_p90_dir == "data/scportrait_diagnostics_area_p90"
    assert config.output_dir == "results/report_figures"
    assert config.dpi == 300
    assert config.formats == ["png", "pdf"]


def test_rejects_empty_experiments() -> None:
    with pytest.raises(ValueError, match="experiments"):
        ReportFiguresConfig(experiments=[])


def test_rejects_scportrait_experiment_not_in_experiments() -> None:
    with pytest.raises(ValueError, match="scportrait_experiments"):
        ReportFiguresConfig(experiments=["A"], scportrait_experiments=["B"])


def test_rejects_a5_experiment_not_in_experiments() -> None:
    with pytest.raises(ValueError, match="a5_experiments"):
        ReportFiguresConfig(
            experiments=["A"], scportrait_experiments=[], a5_experiments=["B"]
        )


def test_load_config_yaml_override(tmp_path: Path) -> None:
    yaml_text = """
report_figures:
  experiments: [Ew2-1, Ew2-2]
  scportrait_experiments: []
  a5_experiments: []
  dpi: 150
  output_dir: /tmp/figs
"""
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(yaml_text)

    config = load_config(yaml_path=yaml_path)

    assert config.experiments == ["Ew2-1", "Ew2-2"]
    assert config.dpi == 150
    assert config.output_dir == "/tmp/figs"


def test_load_config_dotlist_override(tmp_path: Path) -> None:
    yaml_text = "report_figures:\n  dpi: 150\n"
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(yaml_text)

    config = load_config(yaml_path=yaml_path, overrides=["dpi=72"])

    assert config.dpi == 72
