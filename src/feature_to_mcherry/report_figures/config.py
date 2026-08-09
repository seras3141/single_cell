"""Configuration schema for the report-figures generator.

Mirrors the standalone-config precedent set by ``feature_to_mcherry.config`` and
``informativeness.config`` (a dataclass + OmegaConf YAML/CLI loader, constructed
directly rather than registered into the main pipeline's global ``ConfigManager``).
Unlike those two, every field here has a real default: this module reads
already-computed results rather than raw per-cell tables, so there is no valid-preset
question to defer with a required ``???`` field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from omegaconf import DictConfig, OmegaConf

DEFAULT_EXPERIMENTS: List[str] = ["HD1509", "SA110", "HD1883", "Ew2-1", "Ew2-2"]
DEFAULT_SCPORTRAIT_EXPERIMENTS: List[str] = ["HD1509", "SA110", "Ew2-2"]
DEFAULT_A5_EXPERIMENTS: List[str] = ["HD1509", "SA110"]
DEFAULT_COHORT_SIZES: Dict[str, int] = {
    "HD1509": 851800,
    "SA110": 439016,
    "HD1883": 261014,
    "Ew2-1": 445113,
    "Ew2-2": 143887,
}


@dataclass
class ReportFiguresConfig:
    """Configuration for generating the report figures in
    ``docs_local/figures/report_figures_brief.md``.

    Parameters
    ----------
    experiments : list[str]
        All experiments the figures should attempt to cover.
    scportrait_experiments : list[str]
        Subset of ``experiments`` that have a scPortrait comparison run (A2, D3).
    a5_experiments : list[str]
        Subset of ``experiments`` covered by A5 (segmentation diagnosis) --
        narrower than, and not required to coincide with, ``scportrait_experiments``.
    a5_area_p90_dir : str
        Directory holding A5's extracted ``<exp>_<mask_source>.csv`` files (see
        ``docs_local/sync_a5_area_p90.sh``), each just the ``area``/
        ``percentile_90`` columns of the corresponding real
        ``mcherry_metrics/<source>/instance_metrics.csv``.
    cohort_sizes : dict[str, int]
        Fallback per-cell counts (D4) used when ``instance_metrics.csv`` isn't
        locally available for an experiment.
    feature_to_mcherry_dir, informativeness_dir, data_quality_dir : str
        Roots of the corresponding result trees (see ``report_figures.paths``).
    plate_layout_json : str
        Path to the plate-layout JSON (well -> drug/dose).
    output_dir : str
        Directory figures and the manifest are written to.
    dpi : int
        Raster (PNG) resolution.
    formats : list[str]
        File formats written per figure, e.g. ``["png", "pdf"]``.
    seed : int
        Random seed for any subsampling.
    """

    experiments: List[str] = field(default_factory=lambda: list(DEFAULT_EXPERIMENTS))
    scportrait_experiments: List[str] = field(
        default_factory=lambda: list(DEFAULT_SCPORTRAIT_EXPERIMENTS)
    )
    a5_experiments: List[str] = field(
        default_factory=lambda: list(DEFAULT_A5_EXPERIMENTS)
    )
    a5_area_p90_dir: str = "data/scportrait_diagnostics_area_p90"
    cohort_sizes: Dict[str, int] = field(
        default_factory=lambda: dict(DEFAULT_COHORT_SIZES)
    )
    feature_to_mcherry_dir: str = "results/feature_to_mcherry"
    informativeness_dir: str = "results/morphology_informativeness"
    data_quality_dir: str = "results/data_quality"
    data_root: str = "data/MF5V1_processed Timelapse samples 19.03.2024"
    plate_layout_json: str = "config/MF5v1_plate_layout.json"
    output_dir: str = "results/report_figures"
    dpi: int = 300
    formats: List[str] = field(default_factory=lambda: ["png", "pdf"])
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.experiments:
            raise ValueError("experiments must contain at least one experiment")
        unknown_scportrait = set(self.scportrait_experiments) - set(self.experiments)
        if unknown_scportrait:
            raise ValueError(
                "scportrait_experiments contains experiments not in "
                f"experiments: {sorted(unknown_scportrait)}"
            )
        unknown_a5 = set(self.a5_experiments) - set(self.experiments)
        if unknown_a5:
            raise ValueError(
                f"a5_experiments contains experiments not in experiments: "
                f"{sorted(unknown_a5)}"
            )
        if not self.formats:
            raise ValueError("formats must contain at least one format")
        if self.dpi < 1:
            raise ValueError("dpi must be positive")


def load_config(
    yaml_path: Optional[Path] = None,
    overrides: Optional[List[str]] = None,
) -> ReportFiguresConfig:
    """Build a :class:`ReportFiguresConfig` from a YAML file plus overrides.

    Parameters
    ----------
    yaml_path : Path, optional
        Path to a YAML file. If it has a top-level ``report_figures:`` key, that
        subsection is used; otherwise the file's top level is used directly.
    overrides : list[str], optional
        Dot-notation overrides, e.g. ``["dpi=150", "output_dir=/tmp/figs"]``.

    Returns
    -------
    ReportFiguresConfig
    """
    schema = OmegaConf.structured(ReportFiguresConfig)

    if yaml_path is not None:
        file_config = OmegaConf.load(yaml_path)
        if isinstance(file_config, DictConfig) and "report_figures" in file_config:
            file_config = file_config.report_figures
        schema = OmegaConf.merge(schema, file_config)

    if overrides:
        schema = OmegaConf.merge(schema, OmegaConf.from_dotlist(list(overrides)))

    config_object = OmegaConf.to_object(schema)
    if not isinstance(config_object, ReportFiguresConfig):
        raise TypeError(
            f"Expected ReportFiguresConfig after merge, got {type(config_object)!r}"
        )
    return config_object
