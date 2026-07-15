"""Configuration schema for the feature/target data-quality diagnostics.

Mirrors the standalone-config precedent set by ``feature_to_mcherry.config`` and
``feature_to_mcherry.informativeness.config`` (a dataclass + OmegaConf YAML/CLI
loader, constructed directly rather than registered into the main pipeline's global
``ConfigManager``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from omegaconf import DictConfig, OmegaConf

from ..data.contract import TARGET_COLUMNS


@dataclass
class SourceConfig:
    """One experiment/sample to load and analyze.

    Parameters
    ----------
    label : str
        Short identifier for this source, used in report headings, well-color
        legends, and the ``source`` column of the extreme-value report (e.g.
        ``"Ew2-1"``).
    feature_csv, target_csv : str
        Same semantics as :class:`feature_to_mcherry.config.FeatureToMcherryConfig`
        — a single CSV, or a directory of per-(well, timepoint, z) CSVs, auto-detected
        via ``Path.is_dir()``.
    id_column, sample_id_column, timepoint_column, z_index_column : str, optional
        Same semantics as :class:`feature_to_mcherry.config.FeatureToMcherryConfig`.
    exclude_feature_columns : list[str]
        Feature-CSV columns to drop before analysis (e.g. a column known to be
        entirely NaN in this source's backend output).
    """

    label: str
    feature_csv: str
    target_csv: str
    id_column: str = "instance_id"
    sample_id_column: Optional[str] = None
    timepoint_column: Optional[str] = None
    z_index_column: Optional[str] = None
    exclude_feature_columns: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("SourceConfig.label must be set")
        if not self.feature_csv:
            raise ValueError(f"SourceConfig {self.label!r}: feature_csv must be set")
        if not self.target_csv:
            raise ValueError(f"SourceConfig {self.label!r}: target_csv must be set")


@dataclass
class DataQualityConfig:
    """Configuration for the feature/target data-quality diagnostics.

    Parameters
    ----------
    sources : list[SourceConfig]
        One or more experiments/samples to load and analyze. Multiple sources are
        analyzed together (e.g. side-by-side well-timepoint comparison); wells with
        the same name across sources are kept visually consistent (same color) in
        the interactive report.
    target_columns : list[str]
        mCherry percentile columns to include alongside feature columns.
    extreme_quantile_lo, extreme_quantile_hi : float
        Quantile bounds defining "extreme": a value below the ``lo`` quantile or
        above the ``hi`` quantile is flagged. Thresholds are computed independently
        per source (not pooled across sources), matching the ad hoc investigation
        this module productionizes — a source with a systematically different scale
        shouldn't set the threshold for another source. Defaults: 0.001/0.999.
    feature_column_groups : dict[str, list[str]], optional
        Maps a category label (e.g. ``"Size & shape"``) to a list of ``fnmatch``
        patterns selecting feature columns for that category, for the interactive
        report's card grouping. ``None`` (default) puts every feature column in a
        single ungrouped "Features" bucket — this is what makes the tool usable
        out-of-the-box on a new experiment with unfamiliar column names; a specific
        experiment's own config file can supply the categorization it wants (see
        ``config/data_quality_config.yaml`` for the MF5V1/incarta example).
    flag_timepoints : list[int]
        Timepoints to mark with a dashed vertical reference line in the interactive
        report (e.g. a timepoint found to be enriched for extreme values).
    output_dir : str
        Directory to write the extreme-value report / interactive HTML report to.
    """

    sources: List[SourceConfig]
    target_columns: List[str] = field(default_factory=lambda: list(TARGET_COLUMNS))
    extreme_quantile_lo: float = 0.001
    extreme_quantile_hi: float = 0.999
    feature_column_groups: Optional[Dict[str, List[str]]] = None
    flag_timepoints: List[int] = field(default_factory=list)
    output_dir: str = "results/data_quality"

    def __post_init__(self) -> None:
        if not self.sources:
            raise ValueError("sources must contain at least one SourceConfig")
        if not self.target_columns:
            raise ValueError("target_columns must contain at least one column")
        if not (0.0 <= self.extreme_quantile_lo < 0.5):
            raise ValueError(
                "extreme_quantile_lo must be in [0.0, 0.5), got "
                f"{self.extreme_quantile_lo!r}"
            )
        if not (0.5 < self.extreme_quantile_hi <= 1.0):
            raise ValueError(
                "extreme_quantile_hi must be in (0.5, 1.0], got "
                f"{self.extreme_quantile_hi!r}"
            )


def load_config(
    yaml_path: Optional[Path] = None,
    overrides: Optional[List[str]] = None,
) -> DataQualityConfig:
    """Build a :class:`DataQualityConfig` from a YAML file plus overrides.

    Parameters
    ----------
    yaml_path : Path, optional
        Path to a YAML file. If it has a top-level ``data_quality:`` key, that
        subsection is used; otherwise the file's top level is used directly.
    overrides : list[str], optional
        Dot-notation overrides, e.g. ``["extreme_quantile_hi=0.995"]``.

    Returns
    -------
    DataQualityConfig
    """
    schema = OmegaConf.structured(DataQualityConfig)

    if yaml_path is not None:
        file_config = OmegaConf.load(yaml_path)
        if isinstance(file_config, DictConfig) and "data_quality" in file_config:
            file_config = file_config.data_quality
        schema = OmegaConf.merge(schema, file_config)

    if overrides:
        schema = OmegaConf.merge(schema, OmegaConf.from_dotlist(list(overrides)))

    config_object = OmegaConf.to_object(schema)
    if not isinstance(config_object, DataQualityConfig):
        raise TypeError(
            f"Expected DataQualityConfig after merge, got {type(config_object)!r}"
        )
    return config_object
