"""Configuration schema for the morphology-informativeness feasibility gate.

Mirrors the standalone-config precedent set by ``feature_to_mcherry.config`` (a
dataclass + OmegaConf YAML/CLI loader, constructed directly rather than registered
into the main pipeline's global ``ConfigManager``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from omegaconf import DictConfig, OmegaConf

from ..data.contract import TARGET_COLUMNS

DEFAULT_MORPHOLOGY_PATTERNS: List[str] = [
    "area",
    "perimeter",
    "eccentricity",
    "solidity",
    "extent",
    "major_axis_length",
    "minor_axis_length",
    "aspect_ratio",
    "*contrast",
    "*correlation",
    "*entropy",
    "*idm",
    "*homogeneity",
]

DEFAULT_SUSPECT_PATTERNS: List[str] = [
    "*mean_intensity",
    "*_intensity_std",
    "*intensity_std*",
]


@dataclass
class InformativenessConfig:
    """Configuration for the morphology-informativeness feasibility gate.

    Parameters
    ----------
    feature_csv, target_csv : str
        Paths to the per-cell feature CSV and the ``mcherry_metrics`` instance-metrics
        CSV — each may instead be a *directory* of per-(well, timepoint, z) CSVs,
        auto-detected via ``Path.is_dir()``. Required — no valid default.
    id_column, sample_id_column, timepoint_column, z_index_column : str, optional
        Same semantics as :class:`feature_to_mcherry.config.FeatureToMcherryConfig`,
        including the directory-input auto-detection for both ``feature_csv`` and
        ``target_csv``.
    target_columns : list[str]
        mCherry percentile columns used as targets.
    group_by : {"sample_id", "timepoint"}
        Grouping column for grouped cross-validation.
    n_splits : int
        Number of grouped CV folds for the floor models.
    morphology_feature_patterns : list[str]
        ``fnmatch``-style patterns (case-insensitive) selecting interpretable
        size/shape/texture feature columns from the feature CSV.
    suspect_feature_patterns : list[str]
        Patterns identifying features that are size/thickness/focus proxies (e.g.
        raw brightfield intensity) rather than unambiguous morphology signal. Kept in
        the "with suspect" floor variant, excluded from the "without suspect" one.
    ridge_alpha : float
        L2 regularization strength for the linear floor model.
    nonlinear_backend : {"auto", "lightgbm", "sklearn"}
        Nonlinear floor model backend. ``"auto"`` uses LightGBM if importable, else
        falls back to ``sklearn.ensemble.GradientBoostingRegressor``.
    top_k_features : int
        Number of top-associated features per target to plot in the scatter figures.
    plate_layout_json : str, optional
        Path to a plate-layout JSON (see ``config/MF5v1_plate_layout.json``) used to
        map wells to drug/dose conditions for the noise-ceiling estimate. If ``None``
        or the file is missing, the noise ceiling is reported as unavailable.
    normalize_to_dmso : bool
        If True, DMSO-normalize the targets before the gate: per-timepoint robust
        z-scores against ``dmso_well`` (``z_``-prefixed), dropping undefined-reference
        cells; the univariate/floor/noise-ceiling then run on the normalized targets.
        Requires ``dmso_well``. **The target input must cover a single experiment** (the
        loaders drop any non-target column, so there is nothing to scope by); for
        multi-experiment use call ``data.normalize.normalize_targets_to_dmso`` directly
        with ``experiment_column``. See ``data.normalize``.
    dmso_well : str, optional
        DMSO control well label; required when ``normalize_to_dmso`` is True.
    output_dir : str
        Directory to write the results bundle (CSVs, JSON, report, figures) to.
    """

    feature_csv: str
    target_csv: str
    id_column: str = "instance_id"
    sample_id_column: Optional[str] = None
    timepoint_column: Optional[str] = None
    z_index_column: Optional[str] = None
    target_columns: List[str] = field(default_factory=lambda: list(TARGET_COLUMNS))
    group_by: str = "sample_id"
    n_splits: int = 5
    morphology_feature_patterns: List[str] = field(
        default_factory=lambda: list(DEFAULT_MORPHOLOGY_PATTERNS)
    )
    suspect_feature_patterns: List[str] = field(
        default_factory=lambda: list(DEFAULT_SUSPECT_PATTERNS)
    )
    ridge_alpha: float = 1.0
    nonlinear_backend: str = "auto"
    top_k_features: int = 5
    well_timepoint_scatter_top_k: int = 3
    well_timepoint_scatter_max_wells: Optional[int] = None
    well_timepoint_scatter_max_points_per_well: int = 2000
    well_timepoint_scatter_seed: int = 0
    well_timepoint_colormap: str = "viridis"
    plate_layout_json: Optional[str] = None
    normalize_to_dmso: bool = False
    dmso_well: Optional[str] = None
    output_dir: str = "results/morphology_informativeness"

    def __post_init__(self) -> None:
        if not self.feature_csv:
            raise ValueError("feature_csv must be set")
        if not self.target_csv:
            raise ValueError("target_csv must be set")
        if not self.target_columns:
            raise ValueError("target_columns must contain at least one column")
        if self.group_by not in ("sample_id", "timepoint"):
            raise ValueError(
                f"group_by must be 'sample_id' or 'timepoint', got {self.group_by!r}"
            )
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if not self.morphology_feature_patterns:
            raise ValueError(
                "morphology_feature_patterns must contain at least one pattern"
            )
        if self.nonlinear_backend not in ("auto", "lightgbm", "sklearn"):
            raise ValueError(
                "nonlinear_backend must be 'auto', 'lightgbm', or 'sklearn', got "
                f"{self.nonlinear_backend!r}"
            )
        if self.top_k_features < 1:
            raise ValueError("top_k_features must be at least 1")
        if self.well_timepoint_scatter_top_k < 1:
            raise ValueError("well_timepoint_scatter_top_k must be at least 1")
        if (
            self.well_timepoint_scatter_max_wells is not None
            and self.well_timepoint_scatter_max_wells < 1
        ):
            raise ValueError(
                "well_timepoint_scatter_max_wells must be None or at least 1"
            )
        if self.well_timepoint_scatter_max_points_per_well < 1:
            raise ValueError(
                "well_timepoint_scatter_max_points_per_well must be at least 1"
            )
        if self.normalize_to_dmso and not self.dmso_well:
            raise ValueError("dmso_well must be set when normalize_to_dmso is True")


def load_config(
    yaml_path: Optional[Path] = None,
    overrides: Optional[List[str]] = None,
) -> InformativenessConfig:
    """Build an :class:`InformativenessConfig` from a YAML file plus overrides.

    Parameters
    ----------
    yaml_path : Path, optional
        Path to a YAML file. If it has a top-level ``morphology_informativeness:``
        key, that subsection is used; otherwise the file's top level is used
        directly.
    overrides : list[str], optional
        Dot-notation overrides, e.g. ``["ridge_alpha=10", "n_splits=3"]``.

    Returns
    -------
    InformativenessConfig
    """
    schema = OmegaConf.structured(InformativenessConfig)

    if yaml_path is not None:
        file_config = OmegaConf.load(yaml_path)
        if (
            isinstance(file_config, DictConfig)
            and "morphology_informativeness" in file_config
        ):
            file_config = file_config.morphology_informativeness
        schema = OmegaConf.merge(schema, file_config)

    if overrides:
        schema = OmegaConf.merge(schema, OmegaConf.from_dotlist(list(overrides)))

    config_object = OmegaConf.to_object(schema)
    if not isinstance(config_object, InformativenessConfig):
        raise TypeError(
            f"Expected InformativenessConfig after merge, got {type(config_object)!r}"
        )
    return config_object
