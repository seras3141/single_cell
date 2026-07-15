"""Configuration schema for the feature_to_mcherry module.

This module keeps its own lightweight config (dataclass + OmegaConf-based YAML/CLI
loading), rather than registering into the main pipeline's global
``src.utils.config.ConfigManager``/``PipelineConfig``. This mirrors the precedent set
by the closest sibling module, ``src.mcherry_metrics`` (its own
``config/models.py::ExtractionConfig``, constructed directly rather than through the
global schema) — feature_to_mcherry is a standalone analysis module, not part of the
main segmentation/tracking/inference pipeline that ``ConfigManager`` governs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from omegaconf import DictConfig, OmegaConf

from .data.contract import TARGET_COLUMNS


@dataclass
class FeatureToMcherryConfig:
    """Configuration for the feature-to-mCherry-percentile regression baselines.

    Parameters
    ----------
    feature_csv : str
        Path to the per-cell feature CSV (any backend from ``src.feature_extraction``),
        or a *directory* of per-(well, timepoint, z) feature CSVs (e.g. the incarta
        ``split_data`` layout) — auto-detected via ``Path.is_dir()`` and loaded with
        :func:`data.loaders.load_features_from_directory` instead of
        :func:`data.loaders.load_features`. Required — no default, since there is no
        valid preset pair to fall back to.
    target_csv : str
        Path to an ``src.mcherry_metrics`` instance-metrics CSV, or a *directory* of
        per-(well, timepoint, z) target CSVs (e.g. a ``split_data`` directory written
        alongside the combined ``instance_metrics.csv``) — auto-detected via
        ``Path.is_dir()`` and loaded with
        :func:`data.loaders.load_targets_from_directory` instead of
        :func:`data.loaders.load_targets`. Required, no default.
    id_column : str
        Name of the per-cell id column in the feature CSV (e.g. ``"instance_id"`` for
        the regionprops/incarta backends). Renamed to ``cell_id`` internally.
    sample_id_column, timepoint_column, z_index_column : str, optional
        Column names in the feature CSV already carrying ``sample_id``/``timepoint``/
        ``z_index``. Resolution order per field (see
        :func:`data.loaders.load_features`): explicit column here → an already-present
        column of the canonical name (e.g. the incarta ``split_data`` layout, which
        already carries these per row) → derived from the feature CSV's
        ``image_filename`` column (e.g. the regionprops layout) → error.
    target_columns : list[str]
        mCherry percentile columns used as regression targets.
    exclude_feature_columns : list[str]
        Feature-CSV columns to drop before building the model matrix (e.g. a column
        known to be entirely NaN in a given backend's output, or a positional column
        like a centroid that carries no morphology signal). Applied after loading,
        by name; a name not present in the loaded feature table is ignored.
    group_by : {"sample_id", "timepoint"}
        Column used for grouped cross-validation.
    n_splits : int
        Number of grouped CV folds.
    ridge_alpha : float
        L2 regularization strength for the Ridge baseline.
    quantile_alpha : float
        L1 regularization strength for the linear quantile regression baseline.
    quantile_solver : str
        Solver passed to ``sklearn.linear_model.QuantileRegressor``.
    quantile_train_subsample_size : int, optional
        If set, cap the number of training rows passed to the linear-quantile
        model's fit in each CV fold to at most this many (a fixed-seed random
        subsample; validation is always done on the full fold, so every cell still
        gets a real out-of-fold prediction). ``QuantileRegressor``'s LP solver scales
        poorly past tens of thousands of rows — this makes real-scale datasets
        (100K+ cells) tractable at the cost of fitting on less data. Ridge is
        unaffected (closed-form, doesn't need it). ``None`` disables subsampling.
    quantile_train_subsample_seed : int
        Seed for the training subsample above, for reproducibility.
    sort_quantiles : bool
        If True, sort predicted quantiles ascending per cell as a post-hoc fix for
        quantile crossing.
    output_dir : str
        Directory to write the results CSV and summary report to.
    """

    feature_csv: str
    target_csv: str
    id_column: str = "instance_id"
    sample_id_column: Optional[str] = None
    timepoint_column: Optional[str] = None
    z_index_column: Optional[str] = None
    target_columns: List[str] = field(default_factory=lambda: list(TARGET_COLUMNS))
    exclude_feature_columns: List[str] = field(default_factory=list)
    group_by: str = "sample_id"
    n_splits: int = 5
    ridge_alpha: float = 1.0
    quantile_alpha: float = 0.0
    quantile_solver: str = "highs"
    quantile_train_subsample_size: Optional[int] = None
    quantile_train_subsample_seed: int = 0
    sort_quantiles: bool = True
    output_dir: str = "results/feature_to_mcherry"

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
        if (
            self.quantile_train_subsample_size is not None
            and self.quantile_train_subsample_size < 1
        ):
            raise ValueError("quantile_train_subsample_size must be at least 1")


def load_config(
    yaml_path: Optional[Path] = None,
    overrides: Optional[List[str]] = None,
) -> FeatureToMcherryConfig:
    """Build a :class:`FeatureToMcherryConfig` from a YAML file plus overrides.

    Parameters
    ----------
    yaml_path : Path, optional
        Path to a YAML file. If it has a top-level ``feature_to_mcherry:`` key, that
        subsection is used; otherwise the file's top level is used directly.
    overrides : list[str], optional
        Dot-notation overrides, e.g. ``["ridge_alpha=10", "group_by=timepoint"]``.

    Returns
    -------
    FeatureToMcherryConfig
    """
    schema = OmegaConf.structured(FeatureToMcherryConfig)

    if yaml_path is not None:
        file_config = OmegaConf.load(yaml_path)
        if isinstance(file_config, DictConfig) and "feature_to_mcherry" in file_config:
            file_config = file_config.feature_to_mcherry
        schema = OmegaConf.merge(schema, file_config)

    if overrides:
        schema = OmegaConf.merge(schema, OmegaConf.from_dotlist(list(overrides)))

    config_object = OmegaConf.to_object(schema)
    if not isinstance(config_object, FeatureToMcherryConfig):
        raise TypeError(
            f"Expected FeatureToMcherryConfig after merge, got {type(config_object)!r}"
        )
    return config_object
