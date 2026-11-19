"""Resolve per-experiment result paths; a missing file resolves to ``None``.

Three ``feature_to_mcherry`` layouts coexist, tried per field:

1. Old flat sibling dirs, ``<ftm>_<exp>[_scportrait[_injected]]/``, matched
   case-insensitively with hyphens folded to underscores (``..._ew2_2_scportrait``).
2. Nested, ``<ftm>/<Exp>/{baselines,data_quality,morphology_informativeness}/``,
   with single-experiment ``data_quality`` reports.
3. Top-level split, ``<ftm>/<exp>/baseline_ladder.csv`` plus separate
   ``morphology_informativeness/<exp>/`` and ``data_quality/<pair>/`` trees, where
   ``<pair>`` joins experiment names with ``_``; names never contain ``_``.

Only exact experiment names match under ``<ftm>/``, so variant dirs such as
``Ew2-2_alpha1p0`` are never mistaken for an experiment. Raw per-cell tables sit in
the ``data_root`` dir whose first whitespace-separated token is the experiment name.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import List, Optional

from .config import ReportFiguresConfig


@dataclass
class ExperimentPaths:
    """Resolved (or unresolved) result paths for one experiment. ``None`` fields mean
    "not found locally" — not an error."""

    baseline_ladder_csv: Optional[Path] = None
    scportrait_floor_metrics_csv: Optional[Path] = None
    scportrait_injected_floor_metrics_csv: Optional[Path] = None
    informativeness_dir: Optional[Path] = None
    floor_metrics_csv: Optional[Path] = None
    univariate_correlations_csv: Optional[Path] = None
    feature_importances_csv: Optional[Path] = None
    decomposition_dir: Optional[Path] = None
    data_quality_csv: Optional[Path] = None
    feature_dir: Optional[Path] = None
    instance_metrics_csv: Optional[Path] = None


def _first_existing(*candidates: Path) -> Optional[Path]:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _normalize_dir_name(name: str) -> str:
    """Lowercase and fold hyphens to underscores, as the old flat dirs are named."""
    return name.lower().replace("-", "_")


def find_sibling_dir_case_insensitive(parent: Path, target_name: str) -> Optional[Path]:
    """Find ``parent/<name>`` matching ``target_name`` up to case and
    hyphen/underscore folding (see :func:`_normalize_dir_name`)."""
    if not parent.is_dir():
        return None
    normalized_target = _normalize_dir_name(target_name)
    for candidate in parent.iterdir():
        if (
            candidate.is_dir()
            and _normalize_dir_name(candidate.name) == normalized_target
        ):
            return candidate
    return None


def _floor_metrics_in(run_dir: Optional[Path]) -> Optional[Path]:
    if run_dir is None:
        return None
    return _first_existing(run_dir / "morphology_informativeness" / "floor_metrics.csv")


def resolve_experiment_paths(exp: str, config: ReportFiguresConfig) -> ExperimentPaths:
    """Resolve every known result path for ``exp``, tolerating missing files."""
    ftm_dir = Path(config.feature_to_mcherry_dir)
    exp_dir = ftm_dir / exp

    old_flat_dir = find_sibling_dir_case_insensitive(
        ftm_dir.parent, f"{ftm_dir.name}_{exp}"
    )
    baseline_ladder_csv = _first_existing(
        exp_dir / "baseline_ladder.csv",  # convention 3
        exp_dir / "baselines" / "baseline_ladder.csv",  # convention 2
        *(
            [old_flat_dir / "baseline_ladder.csv"] if old_flat_dir else []
        ),  # convention 1
    )

    scportrait_dir = find_sibling_dir_case_insensitive(
        ftm_dir.parent, f"{ftm_dir.name}_{exp}_scportrait"
    )
    scportrait_floor_metrics_csv = _floor_metrics_in(scportrait_dir)
    scportrait_injected_floor_metrics_csv = _floor_metrics_in(
        find_sibling_dir_case_insensitive(
            ftm_dir.parent, f"{ftm_dir.name}_{exp}_scportrait_injected"
        )
    )
    decomposition_dir = (
        exp_dir / "decomposition" if (exp_dir / "decomposition").is_dir() else None
    )

    # Top-level (convention 3) wins over nested.
    informativeness_candidates = [
        Path(config.informativeness_dir) / exp,
        exp_dir / "morphology_informativeness",
    ]
    informativeness_dir = next(
        (c for c in informativeness_candidates if c.is_dir()), None
    )
    floor_metrics_csv = None
    univariate_correlations_csv = None
    feature_importances_csv = None
    if informativeness_dir is not None:
        floor_metrics_csv = _first_existing(informativeness_dir / "floor_metrics.csv")
        univariate_correlations_csv = _first_existing(
            informativeness_dir / "univariate_correlations.csv"
        )
        feature_importances_csv = _first_existing(
            informativeness_dir / "feature_importances.csv"
        )

    data_quality_csv = _first_existing(
        exp_dir / "data_quality" / "extreme_value_report.csv"
    )
    # Pair-combined fallback; callers filter by source.
    if data_quality_csv is None:
        data_quality_root = Path(config.data_quality_dir)
        if data_quality_root.is_dir():
            for subdir in sorted(data_quality_root.iterdir()):
                if not subdir.is_dir():
                    continue
                if exp in subdir.name.split("_"):
                    candidate = subdir / "extreme_value_report.csv"
                    if candidate.exists():
                        data_quality_csv = candidate
                        break

    feature_dir = None
    instance_metrics_csv = None
    data_root = Path(config.data_root)
    if data_root.is_dir():
        for subdir in sorted(data_root.iterdir()):
            if not subdir.is_dir():
                continue
            if subdir.name.split(" ")[0] == exp:
                candidate_features = (
                    subdir
                    / "inference_tracked"
                    / "cellpose_sam"
                    / "features_incarta"
                    / "split_data"
                )
                candidate_targets = (
                    subdir / "mcherry_metrics" / "cellpose_sam" / "instance_metrics.csv"
                )
                if candidate_features.is_dir():
                    feature_dir = candidate_features
                if candidate_targets.exists():
                    instance_metrics_csv = candidate_targets
                break

    return ExperimentPaths(
        baseline_ladder_csv=baseline_ladder_csv,
        scportrait_floor_metrics_csv=scportrait_floor_metrics_csv,
        scportrait_injected_floor_metrics_csv=scportrait_injected_floor_metrics_csv,
        informativeness_dir=informativeness_dir,
        floor_metrics_csv=floor_metrics_csv,
        univariate_correlations_csv=univariate_correlations_csv,
        feature_importances_csv=feature_importances_csv,
        decomposition_dir=decomposition_dir,
        data_quality_csv=data_quality_csv,
        feature_dir=feature_dir,
        instance_metrics_csv=instance_metrics_csv,
    )


def missing_fields(paths: ExperimentPaths) -> List[str]:
    """Field names currently unresolved (``None``), for manifest reporting."""
    return [f.name for f in fields(paths) if getattr(paths, f.name) is None]
