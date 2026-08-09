"""Resolve per-experiment result paths across this repo's (inconsistent) naming
conventions, without raising on a missing experiment — callers report a figure as
blocked/partial instead of crashing.

At least three ``feature_to_mcherry``-family conventions coexist across this
project's history (confirmed by direct inspection of the HPC filesystem,
2026-07-28 — do not assume any one of these is "the" convention going forward):

1. **Old flat, lowercase** (HD1509/SA110's first baseline run, 2026-07-24):
   ``results/feature_to_mcherry_<exp-lowercase>[_scportrait]/baseline_ladder.csv``,
   a sibling of ``feature_to_mcherry_dir`` itself. Case matters here — the
   directory is lowercase (``feature_to_mcherry_hd1509``) even though the
   experiment's canonical name is ``HD1509`` — so matching must be
   case-insensitive, not a naive ``f"{exp}"`` substitution. For hyphenated names
   the hyphen also folds to an underscore (``feature_to_mcherry_ew2_2_scportrait``,
   not ``..._ew2-2_...``) — confirmed 2026-07-28 when Ew2-2's scPortrait dir
   didn't resolve under a case-only fold.
2. **Nested-per-experiment** (current HPC convention as of 2026-07-28, all five
   experiments): ``results/feature_to_mcherry/<Exp>/{baselines,data_quality,
   morphology_informativeness}/`` — note ``baseline_ladder.csv`` lives under a
   ``baselines/`` subdirectory here, and ``data_quality``/``morphology_informativeness``
   are nested *inside* the per-experiment directory rather than being separate
   top-level trees. ``data_quality``'s ``extreme_value_report.csv`` here is
   single-experiment (one ``source`` value), not the pair-combined file convention
   below.
3. **Local top-level split** (what this machine's ``results/`` currently has for
   Ew2-1/Ew2-2): ``feature_to_mcherry/<exp>/baseline_ladder.csv`` directly (no
   ``baselines/`` nesting), plus separate top-level
   ``morphology_informativeness/<exp>/`` and ``data_quality/<pair>/`` trees.

All three are tried, in the order above is not implied — see each field's specific
fallback chain below; whichever exists wins.

``results/feature_to_mcherry/`` locally also contains alpha-sweep variant
subdirectories (e.g. ``Ew2-2_alpha1p0``) that must not be mistaken for an experiment —
resolution only ever matches the exact experiment name for that tier.

``results/data_quality/`` (convention 3) stores one combined report per
*experiment-pair* directory (e.g. ``Ew2-1_Ew2-2/``), not per single experiment; the
directory name is the underscore-joined list of experiments it covers (experiment
names use hyphens, not underscores, internally, so splitting on ``"_"`` is
unambiguous).

Raw per-cell tables (incarta ``split_data/`` and ``instance_metrics.csv``) live under
``config.data_root``, in a directory whose *first whitespace-separated token* is the
experiment name (e.g. ``"Ew2-1 MF5V1 0-72h 06-03-26"``) — confirmed present locally
only for Ew2-1/Ew2-2 today; unresolved for the other three experiments until/unless
their raw data is also copied locally (Phase 0 only syncs small ``results/`` files,
not raw per-cell tables).
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
    scportrait_baseline_ladder_csv: Optional[Path] = None
    scportrait_floor_metrics_csv: Optional[Path] = None
    informativeness_dir: Optional[Path] = None
    floor_metrics_csv: Optional[Path] = None
    univariate_correlations_csv: Optional[Path] = None
    feature_importances_csv: Optional[Path] = None
    data_quality_csv: Optional[Path] = None
    feature_dir: Optional[Path] = None
    instance_metrics_csv: Optional[Path] = None


def _first_existing(*candidates: Path) -> Optional[Path]:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _normalize_dir_name(name: str) -> str:
    """Lowercase and fold hyphens to underscores for loose directory-name matching.

    The old flat convention lowercases (``feature_to_mcherry_hd1509``) *and*
    replaces hyphens with underscores for hyphenated experiment names
    (``feature_to_mcherry_ew2_2_scportrait``, not ``..._ew2-2_...``) — a naive
    case-only fold misses the latter.
    """
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


def resolve_experiment_paths(exp: str, config: ReportFiguresConfig) -> ExperimentPaths:
    """Resolve every known result path for ``exp``, tolerating missing files."""
    ftm_dir = Path(config.feature_to_mcherry_dir)
    exp_dir = ftm_dir / exp  # feature_to_mcherry/<Exp>/ -- conventions 2 and 3

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
    scportrait_baseline_ladder_csv = (
        _first_existing(scportrait_dir / "baseline_ladder.csv")
        if scportrait_dir
        else None
    )
    scportrait_floor_metrics_csv = (
        _first_existing(
            scportrait_dir / "morphology_informativeness" / "floor_metrics.csv"
        )
        if scportrait_dir
        else None
    )

    # informativeness: convention 3 (top-level) first, else convention 2 (nested).
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

    # data_quality: convention 2 (nested, single-experiment) first, else
    # convention 3 (top-level, pair-combined -- filter rows by `source` downstream).
    data_quality_csv = _first_existing(
        exp_dir / "data_quality" / "extreme_value_report.csv"
    )
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
        scportrait_baseline_ladder_csv=scportrait_baseline_ladder_csv,
        scportrait_floor_metrics_csv=scportrait_floor_metrics_csv,
        informativeness_dir=informativeness_dir,
        floor_metrics_csv=floor_metrics_csv,
        univariate_correlations_csv=univariate_correlations_csv,
        feature_importances_csv=feature_importances_csv,
        data_quality_csv=data_quality_csv,
        feature_dir=feature_dir,
        instance_metrics_csv=instance_metrics_csv,
    )


def missing_fields(paths: ExperimentPaths) -> List[str]:
    """Field names currently unresolved (``None``), for manifest reporting."""
    return [f.name for f in fields(paths) if getattr(paths, f.name) is None]
