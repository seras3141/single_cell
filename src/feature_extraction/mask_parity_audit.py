"""Audit per-well radiomics tables against the masks they were extracted from.

Masks are the top-level ``*_<well>_*_pred_mask.tif`` files of the mask folder; wells
are discovered from the masks and from the feature and coverage tables.

Per well, three checks, each failure named and never averaged away:

1. coverage completeness: every mask has one ``ok``/``empty`` coverage row;
2. mask parity: the ``cell_id`` set of each mask's features equals its labels with at
   least ``min_pixels`` pixels, allowing the number of rejected labels that coverage
   reports (``n_rejected_labels``; which labels were rejected cannot be verified);
3. table integrity: no duplicate cells, no feature row without a coverage row.
"""

from __future__ import annotations

import glob
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.feature_extraction.feature_extraction_pipeline import (
    COVERAGE_SUBDIR,
    file_stack_key,
)
from src.feature_extraction.feature_extractor_pyradiomics import _label_counts
from src.utils.config_schemas import PyradiomicsConfig
from src.utils.file_utils import ConfigurableFileHandler
from src.utils.image_utils import load_labels

logger = logging.getLogger(__name__)

AUDIT_FILENAME = "mask_parity_audit.json"
MASK_GLOB = "*_pred_mask.tif"
_MAX_EXAMPLES = 5
_CELL_KEY = ["sample_id", "timepoint", "z_index"]
_COVERAGE_COLUMNS = _CELL_KEY + [
    "mask_filename",
    "status",
    "n_cells",
    "n_skipped_small",
    "n_label_errors",
]


@dataclass
class WellAudit:
    well: str
    n_masks: int = 0
    n_feature_rows: int = 0
    n_rejected_labels: int = 0
    status_counts: Dict[str, int] = field(default_factory=dict)
    failures: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures


@dataclass
class TreeAudit:
    features_dir: str
    mask_dir: str
    min_pixels: int
    wells: List[WellAudit] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.wells) and all(well.passed for well in self.wells)

    @property
    def n_feature_rows(self) -> int:
        return sum(well.n_feature_rows for well in self.wells)

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["passed"] = self.passed
        data["n_feature_rows"] = self.n_feature_rows
        return data


_MaskResult = Tuple[str, Optional[List[int]], int, str]


def _mask_labels(args: Tuple[Path, int]) -> _MaskResult:
    """Labels of at least ``min_pixels`` pixels, the smaller-label count, any error."""
    path, min_pixels = args
    try:
        mask = np.asarray(load_labels(path))
        if not np.issubdtype(mask.dtype, np.integer) or mask.ndim != 2:
            raise ValueError(
                f"expected a 2D integer mask, got {mask.dtype} {mask.shape}"
            )
        labels, counts = _label_counts(mask)
    except Exception as exc:  # a corrupt mask must not abort
        return path.name, None, 0, f"{type(exc).__name__}: {exc}"
    kept = labels[counts >= min_pixels]
    return path.name, [int(label) for label in kept], int(labels.size - kept.size), ""


def _read_masks(
    paths: Sequence[Path], min_pixels: int, n_jobs: int
) -> Tuple[Dict[str, Tuple[List[int], int]], Dict[str, str]]:
    tasks = [(path, min_pixels) for path in paths]
    if n_jobs > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            results = list(pool.map(_mask_labels, tasks, chunksize=8))
    else:
        results = [_mask_labels(task) for task in tasks]
    labels = {
        name: (kept, n_small) for name, kept, n_small, _ in results if kept is not None
    }
    errors = {name: error for name, kept, _, error in results if kept is None}
    return labels, errors


def _stack_keys(table: pd.DataFrame) -> List[Tuple[str, str, int]]:
    return list(
        zip(
            table["sample_id"].astype(str),
            table["timepoint"].astype(str),
            table["z_index"].astype(int),
        )
    )


def _key_problems(table: pd.DataFrame, well: str, name: str) -> List[str]:
    """Rows whose key cannot belong to this well's integer-z stacks."""
    z_index = pd.to_numeric(table["z_index"], errors="coerce")
    integral = z_index.notna() & (z_index % 1 == 0)
    problems = []
    if not integral.all():
        problems.append(
            _named(
                f"{name} rows with a non-integral z_index",
                table.loc[~integral, "z_index"].tolist(),
            )
        )
    foreign = table.loc[table["sample_id"].astype(str) != well, "sample_id"]
    if len(foreign):
        problems.append(
            _named(
                f"{name} rows whose sample_id is not {well}",
                sorted(set(foreign.astype(str))),
            )
        )
    return problems


def _require_min_pixels(min_pixels: int) -> None:
    if min_pixels < 1:
        raise ValueError(f"min_pixels must be >= 1, got {min_pixels}")


def _named(problem: str, examples: Sequence[object]) -> str:
    listed = ", ".join(str(example) for example in list(examples)[:_MAX_EXAMPLES])
    more = "" if len(examples) <= _MAX_EXAMPLES else f", ... ({len(examples)} total)"
    return f"{problem}: {listed}{more}"


def audit_well(
    features_dir: Path,
    mask_dir: Path,
    well: str,
    min_pixels: int,
    n_jobs: int = 1,
) -> WellAudit:
    _require_min_pixels(min_pixels)
    audit = WellAudit(well=well)
    features_path = features_dir / f"{well}.parquet"
    coverage_path = features_dir / COVERAGE_SUBDIR / f"{well}.parquet"
    for path in (features_path, coverage_path):
        if not path.exists():
            audit.failures.append(f"missing file {path}")
    mask_paths = sorted(mask_dir.glob(f"*_{glob.escape(well)}_*_pred_mask.tif"))
    audit.n_masks = len(mask_paths)
    masks_on_disk = {path.name for path in mask_paths}
    if not mask_paths:
        audit.failures.append(f"no masks for well {well} in {mask_dir}")
    if audit.failures:
        return audit

    try:
        features = pd.read_parquet(features_path)
        coverage = pd.read_parquet(coverage_path)
    except Exception as exc:  # a truncated table must not abort
        audit.failures.append(f"unreadable table ({type(exc).__name__}: {exc})")
        return audit
    audit.n_feature_rows = len(features)

    missing_columns = [c for c in _COVERAGE_COLUMNS if c not in coverage.columns]
    if missing_columns:
        audit.failures.append(f"coverage table lacks columns {missing_columns}")
        return audit
    missing_columns = [c for c in _CELL_KEY + ["cell_id"] if c not in features.columns]
    if missing_columns:
        audit.failures.append(f"feature table lacks columns {missing_columns}")
        return audit
    audit.status_counts = {
        str(status): int(count)
        for status, count in coverage["status"].value_counts().items()
    }
    audit.n_rejected_labels = int(coverage["n_label_errors"].sum())
    audit.failures += _key_problems(coverage, well, "coverage")
    audit.failures += _key_problems(features, well, "feature")
    if audit.failures:
        return audit

    unsegmented = coverage[coverage["status"] == "unsegmented"]
    stray = unsegmented[unsegmented["z_index"] != 0]
    if len(stray):
        audit.failures.append(
            _named("unsegmented rows outside z0", stray["z_index"].tolist())
        )
    count_columns = ["n_cells", "n_skipped_small", "n_label_errors"]
    counted = unsegmented[(unsegmented[count_columns] != 0).any(axis=1)]
    if len(counted):
        audit.failures.append(
            _named("unsegmented rows with nonzero counts", _stack_keys(counted))
        )
    named = unsegmented[unsegmented["mask_filename"].fillna("") != ""]
    if len(named):
        audit.failures.append(
            _named(
                "unsegmented rows that name a mask", sorted(set(named["mask_filename"]))
            )
        )
    other = coverage[~coverage["status"].isin(["ok", "empty", "unsegmented"])]
    if len(other):
        audit.failures.append(
            _named(
                "coverage rows with an unexpected status", sorted(set(other["status"]))
            )
        )

    with_mask = coverage[coverage["status"].isin(["ok", "empty"])]
    duplicated = with_mask[with_mask["mask_filename"].duplicated(keep=False)]
    if len(duplicated):
        audit.failures.append(
            _named(
                "masks with more than one coverage row",
                sorted(set(duplicated["mask_filename"])),
            )
        )
    shared_stacks = coverage[coverage.duplicated(_CELL_KEY, keep=False)]
    if len(shared_stacks):
        audit.failures.append(
            _named(
                "stacks with more than one coverage row",
                sorted(set(_stack_keys(shared_stacks))),
            )
        )
    handler = ConfigurableFileHandler()
    disagreeing = [
        row.mask_filename
        for row in with_mask.itertuples(index=False)
        if file_stack_key(handler, row.mask_filename)
        != (str(row.sample_id), str(row.timepoint), int(row.z_index))
    ]
    if disagreeing:
        audit.failures.append(
            _named("coverage key disagrees with the mask filename", disagreeing)
        )
    covered = set(with_mask["mask_filename"])
    if masks_on_disk - covered:
        audit.failures.append(
            _named("masks without a coverage row", sorted(masks_on_disk - covered))
        )
    if covered - masks_on_disk:
        audit.failures.append(
            _named(
                "coverage rows for masks not on disk", sorted(covered - masks_on_disk)
            )
        )

    cell_columns = _CELL_KEY + ["cell_id"]
    duplicate_cells = features[features.duplicated(cell_columns, keep=False)]
    if len(duplicate_cells):
        audit.failures.append(
            _named(
                "duplicate (sample_id, timepoint, z_index, cell_id) rows",
                duplicate_cells[cell_columns]
                .drop_duplicates()
                .to_records(index=False)
                .tolist(),
            )
        )
    if int(coverage["n_cells"].sum()) != len(features):
        audit.failures.append(
            f"coverage n_cells sums to {int(coverage['n_cells'].sum())}, "
            f"feature table has {len(features)} rows"
        )

    by_stack = dict(zip(_stack_keys(with_mask), with_mask["mask_filename"]))
    feature_stacks = _stack_keys(features)
    row_mask = [by_stack.get(stack) for stack in feature_stacks]
    orphans = {stack for stack, name in zip(feature_stacks, row_mask) if name is None}
    if orphans:
        audit.failures.append(
            _named("feature rows with no coverage row", sorted(orphans))
        )
    cell_ids = pd.to_numeric(features["cell_id"], errors="coerce")
    integral = (cell_ids.notna() & (cell_ids % 1 == 0)).to_numpy()
    if not integral.all():
        audit.failures.append(
            _named(
                "non-integral cell_id values",
                features.loc[~integral, "cell_id"].tolist(),
            )
        )
    cells_by_mask: Dict[str, set] = {}
    for name, cell_id, is_integral in zip(row_mask, cell_ids, integral):
        if name is not None and is_integral:
            cells_by_mask.setdefault(name, set()).add(int(cell_id))

    labels, unreadable = _read_masks(
        [path for path in mask_paths if path.name in covered], min_pixels, n_jobs
    )
    if unreadable:
        audit.failures.append(
            _named(
                "unreadable masks",
                [f"{name} ({error})" for name, error in sorted(unreadable.items())],
            )
        )
    problems: Dict[str, List[str]] = {}

    def flag(problem: str, mask_name: str) -> None:
        problems.setdefault(problem, []).append(mask_name)

    for row in with_mask.itertuples(index=False):
        if row.mask_filename not in labels:
            continue
        kept, n_small = labels[row.mask_filename]
        eligible = set(kept)
        found = cells_by_mask.get(row.mask_filename, set())
        if found - eligible:
            flag("cells not among the mask's labels", row.mask_filename)
        if len(eligible - found) != row.n_label_errors:
            flag("labels without a row beyond the rejected ones", row.mask_filename)
        if row.n_skipped_small != n_small:
            flag("n_skipped_small differs from the mask", row.mask_filename)
        if row.n_cells != len(found):
            flag("n_cells differs from the feature rows", row.mask_filename)
        if (row.status == "empty") != (len(found) == 0):
            flag("status disagrees with the feature rows", row.mask_filename)
    for problem, names in problems.items():
        audit.failures.append(_named(problem, names))
    return audit


def audit_tree(
    features_dir: Path,
    mask_dir: Path,
    wells: Optional[Sequence[str]] = None,
    min_pixels: Optional[int] = None,
    n_jobs: int = 1,
) -> TreeAudit:
    features_dir, mask_dir = Path(features_dir), Path(mask_dir)
    for directory in (features_dir, mask_dir):
        if not directory.is_dir():
            raise FileNotFoundError(f"Not a directory: {directory}")
    if min_pixels is None:
        min_pixels = PyradiomicsConfig().min_pixels
    _require_min_pixels(min_pixels)
    if wells is None:
        handler = ConfigurableFileHandler()
        found = {file_stack_key(handler, p.name)[0] for p in mask_dir.glob(MASK_GLOB)}
        found.discard("")
        found |= {path.stem for path in features_dir.glob("*.parquet")}
        found |= {
            path.stem for path in (features_dir / COVERAGE_SUBDIR).glob("*.parquet")
        }
        wells = sorted(found)
    if not wells:
        raise ValueError(f"No well tables in {features_dir}")
    tree = TreeAudit(str(features_dir), str(mask_dir), min_pixels)
    for well in wells:
        try:
            result = audit_well(features_dir, mask_dir, well, min_pixels, n_jobs)
        except Exception as exc:  # a bad well must not abort
            result = WellAudit(
                well=well, failures=[f"audit crashed ({type(exc).__name__}: {exc})"]
            )
        logger.info(
            "%s: %s (%d masks, %d rows)",
            well,
            "pass" if result.passed else "FAIL",
            result.n_masks,
            result.n_feature_rows,
        )
        tree.wells.append(result)
    return tree


def write_audit(audit: TreeAudit, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit.to_dict(), indent=2))
    return path
