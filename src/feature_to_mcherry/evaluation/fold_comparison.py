"""Check that two backends' runs assigned every well to the same CV fold."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

FOLD_FILENAME = "fold_assignments.csv"
REQUIRED_COLUMNS = ("group_id", "fold")


def _read_folds(directory: Path) -> pd.DataFrame:
    table = pd.read_csv(Path(directory) / FOLD_FILENAME, dtype={"group_id": str})
    missing = [column for column in REQUIRED_COLUMNS if column not in table.columns]
    if missing:
        raise ValueError(f"{directory / FOLD_FILENAME} lacks columns {missing}")
    return table


def compare_fold_assignments(first_dir: Path, second_dir: Path) -> List[str]:
    """Differences between two runs' fold tables (empty if identical)."""
    problems: List[str] = []
    tables = {}
    for directory in (first_dir, second_dir):
        table = _read_folds(directory)
        repeated = sorted(set(table.loc[table["group_id"].duplicated(), "group_id"]))
        if repeated:
            problems.append(f"wells listed more than once in {directory}: {repeated}")
        tables[directory] = table
    if problems:
        return problems

    first, second = tables[first_dir], tables[second_dir]
    only_first = sorted(set(first["group_id"]) - set(second["group_id"]))
    only_second = sorted(set(second["group_id"]) - set(first["group_id"]))
    if only_first:
        problems.append(f"wells only in {first_dir}: {only_first}")
    if only_second:
        problems.append(f"wells only in {second_dir}: {only_second}")
    merged = first.merge(second, on="group_id", suffixes=("_first", "_second"))
    differing = merged[merged["fold_first"] != merged["fold_second"]]
    for row in differing.itertuples():
        problems.append(
            f"well {row.group_id}: fold {row.fold_first} vs {row.fold_second}"
        )
    return problems


def compare_backends_under_root(
    root: Path, first_backend: str = "radiomics", second_backend: str = "incarta"
) -> Dict[str, List[str]]:
    """Fold differences per ``<gate>/<tree>/<culture>`` run found under ``root``.

    Runs are laid out as ``root/<gate>/<tree>/<culture>/<backend>/baselines``. A run
    that has only one backend, or no fold table for one of them, is reported as a
    problem. The result is empty when no run is found.
    """
    labels = set()
    for backend in (first_backend, second_backend):
        for baselines in Path(root).glob(f"*/*/*/{backend}/baselines"):
            labels.add(baselines.parents[1].relative_to(root).as_posix())

    results: Dict[str, List[str]] = {}
    for label in sorted(labels):
        directories = [
            Path(root) / label / backend / "baselines"
            for backend in (first_backend, second_backend)
        ]
        absent = [d for d in directories if not (d / FOLD_FILENAME).exists()]
        if absent:
            results[label] = [f"no {FOLD_FILENAME} in {d}" for d in absent]
        else:
            results[label] = compare_fold_assignments(*directories)
    return results
