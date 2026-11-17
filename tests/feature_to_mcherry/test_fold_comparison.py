from pathlib import Path

import pandas as pd
import pytest

from src.feature_to_mcherry.evaluation.fold_comparison import (
    FOLD_FILENAME,
    compare_backends_under_root,
    compare_fold_assignments,
)


def _write(directory: Path, folds: dict) -> Path:
    directory.mkdir(parents=True)
    pd.DataFrame({"group_id": list(folds), "fold": list(folds.values())}).to_csv(
        directory / FOLD_FILENAME, index=False
    )
    return directory


def test_identical_folds_report_nothing(tmp_path):
    first = _write(tmp_path / "a", {"E07": 0, "F09": 1})
    second = _write(tmp_path / "b", {"F09": 1, "E07": 0})
    assert compare_fold_assignments(first, second) == []


def test_a_changed_fold_is_reported(tmp_path):
    first = _write(tmp_path / "a", {"E07": 0, "F09": 1})
    second = _write(tmp_path / "b", {"E07": 0, "F09": 3})
    problems = compare_fold_assignments(first, second)
    assert len(problems) == 1 and "F09" in problems[0] and "1 vs 3" in problems[0]


def test_a_well_missing_from_one_side_names_that_side(tmp_path):
    first = _write(tmp_path / "a", {"E07": 0, "G09": 2})
    second = _write(tmp_path / "b", {"E07": 0})
    problems = compare_fold_assignments(first, second)
    assert len(problems) == 1
    assert str(first) in problems[0] and "G09" in problems[0]


def test_a_well_in_two_folds_is_a_problem_even_if_the_other_side_matches(tmp_path):
    first = tmp_path / "a"
    first.mkdir()
    pd.DataFrame({"group_id": ["E07", "E07"], "fold": [0, 1]}).to_csv(
        first / FOLD_FILENAME, index=False
    )
    second = _write(tmp_path / "b", {"E07": 0})
    problems = compare_fold_assignments(first, second)
    assert len(problems) == 1 and "more than once" in problems[0]


def test_a_table_without_the_fold_column_is_rejected(tmp_path):
    first = _write(tmp_path / "a", {"E07": 0})
    second = tmp_path / "b"
    second.mkdir()
    pd.DataFrame({"group_id": ["E07"]}).to_csv(second / FOLD_FILENAME, index=False)
    with pytest.raises(ValueError, match="fold"):
        compare_fold_assignments(first, second)


def _run(root: Path, label: str, backend: str, folds: dict) -> None:
    _write(root / label / backend / "baselines", folds)


def test_backends_under_a_root_are_paired_per_run(tmp_path):
    _run(tmp_path, "gated/sharpness/HD1509", "radiomics", {"E07": 0, "F09": 1})
    _run(tmp_path, "gated/sharpness/HD1509", "incarta", {"E07": 0, "F09": 1})
    _run(tmp_path, "gated/sharpness/HD1883", "radiomics", {"E07": 0, "F09": 1})
    _run(tmp_path, "gated/sharpness/HD1883", "incarta", {"E07": 1, "F09": 0})
    results = compare_backends_under_root(tmp_path)
    assert results["gated/sharpness/HD1509"] == []
    assert len(results["gated/sharpness/HD1883"]) == 2


def test_a_run_with_one_backend_only_is_reported(tmp_path):
    _run(tmp_path, "gated/sharpness/HD1509", "incarta", {"E07": 0})
    results = compare_backends_under_root(tmp_path)
    problems = results["gated/sharpness/HD1509"]
    assert len(problems) == 1 and "radiomics" in problems[0]


def test_a_root_without_runs_gives_no_results(tmp_path):
    assert compare_backends_under_root(tmp_path) == {}
