"""Tests for feature_to_mcherry.data_quality.join_parity and its CLI."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import List

import pandas as pd
import pytest

pytest.importorskip("pyarrow")

from src.feature_to_mcherry.data_quality import join_parity  # noqa: E402
from src.feature_to_mcherry.data_quality.join_parity import (  # noqa: E402
    compare_tables,
    count_raw_target_rows,
    load_feature_keys,
    load_target_keys,
    write_report,
)

WELLS = ("E07", "F08")


def _keys(wells=WELLS) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"sample_id": w, "timepoint": "101", "z_index": z, "cell_id": c}
            for w in wells
            for z in (1, 2)
            for c in (1, 2, 3)
        ]
    )


def _write_tables(
    tmp_path: Path,
    radiomics: pd.DataFrame = None,
    incarta: pd.DataFrame = None,
    target: pd.DataFrame = None,
) -> List[Path]:
    radiomics_dir, incarta_dir = tmp_path / "radiomics", tmp_path / "incarta"
    radiomics_dir.mkdir()
    incarta_dir.mkdir()
    radiomics = _keys() if radiomics is None else radiomics
    incarta = _keys() if incarta is None else incarta
    target = _keys() if target is None else target
    for well, rows in radiomics.groupby("sample_id"):
        rows.assign(original_shape2D_Elongation=0.5).to_parquet(
            radiomics_dir / f"{well}.parquet", index=False
        )
    for well, rows in incarta.groupby("sample_id"):
        rows.assign(area=10.0).to_csv(incarta_dir / f"{well}.csv", index=False)
    target_csv = tmp_path / "instance_metrics.csv"
    target.assign(percentile_75=1.0, percentile_90=2.0, percentile_95=3.0).to_csv(
        target_csv, index=False
    )
    return [radiomics_dir, incarta_dir, target_csv]


def _report(tmp_path: Path, **tables):
    radiomics_dir, incarta_dir, target_csv = _write_tables(tmp_path, **tables)
    return compare_tables(
        "test",
        {
            "radiomics": load_feature_keys(radiomics_dir, "*.parquet"),
            "incarta": load_feature_keys(incarta_dir, "*.csv"),
            "target": load_target_keys(target_csv),
        },
    )


def test_identical_tables_join_exactly(tmp_path: Path) -> None:
    report, gaps = _report(tmp_path)

    assert report.exact
    assert gaps.empty
    assert report.rows == {"radiomics": 12, "incarta": 12, "target": 12}
    assert all(pair.matched == 12 for pair in report.pairs)


def test_a_cell_missing_from_one_table_is_a_gap_in_two_pairs(tmp_path: Path) -> None:
    incarta = _keys().iloc[1:]

    report, gaps = _report(tmp_path, incarta=incarta)
    pairs = {(pair.left, pair.right): pair for pair in report.pairs}

    assert not report.exact
    assert pairs[("radiomics", "incarta")].left_only == 1
    assert pairs[("incarta", "target")].right_only == 1
    assert pairs[("radiomics", "target")].exact
    assert set(gaps["only_in"]) == {"radiomics", "target"}


def test_an_extra_target_row_names_its_well_and_timepoint(tmp_path: Path) -> None:
    extra = pd.DataFrame(
        [{"sample_id": "F08", "timepoint": "101", "z_index": 2, "cell_id": 99}]
    )

    report, _ = _report(tmp_path, target=pd.concat([_keys(), extra]))

    assert not report.exact
    top = report.gap_by_stack[0]
    assert (top["only_in"], top["sample_id"], top["timepoint"]) == (
        "target",
        "F08",
        "101",
    )


def test_a_repeated_key_is_reported_and_not_exact(tmp_path: Path) -> None:
    report, _ = _report(tmp_path, incarta=pd.concat([_keys(), _keys().iloc[:1]]))

    assert report.duplicates["incarta"] == 1
    assert not report.exact


def test_an_empty_table_is_not_an_exact_join() -> None:
    empty = _keys().iloc[0:0]

    report, _ = compare_tables("t", {"a": empty, "b": empty})

    assert report.pairs[0].exact
    assert not report.exact


def test_null_keys_are_counted_and_not_exact() -> None:
    nulls = _keys().assign(cell_id=lambda df: df["cell_id"].where(df["cell_id"] > 1))

    report, _ = compare_tables("t", {"a": nulls, "b": nulls})

    assert report.null_keys == {"a": 4, "b": 4}
    assert not report.exact


def test_a_float_z_index_does_not_match_an_integer_one_like_the_analysis_join() -> None:
    as_float = _keys().astype({"z_index": float})

    report, _ = compare_tables("t", {"a": _keys(), "b": as_float})

    assert not report.exact
    assert report.n_gap_rows == 24


def test_the_raw_target_row_count_is_reported(tmp_path: Path) -> None:
    target = _keys().assign(percentile_75=1.0, percentile_90=1.0, percentile_95=1.0)
    target.loc[0, "percentile_75"] = float("nan")
    csv_path = tmp_path / "instance_metrics.csv"
    target.to_csv(csv_path, index=False)

    assert count_raw_target_rows(csv_path) == 12
    assert len(load_target_keys(csv_path)) == 11


def test_a_stale_gaps_file_is_removed_by_an_exact_run(tmp_path: Path) -> None:
    report, gaps = compare_tables("t", {"a": _keys(), "b": _keys()})
    report_path = tmp_path / "t.json"
    stale = report_path.with_suffix(".gaps.csv")
    stale.write_text("old gaps")

    write_report(report, gaps, report_path)

    assert not stale.exists()


def _run_cli(monkeypatch, *argv) -> int:
    script = Path(__file__).resolve().parents[3] / "scripts" / "run_join_parity.py"
    spec = importlib.util.spec_from_file_location("_run_join_parity", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr("sys.argv", ["run_join_parity.py", *map(str, argv)])
    return module.main()


def _cli_args(tmp_path: Path, paths: List[Path]) -> List[object]:
    return [
        "--label",
        "t",
        "--radiomics-dir",
        paths[0],
        "--incarta-dir",
        paths[1],
        "--target-csv",
        paths[2],
        "--report",
        tmp_path / "out" / "t.json",
    ]


def test_cli_exit_codes_and_report_files(tmp_path: Path, monkeypatch) -> None:
    exact = tmp_path / "exact"
    exact.mkdir()
    args = _cli_args(exact, _write_tables(exact))
    assert _run_cli(monkeypatch, *args) == 0
    assert json.loads((exact / "out" / "t.json").read_text())["exact"] is True
    assert not (exact / "out" / "t.gaps.csv").exists()

    gap = tmp_path / "gap"
    gap.mkdir()
    args = _cli_args(gap, _write_tables(gap, incarta=_keys().iloc[1:]))
    assert _run_cli(monkeypatch, *args) == 1
    assert (gap / "out" / "t.gaps.csv").exists()

    args[args.index("--radiomics-dir") + 1] = tmp_path / "typo"
    assert _run_cli(monkeypatch, *args) == 2


def test_cli_exits_1_on_a_repeated_key_and_2_on_any_other_failure(
    tmp_path: Path, monkeypatch
) -> None:
    repeated = tmp_path / "repeated"
    repeated.mkdir()
    paths = _write_tables(repeated, incarta=pd.concat([_keys(), _keys().iloc[:1]]))
    assert _run_cli(monkeypatch, *_cli_args(repeated, paths)) == 1

    crash = tmp_path / "crash"
    crash.mkdir()
    args = _cli_args(crash, _write_tables(crash))
    monkeypatch.setattr(
        "src.feature_to_mcherry.data_quality.join_parity.compare_tables",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert _run_cli(monkeypatch, *args) == 2


def test_truncated_gaps_are_flagged_and_every_pair_keeps_rows(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(join_parity, "MAX_GAP_ROWS", 4)
    short = _keys().iloc[6:]

    report, gaps = compare_tables("t", {"a": _keys(), "b": short, "c": _keys()})
    gap_path = write_report(report, gaps, tmp_path / "t.json")

    assert report.gaps_truncated
    assert report.n_gap_rows == 12
    written = pd.read_csv(gap_path)
    assert set(written["pair"]) == {"a|b", "b|c"}
    assert len(written) <= 4


def test_cli_rejects_a_label_that_is_not_a_plain_file_name(
    tmp_path: Path, monkeypatch
) -> None:
    args = _cli_args(tmp_path, _write_tables(tmp_path))
    args[args.index("--label") + 1] = "../x"
    with pytest.raises(SystemExit) as raised:
        _run_cli(monkeypatch, *args)
    assert raised.value.code == 2


def test_one_table_is_not_a_comparison() -> None:
    with pytest.raises(ValueError, match="at least two tables"):
        compare_tables("t", {"a": _keys()})


def test_a_long_breakdown_is_flagged_and_capped(monkeypatch) -> None:
    monkeypatch.setattr(join_parity, "_BREAKDOWN_ROWS", 2)
    many_wells = ("E07", "F08", "C03", "D04")

    report, _ = compare_tables("t", {"a": _keys(many_wells), "b": _keys(WELLS)})

    assert report.n_breakdown_rows == 2
    assert not report.breakdown_truncated

    report, _ = compare_tables("t", {"a": _keys(many_wells), "b": _keys(("E07",))})

    assert report.n_breakdown_rows == 3
    assert report.breakdown_truncated
    assert len(report.gap_by_stack) == 2
    assert [row["sample_id"] for row in report.gap_by_stack] == ["C03", "D04"]


def test_the_largest_stacks_come_first_in_the_breakdown() -> None:
    partial_c03 = _keys(("C03",)).iloc[:2]
    right = pd.concat([_keys(WELLS), partial_c03])

    report, _ = compare_tables(
        "t", {"a": _keys(("E07", "F08", "C03", "D04")), "b": right}
    )

    assert [(row["sample_id"], row["n_cells"]) for row in report.gap_by_stack] == [
        ("D04", 6),
        ("C03", 4),
    ]


def test_a_per_pair_cut_is_flagged_even_when_the_total_is_under_the_cap(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(join_parity, "MAX_GAP_ROWS", 6)
    small = _keys(("E07",))

    report, gaps = compare_tables("t", {"a": _keys(), "b": small, "c": small})
    gap_path = write_report(report, gaps, tmp_path / "t.json")

    assert report.n_gap_rows == 12 > 6
    assert report.gaps_truncated
    assert report.gaps_per_pair_cap == 3
    written = pd.read_csv(gap_path)
    assert written["pair"].value_counts().to_dict() == {"a|b": 3, "a|c": 3}

    report, _ = compare_tables("t", {"a": _keys(), "b": _keys(), "c": _keys()})
    assert not report.gaps_truncated


def test_cli_exits_2_when_the_report_path_is_unwritable(
    tmp_path: Path, monkeypatch
) -> None:
    args = _cli_args(tmp_path, _write_tables(tmp_path))
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a directory")
    args[args.index("--report") + 1] = blocker / "t.json"

    assert _run_cli(monkeypatch, *args) == 2


def test_cli_takes_the_feature_id_column_and_warns_on_a_long_breakdown(
    tmp_path: Path, monkeypatch, caplog: pytest.LogCaptureFixture
) -> None:
    renamed = _keys().rename(columns={"cell_id": "label"})
    paths = _write_tables(tmp_path, radiomics=renamed, incarta=renamed)
    args = _cli_args(tmp_path, paths)
    assert _run_cli(monkeypatch, *args, "--id-column", "label") == 0

    gap = tmp_path / "gap"
    gap.mkdir()
    gap_args = _cli_args(gap, _write_tables(gap, target=_keys().iloc[6:]))
    monkeypatch.setattr(join_parity, "_BREAKDOWN_ROWS", 1)
    with caplog.at_level("WARNING"):
        assert _run_cli(monkeypatch, *gap_args) == 1
    assert "breakdown rows" in caplog.text
