"""Mask-parity audit of per-well radiomics tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

pytest.importorskip("pyarrow")

from src.feature_extraction.feature_extraction_pipeline import (  # noqa: E402
    COVERAGE_SUBDIR,
)
from src.feature_extraction.mask_parity_audit import (  # noqa: E402
    audit_tree,
    audit_well,
    write_audit,
)

MIN_PIXELS = 20
WELL = "E07"


def _mask() -> np.ndarray:
    """Labels 1 and 7 (100 px each) and label 9 (4 px, below MIN_PIXELS)."""
    mask = np.zeros((40, 40), np.uint16)
    mask[2:12, 2:12] = 1
    mask[20:30, 20:30] = 7
    mask[35:37, 35:37] = 9
    return mask


def _stack_names(timepoint: str, z: int) -> str:
    return f"pMF5V1_{WELL}_t{timepoint}_z{z}_pred_mask.tif"


@pytest.fixture
def tree(tmp_path: Path):
    """Two segmented stacks and one unsegmented z0 image, all consistent."""
    masks, features = tmp_path / "masks", tmp_path / "features"
    (features / COVERAGE_SUBDIR).mkdir(parents=True)
    masks.mkdir()
    feature_rows, coverage_rows = [], []
    for timepoint, z in (("1", 1), ("1", 2)):
        name = _stack_names(timepoint, z)
        tifffile.imwrite(masks / name, _mask())
        for cell_id in (1, 7):
            feature_rows.append(
                {
                    "sample_id": WELL,
                    "timepoint": timepoint,
                    "z_index": z,
                    "cell_id": cell_id,
                    "original_shape2D_Elongation": 0.5,
                }
            )
        coverage_rows.append(
            {
                "sample_id": WELL,
                "timepoint": timepoint,
                "z_index": z,
                "mask_filename": name,
                "status": "ok",
                "n_cells": 2,
                "n_skipped_small": 1,
                "n_label_errors": 0,
            }
        )
    coverage_rows.append(
        {
            "sample_id": WELL,
            "timepoint": "1",
            "z_index": 0,
            "mask_filename": "",
            "status": "unsegmented",
            "n_cells": 0,
            "n_skipped_small": 0,
            "n_label_errors": 0,
        }
    )

    def write(feature_rows=feature_rows, coverage_rows=coverage_rows):
        pd.DataFrame(feature_rows).to_parquet(features / f"{WELL}.parquet")
        pd.DataFrame(coverage_rows).to_parquet(
            features / COVERAGE_SUBDIR / f"{WELL}.parquet"
        )

    write()
    return {
        "features": features,
        "masks": masks,
        "feature_rows": feature_rows,
        "coverage_rows": coverage_rows,
        "write": write,
    }


def _audit(tree) -> list:
    return audit_well(tree["features"], tree["masks"], WELL, MIN_PIXELS).failures


def test_a_consistent_well_passes(tree):
    audit = audit_well(tree["features"], tree["masks"], WELL, MIN_PIXELS)
    assert audit.passed
    assert audit.n_masks == 2
    assert audit.n_feature_rows == 4
    assert audit.status_counts == {"ok": 2, "unsegmented": 1}


def test_a_missing_cell_is_named(tree):
    tree["write"](feature_rows=tree["feature_rows"][1:])
    failures = _audit(tree)
    assert any("labels without a row" in f for f in failures)


def test_an_extra_cell_is_named(tree):
    extra = dict(tree["feature_rows"][0], cell_id=9)
    tree["write"](feature_rows=tree["feature_rows"] + [extra])
    assert any("not among the mask's labels" in f for f in _audit(tree))


def test_a_wrong_cell_id_is_named(tree):
    rows = [dict(r) for r in tree["feature_rows"]]
    rows[0]["cell_id"] = 3
    tree["write"](feature_rows=rows)
    failures = _audit(tree)
    assert any("not among the mask's labels" in f for f in failures)
    assert any("labels without a row" in f for f in failures)


def test_a_mask_without_a_coverage_row_is_named(tree):
    tifffile.imwrite(tree["masks"] / _stack_names("2", 5), _mask())
    assert any("masks without a coverage row" in f for f in _audit(tree))


def test_a_coverage_row_for_a_missing_mask_is_named(tree):
    (tree["masks"] / _stack_names("1", 2)).unlink()
    assert any("masks not on disk" in f for f in _audit(tree))


def test_a_wrong_skipped_count_is_named(tree):
    rows = [dict(r) for r in tree["coverage_rows"]]
    rows[0]["n_skipped_small"] = 0
    tree["write"](coverage_rows=rows)
    assert any("n_skipped_small" in f for f in _audit(tree))


def test_a_duplicate_feature_row_is_named(tree):
    tree["write"](feature_rows=tree["feature_rows"] + [tree["feature_rows"][0]])
    assert any("duplicate" in f for f in _audit(tree))


def test_unsegmented_above_z0_is_named(tree):
    rows = [dict(r) for r in tree["coverage_rows"]]
    rows[-1]["z_index"] = 3
    tree["write"](coverage_rows=rows)
    assert any("unsegmented rows outside z0" in f for f in _audit(tree))


def test_an_unreadable_mask_is_named_without_aborting(tree):
    (tree["masks"] / _stack_names("1", 2)).write_bytes(b"not a tiff")
    failures = _audit(tree)
    assert any("unreadable masks" in f and "_t1_z2_" in f for f in failures)


def test_an_empty_mask_with_no_rows_passes(tree):
    name = _stack_names("1", 2)
    tifffile.imwrite(tree["masks"] / name, np.zeros((40, 40), np.uint16))
    feature_rows = [r for r in tree["feature_rows"] if r["z_index"] != 2]
    coverage_rows = [dict(r) for r in tree["coverage_rows"]]
    coverage_rows[1].update(status="empty", n_cells=0, n_skipped_small=0)
    tree["write"](feature_rows=feature_rows, coverage_rows=coverage_rows)
    assert _audit(tree) == []


def test_an_unaccounted_label_error_is_named(tree):
    rows = [dict(r) for r in tree["coverage_rows"]]
    rows[0]["n_label_errors"] = 1
    tree["write"](coverage_rows=rows)
    assert any("beyond the rejected ones" in f for f in _audit(tree))


def test_the_rejected_label_count_is_reported(tree):
    rows = [dict(r) for r in tree["coverage_rows"]]
    rows[0]["n_cells"], rows[0]["n_label_errors"] = 1, 1
    tree["write"](
        feature_rows=tree["feature_rows"][1:2] + tree["feature_rows"][2:],
        coverage_rows=rows,
    )
    audit = audit_well(tree["features"], tree["masks"], WELL, MIN_PIXELS)
    assert audit.passed and audit.n_rejected_labels == 1


def test_a_coverage_table_without_status_fails_instead_of_crashing(tree):
    rows = [
        {k: v for k, v in r.items() if k != "status"} for r in tree["coverage_rows"]
    ]
    tree["write"](coverage_rows=rows)
    assert any("lacks columns" in f for f in _audit(tree))


def test_an_unexpected_status_is_named(tree):
    rows = [dict(r) for r in tree["coverage_rows"]]
    rows[0]["status"] = "error"
    tree["write"](coverage_rows=rows)
    assert any("unexpected status" in f for f in _audit(tree))


def test_a_missing_label_beyond_the_allowed_rejections_is_named(tree):
    rows = [dict(r) for r in tree["coverage_rows"]]
    rows[0]["n_cells"], rows[0]["n_label_errors"] = 0, 1
    tree["write"](feature_rows=tree["feature_rows"][2:], coverage_rows=rows)
    assert any("beyond the rejected ones" in f for f in _audit(tree))


def test_feature_rows_without_a_coverage_row_are_named(tree):
    orphan = dict(tree["feature_rows"][0], z_index=9)
    tree["write"](feature_rows=tree["feature_rows"] + [orphan])
    assert any("feature rows with no coverage row" in f for f in _audit(tree))


def test_two_coverage_rows_for_one_mask_are_named(tree):
    tree["write"](coverage_rows=tree["coverage_rows"] + [tree["coverage_rows"][0]])
    failures = _audit(tree)
    assert any("masks with more than one coverage row" in f for f in failures)
    assert any("stacks with more than one coverage row" in f for f in failures)


def test_a_wrong_n_cells_is_named(tree):
    rows = [dict(r) for r in tree["coverage_rows"]]
    rows[0]["n_cells"] = 3
    tree["write"](coverage_rows=rows)
    failures = _audit(tree)
    assert any("n_cells differs from the feature rows" in f for f in failures)
    assert any("n_cells sums to" in f for f in failures)


def test_status_empty_with_rows_is_named(tree):
    rows = [dict(r) for r in tree["coverage_rows"]]
    rows[0]["status"] = "empty"
    tree["write"](coverage_rows=rows)
    assert any("status disagrees" in f for f in _audit(tree))


def test_a_feature_table_without_cell_id_fails_and_still_counts_masks(tree):
    rows = [
        {k: v for k, v in r.items() if k != "cell_id"} for r in tree["feature_rows"]
    ]
    tree["write"](feature_rows=rows)
    audit = audit_well(tree["features"], tree["masks"], WELL, MIN_PIXELS)
    assert any("feature table lacks columns" in f for f in audit.failures)
    assert audit.n_masks == 2


def test_a_truncated_table_is_a_named_failure(tree):
    (tree["features"] / f"{WELL}.parquet").write_bytes(b"truncated")
    assert any("unreadable table" in f for f in _audit(tree))


def test_a_missing_mask_directory_is_an_error(tree, tmp_path):
    with pytest.raises(FileNotFoundError, match="Not a directory"):
        audit_tree(tree["features"], tmp_path / "typo", min_pixels=MIN_PIXELS)


def test_a_well_with_no_masks_fails(tree, tmp_path):
    empty_masks = tmp_path / "other_masks"
    empty_masks.mkdir()
    audit = audit_well(tree["features"], empty_masks, WELL, MIN_PIXELS)
    assert any("no masks for well" in f for f in audit.failures)


def test_a_crash_in_one_well_is_a_named_failure_and_the_rest_are_audited(
    tree, monkeypatch
):
    import src.feature_extraction.mask_parity_audit as audit_module

    real = audit_module.audit_well
    pd.DataFrame(tree["feature_rows"]).to_parquet(tree["features"] / "F08.parquet")

    def flaky(features_dir, mask_dir, well, *args):
        if well == "F08":
            raise RuntimeError("boom")
        return real(features_dir, mask_dir, well, *args)

    monkeypatch.setattr(audit_module, "audit_well", flaky)
    result = audit_tree(tree["features"], tree["masks"], min_pixels=MIN_PIXELS)
    by_well = {well.well: well for well in result.wells}
    assert by_well[WELL].passed
    assert "audit crashed (RuntimeError: boom)" in by_well["F08"].failures[0]
    assert not result.passed


def test_write_audit_creates_the_report_directory(tree, tmp_path):
    result = audit_tree(tree["features"], tree["masks"], min_pixels=MIN_PIXELS)
    report = write_audit(result, tmp_path / "new" / "dir" / "audit.json")
    assert report.exists()


def _run_cli(monkeypatch, *argv):
    import importlib.util

    script = Path(__file__).resolve().parents[2] / "scripts" / "run_feature_audit.py"
    spec = importlib.util.spec_from_file_location("_run_feature_audit", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr("sys.argv", ["run_feature_audit.py", *map(str, argv)])
    return module.main()


def test_cli_exit_codes(tree, tmp_path, monkeypatch):
    common = ["--features-dir", tree["features"], "--mask-dir", tree["masks"]]
    assert _run_cli(monkeypatch, *common, "--min-pixels", MIN_PIXELS) == 0
    assert (tree["features"] / "mask_parity_audit.json").exists()

    tree["write"](feature_rows=tree["feature_rows"][1:])
    assert _run_cli(monkeypatch, *common, "--min-pixels", MIN_PIXELS) == 1

    typo = ["--features-dir", tree["features"], "--mask-dir", tmp_path / "typo"]
    assert _run_cli(monkeypatch, *typo) == 2

    blocked = tree["features"] / "mask_parity_audit.json" / "report.json"
    assert _run_cli(monkeypatch, *common, "--report", blocked) == 2


def test_a_missing_well_file_is_reported(tree):
    audit = audit_well(tree["features"], tree["masks"], "F08", MIN_PIXELS)
    assert not audit.passed
    assert "missing file" in audit.failures[0]


def test_audit_tree_discovers_wells_and_writes_the_report(tree, tmp_path):
    result = audit_tree(tree["features"], tree["masks"], min_pixels=MIN_PIXELS)
    assert result.passed
    assert [well.well for well in result.wells] == [WELL]
    report = write_audit(result, tmp_path / "audit.json")
    assert '"passed": true' in report.read_text()


def test_audit_tree_with_no_wells_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="No well tables"):
        audit_tree(tmp_path, tmp_path)


def test_a_coverage_only_well_is_discovered_and_fails(tree):
    (tree["features"] / COVERAGE_SUBDIR / "F08.parquet").write_bytes(
        (tree["features"] / COVERAGE_SUBDIR / f"{WELL}.parquet").read_bytes()
    )
    result = audit_tree(tree["features"], tree["masks"], min_pixels=MIN_PIXELS)
    by_well = {well.well: well for well in result.wells}
    assert sorted(by_well) == ["E07", "F08"]
    assert any("missing file" in f for f in by_well["F08"].failures)
    assert not result.passed


def test_a_well_with_masks_but_no_tables_is_discovered_and_fails(tree):
    tifffile.imwrite(tree["masks"] / "pMF5V1_G09_t1_z1_pred_mask.tif", _mask())
    result = audit_tree(tree["features"], tree["masks"], min_pixels=MIN_PIXELS)
    by_well = {well.well: well for well in result.wells}
    assert sorted(by_well) == ["E07", "G09"]
    assert any("missing file" in f for f in by_well["G09"].failures)
    assert not result.passed


def test_an_unsegmented_row_that_names_a_mask_is_named(tree):
    rows = [dict(r) for r in tree["coverage_rows"]]
    rows[-1]["mask_filename"] = _stack_names("1", 2)
    tree["write"](coverage_rows=rows)
    assert any("unsegmented rows that name a mask" in f for f in _audit(tree))


def test_a_float_mask_is_named_as_unreadable(tree):
    tifffile.imwrite(tree["masks"] / _stack_names("1", 2), _mask().astype(np.float32))
    assert any("unreadable masks" in f and "2D integer" in f for f in _audit(tree))


def test_a_3d_mask_is_named_as_unreadable(tree):
    tifffile.imwrite(tree["masks"] / _stack_names("1", 2), np.stack([_mask(), _mask()]))
    assert any("unreadable masks" in f and "2D integer" in f for f in _audit(tree))


def test_unsegmented_rows_must_have_zero_counts(tree):
    rows = [dict(r) for r in tree["coverage_rows"]]
    rows[-1]["n_label_errors"] = 2
    tree["write"](coverage_rows=rows)
    assert any("unsegmented rows with nonzero counts" in f for f in _audit(tree))


def test_a_duplicated_unsegmented_stack_is_named(tree):
    tree["write"](coverage_rows=tree["coverage_rows"] + [tree["coverage_rows"][-1]])
    assert any("stacks with more than one coverage row" in f for f in _audit(tree))


def test_a_non_integral_cell_id_is_named_and_does_not_pass_as_a_label(tree):
    rows = [dict(r) for r in tree["feature_rows"]]
    rows[0]["cell_id"] = 1.5
    tree["write"](feature_rows=rows)
    failures = _audit(tree)
    assert any("non-integral cell_id" in f for f in failures)
    assert any("labels without a row" in f for f in failures)


def test_a_non_integral_z_index_in_both_tables_is_named(tree):
    features = [dict(r, z_index=1.5) for r in tree["feature_rows"]]
    coverage = [dict(r, z_index=1.5) for r in tree["coverage_rows"]]
    tree["write"](feature_rows=features, coverage_rows=coverage)
    failures = _audit(tree)
    assert any("coverage rows with a non-integral z_index" in f for f in failures)
    assert any("feature rows with a non-integral z_index" in f for f in failures)


@pytest.mark.parametrize(
    "sentinel", [{"timepoint": ""}, {"timepoint": None}, {"z_index": -1}]
)
def test_sentinel_timepoint_and_z_keys_are_named(tree, sentinel):
    features = [dict(r, **sentinel) for r in tree["feature_rows"]]
    coverage = [dict(r, **sentinel) for r in tree["coverage_rows"]]
    tree["write"](feature_rows=features, coverage_rows=coverage)
    failures = _audit(tree)
    assert any(
        "coverage rows with a missing timepoint or z_index" in f for f in failures
    )
    assert any(
        "feature rows with a missing timepoint or z_index" in f for f in failures
    )


def test_tables_of_another_well_are_named(tree):
    features = [dict(r, sample_id="F08") for r in tree["feature_rows"]]
    coverage = [dict(r, sample_id="F08") for r in tree["coverage_rows"]]
    tree["write"](feature_rows=features, coverage_rows=coverage)
    failures = _audit(tree)
    assert any("coverage rows whose sample_id is not E07" in f for f in failures)
    assert any("feature rows whose sample_id is not E07" in f for f in failures)


def test_swapped_mask_filenames_in_coverage_are_named(tree):
    rows = [dict(r) for r in tree["coverage_rows"]]
    rows[0]["mask_filename"], rows[1]["mask_filename"] = (
        rows[1]["mask_filename"],
        rows[0]["mask_filename"],
    )
    tree["write"](coverage_rows=rows)
    assert any("disagrees with the mask filename" in f for f in _audit(tree))


def test_min_pixels_below_one_is_rejected(tree, monkeypatch):
    with pytest.raises(ValueError, match="min_pixels"):
        audit_tree(tree["features"], tree["masks"], min_pixels=0)
    with pytest.raises(ValueError, match="min_pixels"):
        audit_well(tree["features"], tree["masks"], WELL, -1)
    common = ["--features-dir", tree["features"], "--mask-dir", tree["masks"]]
    assert _run_cli(monkeypatch, *common, "--min-pixels", 0) == 2


def test_a_process_pool_gives_the_same_result(tree):
    sequential = audit_well(tree["features"], tree["masks"], WELL, MIN_PIXELS, 1)
    parallel = audit_well(tree["features"], tree["masks"], WELL, MIN_PIXELS, 2)
    assert sequential.passed and sequential == parallel
