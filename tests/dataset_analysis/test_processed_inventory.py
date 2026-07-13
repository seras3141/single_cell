"""Tests for processed_inventory: annotate_with_raw_issues, build_processed_summary, print_summary_table."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.dataset_analysis.processed_inventory import (
    annotate_with_raw_issues,
    build_processed_summary,
    detect_phantom_samples,
    parse_sample_stem,
    print_summary_table,
)


def _make_inventory() -> pd.DataFrame:
    """Minimal inventory with two stages and a mix of found/missing entries."""
    return pd.DataFrame([
        {"stage": "prepare-3d", "well_id": "E07", "time_point": 1,  "z_index": None, "channel": "BF", "expected_path": "/x/a.tif", "found": False, "file_size_mb": None},
        {"stage": "prepare-3d", "well_id": "E07", "time_point": 2,  "z_index": None, "channel": "BF", "expected_path": "/x/b.tif", "found": True,  "file_size_mb": 1.0},
        {"stage": "prepare-3d", "well_id": "F01", "time_point": 1,  "z_index": None, "channel": "BF", "expected_path": "/x/c.tif", "found": False, "file_size_mb": None},
        {"stage": "mcherry",    "well_id": None,  "time_point": None, "z_index": None, "channel": None, "expected_path": "/x/m.csv", "found": True,  "file_size_mb": 0.5},
    ])


def _make_issues() -> pd.DataFrame:
    return pd.DataFrame([
        {"well_id": "E07", "time_point": 1, "issue_type": "missing_z", "z_index": 1},
        {"well_id": "E07", "time_point": 1, "issue_type": "missing_z", "z_index": 2},
    ])


class TestAnnotateWithRawIssues:
    def test_matching_row_gets_issue_type(self):
        inv = _make_inventory()
        issues = _make_issues()
        result = annotate_with_raw_issues(inv, issues)
        e07_t1 = result[(result["well_id"] == "E07") & (result["time_point"] == 1)]
        assert (e07_t1["raw_issue_type"] == "missing_z").all()

    def test_non_matching_row_gets_nan(self):
        inv = _make_inventory()
        issues = _make_issues()
        result = annotate_with_raw_issues(inv, issues)
        e07_t2 = result[(result["well_id"] == "E07") & (result["time_point"] == 2)]
        assert e07_t2["raw_issue_type"].isna().all()

    def test_no_issue_for_well_gets_nan(self):
        inv = _make_inventory()
        issues = _make_issues()
        result = annotate_with_raw_issues(inv, issues)
        f01 = result[(result["well_id"] == "F01") & (result["time_point"] == 1)]
        assert f01["raw_issue_type"].isna().all()

    def test_deduplicates_multiple_z_rows_to_first_issue_type(self):
        inv = _make_inventory()
        issues = pd.DataFrame([
            {"well_id": "E07", "time_point": 1, "issue_type": "missing_z",         "z_index": 1},
            {"well_id": "E07", "time_point": 1, "issue_type": "missing_channel_z", "z_index": 2},
        ])
        result = annotate_with_raw_issues(inv, issues)
        e07_t1 = result[(result["well_id"] == "E07") & (result["time_point"] == 1)]
        assert (e07_t1["raw_issue_type"] == "missing_z").all()


class TestBuildProcessedSummaryWithIssues:
    def test_explained_count_equals_missing_with_matching_issue(self):
        inv = _make_inventory()
        issues = _make_issues()
        summary = build_processed_summary(inv, issues_df=issues)
        assert summary["prepare-3d"]["explained_by_raw_issues"] == 1

    def test_explained_zero_when_no_issue_matches(self):
        inv = _make_inventory()
        issues = pd.DataFrame(columns=["well_id", "time_point", "issue_type"])
        summary = build_processed_summary(inv, issues_df=issues)
        assert summary["prepare-3d"]["explained_by_raw_issues"] == 0

    def test_all_issue_types_count_as_explained(self):
        inv = _make_inventory()
        issues = pd.DataFrame([
            {"well_id": "E07", "time_point": 1, "issue_type": "missing_channel_z", "z_index": 1},
            {"well_id": "F01", "time_point": 1, "issue_type": "missing_z",         "z_index": 1},
        ])
        summary = build_processed_summary(inv, issues_df=issues)
        assert summary["prepare-3d"]["explained_by_raw_issues"] == 2

    def test_no_issues_df_omits_explained_key(self):
        inv = _make_inventory()
        summary = build_processed_summary(inv)
        assert "explained_by_raw_issues" not in summary["prepare-3d"]

    def test_missing_count_unchanged_by_issues(self):
        inv = _make_inventory()
        issues = _make_issues()
        summary_with = build_processed_summary(inv, issues_df=issues)
        summary_without = build_processed_summary(inv)
        assert summary_with["prepare-3d"]["missing"] == summary_without["prepare-3d"]["missing"]


class TestPrintSummaryTable:
    def _capture(self, summary: dict) -> str:
        with patch("builtins.print") as mock_print:
            print_summary_table(summary)
            return "\n".join(str(c.args[0]) for c in mock_print.call_args_list)

    def test_explained_column_shown_when_any_stage_has_explained(self):
        inv = _make_inventory()
        issues = _make_issues()
        summary = build_processed_summary(inv, issues_df=issues)
        output = self._capture(summary)
        assert "Explained" in output

    def test_explained_column_hidden_when_all_zero(self):
        inv = _make_inventory()
        issues = pd.DataFrame(columns=["well_id", "time_point", "issue_type"])
        summary = build_processed_summary(inv, issues_df=issues)
        output = self._capture(summary)
        assert "Explained" not in output

    def test_explained_column_hidden_without_issues_df(self):
        inv = _make_inventory()
        summary = build_processed_summary(inv)
        output = self._capture(summary)
        assert "Explained" not in output


def _make_raw_inventory(wells_timepoints: list) -> pd.DataFrame:
    return pd.DataFrame([
        {"well_id": well, "time_point": tp, "plate_id": "pMF5V1", "wavelength": 1, "z_index": 0}
        for well, tp in wells_timepoints
    ])


class TestParseSampleStem:
    def test_split_tif(self):
        assert parse_sample_stem("pMF5V1_E07_t101_z0_BF.tif") == "E07_t101"

    def test_3d_tif(self):
        assert parse_sample_stem("pMF5V1_E07_t101_BF_3d.tif") == "E07_t101"

    def test_zarr(self):
        assert parse_sample_stem("pMF5V1_E07_t101_z0_pred_mask.zarr") == "E07_t101"

    def test_multidigit_timepoint(self):
        assert parse_sample_stem("pMF5V1_H07_t001_BF_3d.tif") == "H07_t001"

    def test_unrecognised_name_returns_none(self):
        assert parse_sample_stem("unrelated_file.tif") is None

    def test_path_object_name_is_used(self):
        assert parse_sample_stem("some/path/pMF5V1_G08_t5_BF_3d.tif") == "G08_t5"


class TestDetectPhantomSamples:
    def test_detects_phantom_in_3d_dir(self, tmp_path: Path):
        raw = _make_raw_inventory([("E07", 1), ("F01", 1)])
        d3_dir = tmp_path / "3d_data"
        d3_dir.mkdir()
        (d3_dir / "pMF5V1_E07_t1_BF_3d.tif").touch()
        (d3_dir / "pMF5V1_F01_t1_BF_3d.tif").touch()
        (d3_dir / "pMF5V1_G08_t1_BF_3d.tif").touch()  # phantom

        result = detect_phantom_samples(raw, tmp_path)
        assert len(result["prepare-3d"]) == 1
        assert result["prepare-3d"][0].name == "pMF5V1_G08_t1_BF_3d.tif"

    def test_detects_phantoms_in_both_masks_subdirs(self, tmp_path: Path):
        raw = _make_raw_inventory([("E07", 1)])
        masks_dir = tmp_path / "inference" / "cellpose_sam" / "masks"
        masks_3d_dir = tmp_path / "inference" / "cellpose_sam" / "masks_3d"
        masks_dir.mkdir(parents=True)
        masks_3d_dir.mkdir(parents=True)
        (masks_dir / "pMF5V1_E07_t1_z0_pred_mask.zarr").mkdir()
        (masks_dir / "pMF5V1_F07_t1_z0_pred_mask.zarr").mkdir()    # phantom
        (masks_3d_dir / "pMF5V1_G08_t1_pred_mask.zarr").mkdir()    # phantom

        result = detect_phantom_samples(raw, tmp_path)
        assert len(result["segment-2d"]) == 2

    def test_detects_phantoms_in_both_track_subdirs(self, tmp_path: Path):
        raw = _make_raw_inventory([("E07", 1)])
        final_dir = tmp_path / "inference_tracked" / "cellpose_sam" / "test" / "final"
        final_2d_dir = tmp_path / "inference_tracked" / "cellpose_sam" / "test" / "final_2d"
        final_dir.mkdir(parents=True)
        final_2d_dir.mkdir(parents=True)
        (final_dir / "pMF5V1_E07_t1_pred_mask_3d.zarr").mkdir()
        (final_dir / "pMF5V1_H07_t1_pred_mask_3d.zarr").mkdir()    # phantom
        (final_2d_dir / "pMF5V1_H07_t1_z0_pred_mask.zarr").mkdir() # phantom

        result = detect_phantom_samples(raw, tmp_path)
        assert len(result["track"]) == 2

    def test_no_phantoms_returns_empty_lists(self, tmp_path: Path):
        raw = _make_raw_inventory([("E07", 1)])
        d3_dir = tmp_path / "3d_data"
        d3_dir.mkdir()
        (d3_dir / "pMF5V1_E07_t1_BF_3d.tif").touch()

        result = detect_phantom_samples(raw, tmp_path)
        assert result["prepare-3d"] == []

    def test_missing_stage_dir_returns_empty_list(self, tmp_path: Path):
        raw = _make_raw_inventory([("E07", 1)])
        result = detect_phantom_samples(raw, tmp_path)
        assert result["prepare-3d"] == []

    def test_mcherry_not_in_result(self, tmp_path: Path):
        raw = _make_raw_inventory([("E07", 1)])
        result = detect_phantom_samples(raw, tmp_path)
        assert "mcherry" not in result
