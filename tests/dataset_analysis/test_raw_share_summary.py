"""Tests for raw-share download and acquisition completeness reporting."""

import json

import pandas as pd
import pytest

from src.dataset_analysis.raw_share_summary import (
    build_raw_summary,
    missing_frames_table,
    parse_frame_name,
    scan_frames,
    write_raw_summary,
)


def frame(t, well="E07", site=1, wavelength=1, z=1):
    return f"t{t}_{well}_s{site}_w{wavelength}_z{z}.tif"


def make_folder(root, names, size=32):
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        (root / name).write_bytes(b"x" * size)
    return root


def test_parse_frame_name_extracts_all_axes():
    parsed = parse_frame_name("t11_E07_s1_w2_z13.tif")
    assert parsed == {
        "time_point": 11,
        "well_id": "E07",
        "site": 1,
        "wavelength": 2,
        "z_index": 13,
    }


def test_parse_frame_name_accepts_projection_z0():
    assert parse_frame_name("t1_N11_s1_w3_z0.tif")["z_index"] == 0


@pytest.mark.parametrize(
    "name",
    [
        "not_a_frame.tif",
        "t1_E07_s1_w1_z1.tiff",  # wrong extension
        "t1_E07_s1_w1.tif",  # missing z
        "E07_s1_w1_z1.tif",  # missing t
    ],
)
def test_parse_frame_name_rejects_non_frames(name):
    assert parse_frame_name(name) is None


def test_scan_frames_ignores_unparseable_and_is_non_recursive(tmp_path):
    folder = make_folder(tmp_path / "exp", [frame(1), frame(2)])
    (folder / "notes.txt").write_text("ignore me")
    (folder / "stray.tif").write_bytes(b"x")
    nested = folder / "exp_Projection"
    make_folder(nested, [frame(1, wavelength=1, z=0)])

    frames = scan_frames(folder)
    assert sorted(frames["file_name"]) == [frame(1), frame(2)]


def test_scan_frames_empty_folder_returns_typed_empty_frame(tmp_path):
    folder = tmp_path / "empty"
    folder.mkdir()
    frames = scan_frames(folder)
    assert frames.empty
    assert "file_name" in frames.columns


def test_download_block_reports_gap_against_manifest(tmp_path):
    folder = make_folder(tmp_path / "exp", [frame(1), frame(2)])
    manifest = tmp_path / "exp.filelist.txt"
    manifest.write_text("\n".join([frame(1), frame(2), frame(3)]) + "\n")

    summary = build_raw_summary(folder, manifest_path=manifest)
    assert summary["download"]["expected"] == 3
    assert summary["download"]["found"] == 2
    assert summary["download"]["missing"] == 1
    assert summary["download"]["missing_entries"] == [frame(3)]


def test_download_block_absent_without_manifest(tmp_path):
    folder = make_folder(tmp_path / "exp", [frame(1)])
    assert "download" not in build_raw_summary(folder)


def test_dense_channel_complete_sparse_channel_missing(tmp_path):
    """w1 dense over t1..t3, w2 sampled only at t1 -- the real dataset shape."""
    names = [frame(t, wavelength=1, z=1) for t in (1, 2, 3)]
    names += [frame(1, wavelength=2, z=1)]
    folder = make_folder(tmp_path / "exp", names)

    summary = build_raw_summary(
        folder,
        expected_timepoints=(1, 2, 3),
        expected_wavelengths=(1, 2),
        expected_z=(1,),
    )
    acquisition = summary["acquisition"]
    assert acquisition["w1"]["missing"] == 0
    assert acquisition["w1"]["timepoints_present"] == [1, 2, 3]
    assert acquisition["w2"]["missing"] == 2
    assert acquisition["w2"]["timepoints_present"] == [1]


def test_empty_folder_with_manifest_reports_everything_missing(tmp_path):
    """Regression: axes taken from disk alone made an empty folder look perfect.

    With no files on disk the well axis is unknown, so the expected grid came
    out empty and every channel reported ``0 missing`` for a folder holding
    nothing at all.  The manifest must supply the axes in that case.
    """
    folder = tmp_path / "exp"
    folder.mkdir()
    manifest = tmp_path / "exp.filelist.txt"
    manifest.write_text("\n".join(frame(t, wavelength=1, z=1) for t in (1, 2)) + "\n")

    summary = build_raw_summary(
        folder,
        manifest_path=manifest,
        expected_timepoints=(1, 2),
        expected_wavelengths=(1,),
        expected_z=(1,),
    )
    assert summary["files_on_disk"] == 0
    assert summary["wells"] == ["E07"]
    assert summary["acquisition"]["w1"]["expected"] == 2
    assert summary["acquisition"]["w1"]["missing"] == 2


def test_projection_folder_expects_z0_only(tmp_path):
    folder = make_folder(tmp_path / "exp_Projection", [frame(1, wavelength=1, z=0)])
    summary = build_raw_summary(
        folder, expected_timepoints=(1,), expected_wavelengths=(1,)
    )
    assert summary["is_projection"] is True
    assert summary["expected_z"] == [0]
    assert summary["acquisition"]["w1"]["missing"] == 0


def test_multiple_wells_scale_the_expected_grid(tmp_path):
    names = [
        frame(t, well=well, wavelength=1, z=1) for well in ("E07", "N11") for t in (1,)
    ]
    folder = make_folder(tmp_path / "exp", names)
    summary = build_raw_summary(
        folder,
        expected_timepoints=(1, 2),
        expected_wavelengths=(1,),
        expected_z=(1,),
    )
    # 2 wells x 2 timepoints expected, 2 present
    assert summary["acquisition"]["w1"]["expected"] == 4
    assert summary["acquisition"]["w1"]["missing"] == 2


def test_missing_frames_table_lists_every_absent_frame(tmp_path):
    folder = make_folder(tmp_path / "exp", [frame(1, wavelength=1, z=1)])
    summary = build_raw_summary(
        folder,
        expected_timepoints=(1, 2, 3),
        expected_wavelengths=(1,),
        expected_z=(1,),
    )
    table = missing_frames_table(summary, "exp")
    assert sorted(table["file_name"]) == [
        frame(2, wavelength=1, z=1),
        frame(3, wavelength=1, z=1),
    ]
    assert set(table["reason"]) == {"timepoint_not_acquired"}


def test_missing_table_includes_gaps_inside_an_acquired_timepoint(tmp_path):
    """Regression: a z-hole at a *present* timepoint was counted but not listed.

    The table used to be derived from the timepoint axis, skipping any timepoint
    that existed at all -- so a missing z-plane inflated the ``missing`` count in
    the JSON while never appearing in raw_missing.csv.
    """
    # t1 exists but only at z1; z2 is a gap inside an acquired timepoint.
    folder = make_folder(tmp_path / "exp", [frame(1, wavelength=1, z=1)])
    summary = build_raw_summary(
        folder,
        expected_timepoints=(1,),
        expected_wavelengths=(1,),
        expected_z=(1, 2),
    )
    assert summary["acquisition"]["w1"]["missing"] == 1
    table = missing_frames_table(summary, "exp")
    # count in the JSON and rows in the CSV must agree
    assert len(table) == summary["acquisition"]["w1"]["missing"]
    assert table["file_name"].tolist() == [frame(1, wavelength=1, z=2)]
    assert table["reason"].tolist() == ["gap_within_timepoint"]


def test_missing_table_row_count_matches_counts_for_mixed_gaps(tmp_path):
    """Whole-timepoint absence and in-timepoint gaps both land in the table."""
    folder = make_folder(tmp_path / "exp", [frame(1, wavelength=1, z=1)])
    summary = build_raw_summary(
        folder,
        expected_timepoints=(1, 2),
        expected_wavelengths=(1,),
        expected_z=(1, 2),
    )
    table = missing_frames_table(summary, "exp")
    assert len(table) == summary["acquisition"]["w1"]["missing"] == 3
    assert set(table["reason"]) == {"gap_within_timepoint", "timepoint_not_acquired"}


def test_internal_missing_list_not_written_to_json(tmp_path):
    folder = make_folder(tmp_path / "exp", [frame(1, wavelength=1, z=1)])
    summary = build_raw_summary(
        folder,
        expected_timepoints=(1, 2),
        expected_wavelengths=(1,),
        expected_z=(1,),
    )
    written = write_raw_summary([summary], tmp_path / "out")
    payload = json.loads(written["summary"].read_text())
    assert "_missing_all" not in payload[0]["acquisition"]["w1"]
    # ...but the CSV still has the full list
    assert len(pd.read_csv(written["missing"])) == 1


def test_write_raw_summary_emits_json_and_csvs(tmp_path):
    folder = make_folder(tmp_path / "exp", [frame(1, wavelength=1, z=1)])
    summary = build_raw_summary(
        folder,
        expected_timepoints=(1, 2),
        expected_wavelengths=(1,),
        expected_z=(1,),
    )
    written = write_raw_summary([summary], tmp_path / "raw_summary")

    payload = json.loads(written["summary"].read_text())
    assert payload[0]["folder"] == "exp"
    assert payload[0]["acquisition"]["w1"]["missing"] == 1

    inventory = pd.read_csv(written["inventory"])
    assert inventory.loc[0, "files_on_disk"] == 1
    assert inventory.loc[0, "w1_missing"] == 1

    missing = pd.read_csv(written["missing"])
    assert missing["file_name"].tolist() == [frame(2, wavelength=1, z=1)]


def test_inventory_csv_carries_download_gap(tmp_path):
    folder = make_folder(tmp_path / "exp", [frame(1), frame(2)])
    manifest = tmp_path / "exp.filelist.txt"
    manifest.write_text("\n".join([frame(1), frame(2), frame(3)]) + "\n")
    with_manifest = build_raw_summary(folder, manifest_path=manifest)
    without_manifest = build_raw_summary(folder)

    written = write_raw_summary([with_manifest, without_manifest], tmp_path / "out")
    inventory = pd.read_csv(written["inventory"])
    assert inventory.loc[0, "download_missing"] == 1
    assert pd.isna(inventory.loc[1, "download_missing"])


@pytest.mark.parametrize("axis", ["expected_timepoints", "expected_z"])
def test_empty_expected_axis_is_rejected(tmp_path, axis):
    folder = make_folder(tmp_path / "exp", [frame(1)])
    with pytest.raises(ValueError, match=axis):
        build_raw_summary(folder, **{axis: ()})


def test_write_raw_summary_handles_folder_with_no_gaps(tmp_path):
    folder = make_folder(tmp_path / "exp", [frame(1, wavelength=1, z=1)])
    summary = build_raw_summary(
        folder,
        expected_timepoints=(1,),
        expected_wavelengths=(1,),
        expected_z=(1,),
    )
    written = write_raw_summary([summary], tmp_path / "out")
    missing = pd.read_csv(written["missing"])
    assert missing.empty
    assert list(missing.columns) == ["folder", "wavelength", "file_name", "reason"]
