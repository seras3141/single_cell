"""Tests for staging the z0 projection plane into a standalone tree."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from src.dataset_analysis.z0_tree import (
    ProcessedName,
    build_z0_tree,
    iter_z0_entries,
    parse_processed_name,
)

EXPERIMENT = "Ew2-1 MF5V1 0-72h 06-03-26"
MASKS_REL = "inference/cellpose_sam/masks"


def _make_source_tree(root: Path, wells=("C09", "M11"), timepoints=(1, 11), z_max=2):
    """Build a miniature processed tree: split_data + inference masks over z0..z_max."""
    split = root / EXPERIMENT / "split_data"
    masks = root / EXPERIMENT / MASKS_REL
    split.mkdir(parents=True)
    masks.mkdir(parents=True)

    for well in wells:
        for tp in timepoints:
            for z in range(z_max + 1):
                for channel in ("BF", "mCherry", "FlipGFP"):
                    (split / f"pMF5V1_{well}_t{tp}_z{z}_{channel}.tif").write_text("x")
                # .zarr masks are directories, not files
                (masks / f"pMF5V1_{well}_t{tp}_z{z}_pred_mask.zarr").mkdir()

    # sidecar that must be ignored rather than crash the parser
    (split / "dataset_split.json").write_text("{}")
    return root


class TestParseProcessedName:
    def test_parses_channel_tif(self):
        assert parse_processed_name("pMF5V1_C09_t101_z0_BF.tif") == ProcessedName(
            well="C09", timepoint=101, z_index=0, suffix="BF"
        )

    def test_parses_zarr_mask_directory(self):
        parsed = parse_processed_name("pMF5V1_C09_t101_z0_pred_mask.zarr")
        assert parsed == ProcessedName(
            well="C09", timepoint=101, z_index=0, suffix="pred_mask"
        )

    def test_parses_multi_digit_z_and_timepoint(self):
        parsed = parse_processed_name("pMF5V1_N11_t351_z20_mCherry.tif")
        assert (parsed.timepoint, parsed.z_index) == (351, 20)

    def test_uppercases_well(self):
        assert parse_processed_name("pMF5V1_c09_t1_z0_BF.tif").well == "C09"

    @pytest.mark.parametrize("name", ["dataset_split.json", "summary.csv", "notes.txt"])
    def test_returns_none_for_unparseable(self, name):
        assert parse_processed_name(name) is None


class TestIterZ0Entries:
    def test_yields_only_the_projection_plane(self, tmp_path):
        _make_source_tree(tmp_path)
        entries = list(iter_z0_entries(tmp_path / EXPERIMENT / "split_data"))
        assert entries, "expected z0 entries"
        assert all(parsed.z_index == 0 for _, parsed in entries)
        # 2 wells x 2 timepoints x 3 channels
        assert len(entries) == 12

    def test_skips_sidecar_files(self, tmp_path):
        _make_source_tree(tmp_path)
        names = [
            p.name for p, _ in iter_z0_entries(tmp_path / EXPERIMENT / "split_data")
        ]
        assert "dataset_split.json" not in names

    def test_missing_directory_yields_nothing(self, tmp_path):
        assert list(iter_z0_entries(tmp_path / "absent")) == []

    def test_honours_non_default_z_index(self, tmp_path):
        _make_source_tree(tmp_path)
        entries = list(iter_z0_entries(tmp_path / EXPERIMENT / "split_data", z_index=2))
        assert all(parsed.z_index == 2 for _, parsed in entries)


class TestBuildZ0Tree:
    def test_mirrors_structure_and_links_only_z0(self, tmp_path):
        src = _make_source_tree(tmp_path / "src")
        dest = tmp_path / "dest"

        summary = build_z0_tree(src, dest)

        split_out = dest / EXPERIMENT / "split_data"
        masks_out = dest / EXPERIMENT / MASKS_REL
        assert split_out.is_dir() and masks_out.is_dir()
        assert len(list(split_out.iterdir())) == 12
        assert len(list(masks_out.iterdir())) == 4  # 2 wells x 2 timepoints
        staged = [parse_processed_name(p.name) for p in split_out.iterdir()]
        assert all(parsed.z_index == 0 for parsed in staged)
        assert set(summary["stage"]) == {"split_data", "inference_masks"}

    def test_symlinks_resolve_to_the_source_payload(self, tmp_path):
        src = _make_source_tree(tmp_path / "src")
        dest = tmp_path / "dest"
        build_z0_tree(src, dest)

        link = dest / EXPERIMENT / "split_data" / "pMF5V1_C09_t1_z0_BF.tif"
        assert link.is_symlink()
        assert (
            link.resolve()
            == (src / EXPERIMENT / "split_data" / "pMF5V1_C09_t1_z0_BF.tif").resolve()
        )
        assert link.read_text() == "x"

    def test_zarr_mask_directories_are_linked(self, tmp_path):
        src = _make_source_tree(tmp_path / "src")
        dest = tmp_path / "dest"
        build_z0_tree(src, dest)

        link = dest / EXPERIMENT / MASKS_REL / "pMF5V1_C09_t1_z0_pred_mask.zarr"
        assert link.is_symlink() and link.is_dir()

    def test_is_idempotent(self, tmp_path):
        src = _make_source_tree(tmp_path / "src")
        dest = tmp_path / "dest"

        first = build_z0_tree(src, dest)
        second = build_z0_tree(src, dest)

        assert first["n_created"].sum() == 16
        assert second["n_created"].sum() == 0
        assert second["n_repaired"].sum() == 0
        assert second["n_existing"].sum() == 16
        assert second["n_z0"].sum() == first["n_z0"].sum()

    def test_dry_run_touches_nothing(self, tmp_path):
        src = _make_source_tree(tmp_path / "src")
        dest = tmp_path / "dest"

        summary = build_z0_tree(src, dest, dry_run=True)

        assert not dest.exists()
        assert summary["n_created"].sum() == 16

    def test_copy_mode_produces_independent_payload(self, tmp_path):
        src = _make_source_tree(tmp_path / "src")
        dest = tmp_path / "dest"
        build_z0_tree(src, dest, mode="copy")

        copied = dest / EXPERIMENT / "split_data" / "pMF5V1_C09_t1_z0_BF.tif"
        assert copied.is_file() and not copied.is_symlink()
        mask_dir = dest / EXPERIMENT / MASKS_REL / "pMF5V1_C09_t1_z0_pred_mask.zarr"
        assert mask_dir.is_dir()

    def test_rejects_unknown_mode(self, tmp_path):
        src = _make_source_tree(tmp_path / "src")
        with pytest.raises(ValueError, match="symlink"):
            build_z0_tree(src, tmp_path / "dest", mode="hardlink")

    def test_rejects_unknown_mode_in_dry_run(self, tmp_path):
        """A typo'd mode must fail loudly, not report a successful dry run."""
        src = _make_source_tree(tmp_path / "src")
        with pytest.raises(ValueError, match="symlink"):
            build_z0_tree(src, tmp_path / "dest", mode="hardlink", dry_run=True)

    def test_rejects_unknown_mode_before_creating_directories(self, tmp_path):
        src = _make_source_tree(tmp_path / "src")
        dest = tmp_path / "dest"
        with pytest.raises(ValueError):
            build_z0_tree(src, dest, mode="hardlink")
        assert not dest.exists(), "no half-built tree may be left behind"

    def test_missing_source_root_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="source root"):
            build_z0_tree(tmp_path / "absent", tmp_path / "dest")

    def test_experiment_filter_selects_a_subset(self, tmp_path):
        src = _make_source_tree(tmp_path / "src")
        (src / "OtherExp" / "split_data").mkdir(parents=True)
        (src / "OtherExp" / "split_data" / "pMF5V1_A01_t1_z0_BF.tif").write_text("y")

        summary = build_z0_tree(src, tmp_path / "dest", experiments=[EXPERIMENT])

        assert set(summary["experiment"]) == {EXPERIMENT}
        assert not (tmp_path / "dest" / "OtherExp").exists()

    def test_missing_named_experiment_raises(self, tmp_path):
        """Experiment dirs carry spaces and dates, so a typo must not pass."""
        src = _make_source_tree(tmp_path / "src")
        with pytest.raises(FileNotFoundError, match="not found"):
            build_z0_tree(src, tmp_path / "dest", experiments=["Ew2-1 typo"])

    def test_missing_stage_dir_is_skipped_not_fatal(self, tmp_path):
        src = tmp_path / "src"
        (src / EXPERIMENT / "split_data").mkdir(parents=True)
        (src / EXPERIMENT / "split_data" / "pMF5V1_A01_t1_z0_BF.tif").write_text("y")

        summary = build_z0_tree(src, tmp_path / "dest")

        assert list(summary["stage"]) == ["split_data"]

    def test_reports_source_counts_alongside_z0_counts(self, tmp_path):
        src = _make_source_tree(tmp_path / "src")
        summary = build_z0_tree(src, tmp_path / "dest")

        split_row = summary[summary["stage"] == "split_data"].iloc[0]
        # 2 wells x 2 timepoints x 3 z x 3 channels + 1 sidecar
        assert split_row["n_source"] == 37
        assert split_row["n_z0"] == 12


class TestZ0TreeRepair:
    """Re-running must repair a tree left inconsistent by an interrupted job."""

    def test_symlink_targets_the_processed_tree_not_the_raw_share(self, tmp_path):
        """split_data entries are themselves symlinks into the raw share.

        Staging must point at the processed tree, so a later fix to the processed
        symlink propagates and the z0 tree does not silently depend on the raw mount.
        """
        raw = tmp_path / "raw"
        raw.mkdir()
        raw_file = raw / "t1_C09_s1_w3_z0.tif"
        raw_file.write_text("raw-payload")

        src = tmp_path / "src"
        split = src / EXPERIMENT / "split_data"
        split.mkdir(parents=True)
        (src / EXPERIMENT / MASKS_REL).mkdir(parents=True)
        processed_entry = split / "pMF5V1_C09_t1_z0_BF.tif"
        processed_entry.symlink_to(raw_file)

        dest = tmp_path / "dest"
        build_z0_tree(src, dest)

        staged = dest / EXPERIMENT / "split_data" / "pMF5V1_C09_t1_z0_BF.tif"
        assert Path(os.readlink(staged)) == processed_entry.absolute()
        assert Path(os.readlink(staged)) != raw_file.absolute()
        assert staged.read_text() == "raw-payload"

    def test_dangling_symlink_is_rebuilt(self, tmp_path):
        src = _make_source_tree(tmp_path / "src")
        dest = tmp_path / "dest"
        build_z0_tree(src, dest)

        staged = dest / EXPERIMENT / "split_data" / "pMF5V1_C09_t1_z0_BF.tif"
        staged.unlink()
        staged.symlink_to(tmp_path / "gone.tif")
        assert not staged.exists() and staged.is_symlink()

        summary = build_z0_tree(src, dest)

        assert summary["n_repaired"].sum() == 1
        assert staged.exists() and staged.read_text() == "x"

    def test_symlink_to_the_wrong_target_is_rebuilt(self, tmp_path):
        src = _make_source_tree(tmp_path / "src")
        dest = tmp_path / "dest"
        build_z0_tree(src, dest)

        decoy = tmp_path / "decoy.tif"
        decoy.write_text("wrong")
        staged = dest / EXPERIMENT / "split_data" / "pMF5V1_C09_t1_z0_BF.tif"
        staged.unlink()
        staged.symlink_to(decoy)

        summary = build_z0_tree(src, dest)

        assert summary["n_repaired"].sum() == 1
        assert staged.read_text() == "x"

    def test_copy_mode_replaces_a_symlink_from_a_previous_run(self, tmp_path):
        src = _make_source_tree(tmp_path / "src")
        dest = tmp_path / "dest"
        build_z0_tree(src, dest, mode="symlink")

        summary = build_z0_tree(src, dest, mode="copy")

        staged = dest / EXPERIMENT / "split_data" / "pMF5V1_C09_t1_z0_BF.tif"
        assert summary["n_repaired"].sum() == 16
        assert staged.is_file() and not staged.is_symlink()

    def test_force_rematerializes_current_entries(self, tmp_path):
        src = _make_source_tree(tmp_path / "src")
        dest = tmp_path / "dest"
        build_z0_tree(src, dest, mode="copy")

        summary = build_z0_tree(src, dest, mode="copy", force=True)

        assert summary["n_repaired"].sum() == 16
        assert summary["n_existing"].sum() == 0

    def test_repair_counts_toward_n_z0_not_lost(self, tmp_path):
        src = _make_source_tree(tmp_path / "src")
        dest = tmp_path / "dest"
        first = build_z0_tree(src, dest)
        second = build_z0_tree(src, dest, force=True)

        assert second["n_z0"].sum() == first["n_z0"].sum()


class TestDestructiveRoots:
    def test_identical_roots_are_rejected(self, tmp_path):
        """Staging in place would delete every source file and self-symlink it."""
        src = _make_source_tree(tmp_path / "src")
        with pytest.raises(ValueError, match="same directory"):
            build_z0_tree(src, src)

    def test_roots_that_resolve_the_same_are_rejected(self, tmp_path):
        src = _make_source_tree(tmp_path / "src")
        alias = tmp_path / "alias"
        alias.symlink_to(src)
        with pytest.raises(ValueError, match="same directory"):
            build_z0_tree(src, alias)

    def test_the_source_survives_a_rejected_run(self, tmp_path):
        src = _make_source_tree(tmp_path / "src")
        before = sorted(p.name for p in (src / EXPERIMENT / "split_data").iterdir())
        with pytest.raises(ValueError):
            build_z0_tree(src, src)
        after = sorted(p.name for p in (src / EXPERIMENT / "split_data").iterdir())
        assert before == after

    def test_copy_mode_rebuilds_a_kind_mismatch(self, tmp_path):
        """A .zarr store replaced by a regular file must be repaired, not kept."""
        src = _make_source_tree(tmp_path / "src")
        dest = tmp_path / "dest"
        build_z0_tree(src, dest, mode="copy")

        staged = dest / EXPERIMENT / MASKS_REL / "pMF5V1_C09_t1_z0_pred_mask.zarr"
        assert staged.is_dir()
        shutil.rmtree(staged)
        staged.write_text("not a store")  # wrong kind

        summary = build_z0_tree(src, dest, mode="copy")

        assert summary["n_repaired"].sum() == 1
        assert staged.is_dir()
