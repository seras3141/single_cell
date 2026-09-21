"""Tests for the BF + mask filmstrip across timepoints."""

from __future__ import annotations

import logging

import numpy as np
import pytest
import tifffile

from src.visualize.filmstrip import (
    DEFAULT_TIMEPOINTS,
    MASK_STYLE_OUTLINE,
    _outline_rgba,
    available_wells,
    condition_label,
    EARLY_TIMEPOINTS,
    Frame,
    _centre_crop,
    build_filmstrip,
    resolve_frames,
)

#: The real acquisition grid: 1, 11, 21, ... 351.
GRID = list(range(1, 352, 10))


def _make_tree(tmp_path, wells=("M11", "E07"), timepoints=None, with_masks=True):
    """A miniature z0 tree: split_data BF frames + inference masks."""
    timepoints = timepoints if timepoints is not None else GRID
    split = tmp_path / "split_data"
    masks = tmp_path / "masks"
    split.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    for well in wells:
        for tp in timepoints:
            bf = np.full((16, 16), 100, dtype=np.uint16)
            tifffile.imwrite(str(split / f"pMF5V1_{well}_t{tp}_z0_BF.tif"), bf)
            tifffile.imwrite(str(split / f"pMF5V1_{well}_t{tp}_z0_mCherry.tif"), bf)
            if with_masks:
                lab = np.zeros((16, 16), dtype=np.uint16)
                lab[0, 0] = 1
                lab[5, 5] = 2
                tifffile.imwrite(
                    str(masks / f"pMF5V1_{well}_t{tp}_z0_pred_mask.tif"), lab
                )
    return split, masks


class TestResolveFrames:
    def test_snaps_requested_timepoints_to_the_acquisition_grid(self, tmp_path):
        """t=50/100/150/200 are not acquired; the grid steps by 10 from 1."""
        split, masks = _make_tree(tmp_path)
        frames = resolve_frames(split, masks, "M11", [1, 50, 100, 150, 200])
        assert [f.timepoint for f in frames] == [1, 51, 101, 151, 201]
        assert [f.requested for f in frames] == [1, 50, 100, 150, 200]

    def test_logs_each_substitution(self, tmp_path, caplog):
        split, masks = _make_tree(tmp_path)
        with caplog.at_level(logging.INFO):
            resolve_frames(split, masks, "M11", [50])
        assert "t=50 not acquired" in caplog.text

    def test_exact_timepoints_are_untouched(self, tmp_path):
        split, masks = _make_tree(tmp_path)
        frames = resolve_frames(split, masks, "M11", EARLY_TIMEPOINTS)
        assert [f.timepoint for f in frames] == list(EARLY_TIMEPOINTS)
        assert all(f.timepoint == f.requested for f in frames)

    def test_selects_only_the_requested_well(self, tmp_path):
        split, masks = _make_tree(tmp_path)
        frames = resolve_frames(split, masks, "E07", [1])
        assert "E07" in frames[0].bf_path.name
        assert "M11" not in frames[0].bf_path.name

    def test_well_id_is_case_insensitive(self, tmp_path):
        split, masks = _make_tree(tmp_path)
        assert resolve_frames(split, masks, "m11", [1])

    def test_pairs_each_frame_with_its_mask(self, tmp_path):
        split, masks = _make_tree(tmp_path)
        frames = resolve_frames(split, masks, "M11", [1, 11])
        assert all(f.mask_path is not None for f in frames)
        assert "pred_mask" in frames[0].mask_path.name

    def test_missing_mask_leaves_bf_alone_with_a_warning(self, tmp_path, caplog):
        split, masks = _make_tree(tmp_path, with_masks=False)
        with caplog.at_level(logging.WARNING):
            frames = resolve_frames(split, masks, "M11", [1])
        assert frames[0].mask_path is None
        assert "no mask" in caplog.text

    def test_collapsing_requests_are_not_duplicated(self, tmp_path, caplog):
        """Two requests snapping to the same frame yield one column, not two."""
        split, masks = _make_tree(tmp_path)
        with caplog.at_level(logging.WARNING):
            frames = resolve_frames(split, masks, "M11", [50, 52])
        assert [f.timepoint for f in frames] == [51]
        assert "already in the strip" in caplog.text

    def test_unknown_well_raises(self, tmp_path):
        split, masks = _make_tree(tmp_path)
        with pytest.raises(FileNotFoundError, match="no z0 BF frames"):
            resolve_frames(split, masks, "Z99", [1])

    def test_ignores_other_channels(self, tmp_path):
        split, masks = _make_tree(tmp_path)
        frames = resolve_frames(split, masks, "M11", [1])
        assert frames[0].bf_path.name.endswith("_BF.tif")


class TestCentreCrop:
    def test_full_field_is_returned_unchanged(self):
        arr = np.arange(16).reshape(4, 4)
        assert np.array_equal(_centre_crop(arr, 1.0), arr)

    def test_half_crop_is_centred(self):
        arr = np.arange(16).reshape(4, 4)
        cropped = _centre_crop(arr, 0.5)
        assert cropped.shape == (2, 2)
        assert np.array_equal(cropped, np.array([[5, 6], [9, 10]]))

    @pytest.mark.parametrize("fraction", [0, -0.5, 1.5])
    def test_rejects_an_out_of_range_fraction(self, fraction):
        with pytest.raises(ValueError, match="crop fraction"):
            _centre_crop(np.zeros((4, 4)), fraction)


class TestBuildFilmstrip:
    def test_writes_a_strip(self, tmp_path):
        split, masks = _make_tree(tmp_path)
        frames = resolve_frames(split, masks, "M11", DEFAULT_TIMEPOINTS)
        out = build_filmstrip(frames, tmp_path / "strip.png", title="M11")
        assert out.is_file() and out.stat().st_size > 0

    def test_single_row_mode(self, tmp_path):
        split, masks = _make_tree(tmp_path)
        frames = resolve_frames(split, masks, "M11", [1, 11])
        out = build_filmstrip(frames, tmp_path / "one_row.png", show_raw_row=False)
        assert out.is_file()

    def test_renders_with_a_crop(self, tmp_path):
        split, masks = _make_tree(tmp_path)
        frames = resolve_frames(split, masks, "M11", [1, 11])
        out = build_filmstrip(frames, tmp_path / "crop.png", crop=0.5)
        assert out.is_file()

    def test_dpi_raises_the_pixel_count(self, tmp_path):
        """Whole-field needs the pixels; 150 dpi downsamples 1024 px ~2.3x."""
        split, masks = _make_tree(tmp_path)
        frames = resolve_frames(split, masks, "M11", [1, 11])

        low = build_filmstrip(frames, tmp_path / "low.png", dpi=100)
        high = build_filmstrip(frames, tmp_path / "high.png", dpi=300)

        assert high.stat().st_size > low.stat().st_size

    def test_jpeg_destination_writes_a_jpeg(self, tmp_path):
        """Format, not size: these synthetic frames are flat, so PNG out-compresses
        JPEG on them (13.9 KB vs 51.1 KB measured). The size win is a property of real
        microscopy content and is verified on real frames, not here."""
        split, masks = _make_tree(tmp_path)
        frames = resolve_frames(split, masks, "M11", [1, 11])

        jpg = build_filmstrip(frames, tmp_path / "strip.jpg", dpi=200, quality=85)

        assert jpg.is_file() and jpg.stat().st_size > 0
        # JPEG magic bytes, so a mislabelled PNG cannot pass.
        assert jpg.read_bytes()[:3] == b"\xff\xd8\xff"

    def test_lower_quality_gives_a_smaller_jpeg(self, tmp_path):
        split, masks = _make_tree(tmp_path)
        frames = resolve_frames(split, masks, "M11", [1, 11])

        low = build_filmstrip(frames, tmp_path / "low.jpg", dpi=200, quality=20)
        high = build_filmstrip(frames, tmp_path / "high.jpg", dpi=200, quality=95)

        assert low.stat().st_size < high.stat().st_size

    def test_quality_on_a_png_destination_raises(self, tmp_path):
        """Silently ignoring it would leave a caller thinking output was compressed."""
        split, masks = _make_tree(tmp_path)
        frames = resolve_frames(split, masks, "M11", [1, 11])

        with pytest.raises(ValueError, match="not a JPEG"):
            build_filmstrip(frames, tmp_path / "strip.png", quality=85)

    def test_panel_inches_widens_the_figure(self, tmp_path):
        split, masks = _make_tree(tmp_path)
        frames = resolve_frames(split, masks, "M11", [1, 11])

        narrow = build_filmstrip(frames, tmp_path / "narrow.png", panel_inches=2.0)
        wide = build_filmstrip(frames, tmp_path / "wide.png", panel_inches=5.0)

        assert wide.stat().st_size > narrow.stat().st_size

    def test_renders_a_frame_with_no_mask(self, tmp_path):
        split, masks = _make_tree(tmp_path, with_masks=False)
        frames = resolve_frames(split, masks, "M11", [1])
        assert build_filmstrip(frames, tmp_path / "nomask.png").is_file()

    def test_annotations_are_optional(self, tmp_path):
        split, masks = _make_tree(tmp_path)
        frames = resolve_frames(split, masks, "M11", [1])
        annotations = {1: {"n_objects": 412, "coverage_fraction": 0.153}}
        assert build_filmstrip(
            frames, tmp_path / "annotated.png", annotations=annotations
        ).is_file()

    def test_creates_the_output_directory(self, tmp_path):
        split, masks = _make_tree(tmp_path)
        frames = resolve_frames(split, masks, "M11", [1])
        out = build_filmstrip(frames, tmp_path / "a" / "b" / "strip.png")
        assert out.is_file()

    def test_empty_frames_raise(self, tmp_path):
        with pytest.raises(ValueError, match="no frames"):
            build_filmstrip([], tmp_path / "empty.png")

    def test_frame_is_a_readable_record(self, tmp_path):
        split, masks = _make_tree(tmp_path)
        frames = resolve_frames(split, masks, "M11", [50])
        assert isinstance(frames[0], Frame)
        assert frames[0].timepoint == 51 and frames[0].requested == 50


class TestAvailableWells:
    def test_lists_wells_present_at_z0(self, tmp_path):
        _, masks = _make_tree(tmp_path, wells=("M11", "E07", "F08"), timepoints=[1])
        assert available_wells(masks) == ["E07", "F08", "M11"]

    def test_ignores_other_planes(self, tmp_path):
        _, masks = _make_tree(tmp_path, wells=("M11",), timepoints=[1])
        assert available_wells(masks, z_index=7) == []

    def test_missing_directory_is_empty_not_fatal(self, tmp_path):
        assert available_wells(tmp_path / "absent") == []


class TestConditionLabel:
    def test_drug_with_concentration(self):
        row = {"drug": "Navitoclax", "concentration_uM": 75.0, "is_dmso": False}
        assert condition_label("E07", row) == "Navitoclax 75 µM"

    def test_trims_trailing_zeros(self):
        row = {"drug": "Doxorubicin", "concentration_uM": 0.001, "is_dmso": False}
        assert condition_label("D10", row) == "Doxorubicin 0.001 µM"

    def test_dmso_flag_wins_over_the_control_drug_string(self):
        """Control wells read drug == 'control'; the vehicle identity is is_dmso."""
        row = {"drug": "control", "concentration_uM": None, "is_dmso": True}
        assert condition_label("N11", row) == "DMSO"

    def test_drug_without_a_concentration(self):
        row = {"drug": "Navitoclax", "concentration_uM": None, "is_dmso": False}
        assert condition_label("E07", row) == "Navitoclax"

    def test_nan_concentration_is_not_printed(self):
        row = {"drug": "Selinexor", "concentration_uM": float("nan"), "is_dmso": False}
        assert condition_label("G07", row) == "Selinexor"

    def test_falls_back_to_the_well_id_without_an_annotation(self):
        assert condition_label("E07", None) == "E07"
        assert condition_label("E07", {}) == "E07"


class TestOutlineStyle:
    def test_outline_mode_renders(self, tmp_path):
        split, masks = _make_tree(tmp_path)
        frames = resolve_frames(split, masks, "M11", [1, 11])
        out = build_filmstrip(
            frames,
            tmp_path / "outline.png",
            mask_style=MASK_STYLE_OUTLINE,
            base_cmap="inferno",
        )
        assert out.is_file() and out.stat().st_size > 0

    def test_rejects_an_unknown_mask_style(self, tmp_path):
        split, masks = _make_tree(tmp_path)
        frames = resolve_frames(split, masks, "M11", [1])
        with pytest.raises(ValueError, match="mask_style"):
            build_filmstrip(frames, tmp_path / "x.png", mask_style="dotted")

    def test_outline_mode_tolerates_a_missing_mask(self, tmp_path):
        split, masks = _make_tree(tmp_path, with_masks=False)
        frames = resolve_frames(split, masks, "M11", [1])
        out = build_filmstrip(
            frames, tmp_path / "nomask.png", mask_style=MASK_STYLE_OUTLINE
        )
        assert out.is_file()

    def test_outline_marks_boundaries_not_interiors(self):
        """A filled label would paint every pixel; an outline paints only its border."""
        mask = np.zeros((12, 12), dtype=np.uint16)
        mask[4:8, 4:8] = 1
        rgba = _outline_rgba(mask)
        assert rgba[..., 3].sum() > 0, "expected some boundary pixels"
        assert rgba[5, 5, 3] == 0, "object interior must stay clear"
        assert rgba[0, 0, 3] == 0, "background must stay clear"

    def test_outline_separates_touching_objects(self):
        mask = np.zeros((12, 12), dtype=np.uint16)
        mask[2:10, 2:6] = 1
        mask[2:10, 6:10] = 2  # abuts label 1 with no gap
        rgba = _outline_rgba(mask)
        # the shared edge must carry boundary pixels for both objects
        assert rgba[5, 5:7, 3].sum() > 0


class TestChannelSelection:
    def test_resolve_frames_can_pick_mcherry(self, tmp_path):
        split, masks = _make_tree(tmp_path)
        frames = resolve_frames(split, masks, "M11", [1], channel="mCherry")
        assert frames[0].bf_path.name.endswith("_mCherry.tif")
        assert frames[0].mask_path is not None

    def test_unknown_channel_raises(self, tmp_path):
        split, masks = _make_tree(tmp_path)
        with pytest.raises(FileNotFoundError, match="Cy5"):
            resolve_frames(split, masks, "M11", [1], channel="Cy5")


class TestConditionLabelNaN:
    def test_nan_drug_falls_back_to_the_well_id(self):
        """pandas leaves an unannotated well as NaN, and bool(nan) is True."""
        row = {"drug": float("nan"), "concentration_uM": float("nan"), "is_dmso": False}
        assert condition_label("E07", row) == "E07"

    def test_nan_drug_is_not_rendered_as_the_string_nan(self):
        row = {"drug": float("nan"), "concentration_uM": 5.0, "is_dmso": False}
        assert "nan" not in condition_label("E07", row)


def test_one_frame_index_is_ten_minutes():
    """Owner decision 2026-09-21: frame 351 is 58.3 h, not 72 h."""
    from src.visualize.filmstrip import MINUTES_PER_INDEX, TI_TO_HOURS

    assert MINUTES_PER_INDEX == 10.0
    assert (351 - 1) * TI_TO_HOURS == pytest.approx(58.333, abs=0.01)


class TestCopilotEdgeCases:
    def test_nan_is_dmso_is_not_read_as_true(self):
        """bool(nan) is True, which would label every unannotated well DMSO."""
        row = {"drug": "Navitoclax", "concentration_uM": 75.0, "is_dmso": float("nan")}
        assert condition_label("E07", row) == "Navitoclax 75 µM"

    def test_explicit_true_is_still_dmso(self):
        row = {"drug": "control", "concentration_uM": None, "is_dmso": True}
        assert condition_label("N11", row) == "DMSO"

    def test_string_true_from_csv_is_dmso(self):
        row = {"drug": "control", "concentration_uM": None, "is_dmso": "True"}
        assert condition_label("N11", row) == "DMSO"

    def test_tiny_crop_keeps_at_least_one_pixel(self):
        """0.01 of a 16px axis truncates to 0, and an empty array breaks display."""
        cropped = _centre_crop(np.arange(256).reshape(16, 16), 0.01)
        assert cropped.shape == (1, 1)
        assert cropped.size > 0

    def test_tiny_crop_renders_without_error(self, tmp_path):
        split, masks = _make_tree(tmp_path)
        frames = resolve_frames(split, masks, "M11", [1])
        assert build_filmstrip(frames, tmp_path / "tiny.png", crop=0.01).is_file()
