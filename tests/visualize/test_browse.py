"""Unit tests for src.visualize.browse."""

import numpy as np
import pytest
import tifffile

from src.inference.output_manager import OutputManager
from src.utils.file_utils import ConfigurableFileHandler
from src.visualize.browse import build_index, resolve_selection
from src.visualize.paths import resolve_related_paths


@pytest.fixture
def file_handler():
    return ConfigurableFileHandler()


@pytest.fixture
def output_manager(tmp_path):
    return OutputManager(
        base_output_dir=tmp_path / "results",
        model_name="model",
        dataset_name="test",
        label_format="tif",
    )


@pytest.fixture
def plate_dir(tmp_path, output_manager):
    """2 wells x 2 timepoints of BF+mCherry; only one file has a mask written."""
    data_dir = tmp_path / "raw"
    data_dir.mkdir()

    bf_array = np.zeros((4, 4), dtype=np.uint8)
    mcherry_array = np.zeros((4, 4), dtype=np.uint16)
    mask_array = np.zeros((4, 4), dtype=np.uint16)

    wells = ["A01", "A02"]
    timepoints = ["1", "2"]

    for well in wells:
        for t in timepoints:
            stem = f"p2126_{well}_t{t}_z1"
            bf_path = data_dir / f"{stem}_BF.tif"
            mcherry_path = data_dir / f"{stem}_mCherry.tif"
            tifffile.imwrite(bf_path, bf_array)
            tifffile.imwrite(mcherry_path, mcherry_array)

    # only well A01, timepoint 1 has an inference mask
    output_manager.save_prediction(
        masks=mask_array,
        metadata={"num_cells": 0, "parameters": {}},
        input_path=data_dir / "p2126_A01_t1_z1_BF.tif",
        save_overlay=False,
    )

    return data_dir


class TestBuildIndex:
    def test_wells_and_timepoints(self, plate_dir, file_handler):
        index = build_index(plate_dir, file_handler)
        assert index["wells"] == ["A01", "A02"]
        assert index["timepoints"] == ["1", "2"]

    def test_by_well_time_resolves_correct_file(self, plate_dir, file_handler):
        index = build_index(plate_dir, file_handler)
        bf_path = index["by_well_time"][("A01", "1")]
        assert bf_path.name == "p2126_A01_t1_z1_BF.tif"

    def test_groups_match_unique_id(self, plate_dir, file_handler):
        index = build_index(plate_dir, file_handler)
        for bf_path in plate_dir.glob("*_BF.tif"):
            unique_id = file_handler.extract_unique_id(bf_path.name)
            assert unique_id in index["groups"]
            assert str(bf_path) in index["groups"][unique_id]


class TestResolveSelection:
    def test_known_selection_matches_direct_resolution(
        self, plate_dir, file_handler, output_manager
    ):
        index = build_index(plate_dir, file_handler)
        bf_path = plate_dir / "p2126_A01_t1_z1_BF.tif"

        expected = resolve_related_paths(bf_path, output_manager)
        result = resolve_selection(index, output_manager, "A01", "1")

        assert result == expected
        assert result["mask"] is not None

    def test_selection_without_mask(self, plate_dir, file_handler, output_manager):
        index = build_index(plate_dir, file_handler)
        result = resolve_selection(index, output_manager, "A02", "2")

        assert result["bf"] is not None
        assert result["mcherry"] is not None
        assert result["mask"] is None

    def test_unknown_selection_returns_all_none(
        self, plate_dir, file_handler, output_manager
    ):
        index = build_index(plate_dir, file_handler)
        result = resolve_selection(index, output_manager, "Z99", "99")

        assert result == {"bf": None, "mcherry": None, "mask": None}


class TestBuildBrowserWidget:
    pytest.importorskip("ipywidgets")

    def test_dropdown_options_match_index(
        self, plate_dir, file_handler, output_manager
    ):
        from src.visualize.browse import build_browser_widget

        index = build_index(plate_dir, file_handler)
        widget = build_browser_widget(index, output_manager)
        well_dropdown, timepoint_dropdown, output = widget.children

        assert list(well_dropdown.options) == index["wells"]
        assert list(timepoint_dropdown.options) == index["timepoints"]

    def test_changing_selection_redraws_without_raising(
        self, plate_dir, file_handler, output_manager
    ):
        # Output.outputs only captures display() calls inside a live Jupyter kernel;
        # under plain pytest, display() falls through to stdout instead. So this only
        # asserts the redraw wiring doesn't raise on selection change, per the plan's
        # "assert on resolved state, not rendered content" guidance.
        from src.visualize.browse import build_browser_widget

        index = build_index(plate_dir, file_handler)
        widget = build_browser_widget(index, output_manager)
        well_dropdown, timepoint_dropdown, _output = widget.children

        well_dropdown.value = "A02"
        timepoint_dropdown.value = "2"
