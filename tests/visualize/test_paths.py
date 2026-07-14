"""Unit tests for src.visualize.paths.resolve_related_paths."""

from pathlib import Path

import numpy as np
import pytest

from src.inference.output_manager import OutputManager
from src.visualize.paths import resolve_related_paths


@pytest.fixture
def masks_2d():
    rng = np.random.default_rng(0)
    return rng.integers(0, 10, (16, 16), dtype=np.uint16)


@pytest.fixture
def output_manager(tmp_path):
    return OutputManager(
        base_output_dir=tmp_path / "results",
        model_name="model",
        dataset_name="test",
        label_format="tif",
    )


class TestResolveRelatedPaths:
    def test_all_siblings_present(self, tmp_path, output_manager, masks_2d):
        bf_path = tmp_path / "p2126_A01_t1_z1_BF.tif"
        mcherry_path = tmp_path / "p2126_A01_t1_z1_mCherry.tif"
        bf_path.touch()
        mcherry_path.touch()

        output_manager.save_prediction(
            masks=masks_2d,
            metadata={"num_cells": 5, "parameters": {}},
            input_path=bf_path,
            save_overlay=False,
        )

        result = resolve_related_paths(bf_path, output_manager)

        assert result["bf"] == bf_path
        assert result["mcherry"] == mcherry_path
        assert result["mask"] == output_manager.expected_mask_path(bf_path)
        assert result["mask"].exists()

    def test_missing_mcherry_returns_none(self, tmp_path, output_manager):
        bf_path = tmp_path / "p2126_A01_t1_z1_BF.tif"
        bf_path.touch()

        result = resolve_related_paths(bf_path, output_manager)

        assert result["mcherry"] is None

    def test_missing_mask_returns_none(self, tmp_path, output_manager):
        bf_path = tmp_path / "p2126_A01_t1_z1_BF.tif"
        bf_path.touch()

        result = resolve_related_paths(bf_path, output_manager)

        assert result["mask"] is None

    def test_stem_without_bf_suffix_does_not_crash(self, tmp_path, output_manager):
        bf_path = tmp_path / "some_other_name.tif"
        bf_path.touch()

        result = resolve_related_paths(bf_path, output_manager)

        assert result["bf"] == bf_path
        assert result["mcherry"] is None

    def test_custom_suffixes(self, tmp_path, output_manager):
        bf_path = tmp_path / "p2126_A01_t1_z1_Bright.tif"
        rfp_path = tmp_path / "p2126_A01_t1_z1_RFP.tif"
        bf_path.touch()
        rfp_path.touch()

        result = resolve_related_paths(
            bf_path,
            output_manager,
            bf_suffix="_Bright",
            mcherry_suffix="_RFP",
        )

        assert result["mcherry"] == rfp_path
