"""Tests for the representative-slice-per-cell core module (Phase 2).

Synthetic only (no real data, no GPU). Covers the interior sharpness measure, the
blur-gate/max-area chooser, the leakage guard, trackpy short-cell survival, per-z mask
writing (one z per cell, label = cell_id, z = index + offset), and an end-to-end run.
"""

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

from src.postprocessing.representative_slice import (
    SELECTION_COLUMNS,
    _interior_sharpness,
    apply_blur_filter,
    compute_slice_metrics,
    link_slices,
    run,
    select_representative,
    write_all_slice_masks,
    write_filtered_masks,
)
from src.utils.config_schemas import RepresentativeSliceConfig
from src.utils.image_utils import load_labels


# ─── _interior_sharpness ─────────────────────────────────────────────────────
class TestInteriorSharpness:
    def test_in_focus_scores_higher_than_blurred(self):
        rng = np.random.default_rng(0)
        mask = np.ones((12, 12), dtype=bool)
        sharp = rng.integers(0, 1000, size=(12, 12)).astype(np.float64)  # high-freq
        smooth = np.tile(np.linspace(0, 1000, 12), (12, 1))  # smooth gradient
        sharp_score = _interior_sharpness(sharp, mask, 1)
        smooth_score = _interior_sharpness(smooth, mask, 1)
        assert sharp_score > smooth_score

    def test_erosion_excludes_bright_mask_edge(self):
        # flat interior, a bright 1px ring at the border -> the edge dominates var
        crop = np.zeros((12, 12), dtype=np.float64)
        crop[0, :] = crop[-1, :] = crop[:, 0] = crop[:, -1] = 5000.0
        mask = np.ones((12, 12), dtype=bool)
        with_erosion = _interior_sharpness(crop, mask, 2)
        without_erosion = _interior_sharpness(crop, mask, 0)
        assert with_erosion < without_erosion
        assert with_erosion == pytest.approx(0.0)  # flat interior after erosion


# ─── select_representative ───────────────────────────────────────────────────
def _cfg(**kw) -> RepresentativeSliceConfig:
    return RepresentativeSliceConfig(**kw)


class TestSelectRepresentative:
    def test_gate_then_max_area(self):
        # cell 1 over z=5..10: sharpest+largest-among-gated is z=7; z=10 is a large
        # but off-focus cap that the gate (0.7) must exclude.
        df = pd.DataFrame(
            {
                "cell_id": [1, 1, 1, 1, 1, 1],
                "z_stack_index": [5, 6, 7, 8, 9, 10],
                "x": [0.0] * 6,
                "y": [0.0] * 6,
                "area": [100, 150, 200, 180, 90, 300],
                "sharpness": [0.5, 0.8, 1.0, 0.9, 0.4, 0.3],
            }
        )
        out = select_representative(df, _cfg(sharpness_gate_fraction=0.7))
        chosen = out[out["selected"]]
        assert len(chosen) == 1
        assert int(chosen.iloc[0]["z_stack_index"]) == 7

    def test_single_slice_cell_selected(self):
        df = pd.DataFrame(
            {
                "cell_id": [2],
                "z_stack_index": [3],
                "x": [0.0],
                "y": [0.0],
                "area": [50],
                "sharpness": [0.9],
            }
        )
        out = select_representative(df, _cfg())
        assert out["selected"].tolist() == [True]

    def test_area_tie_broken_by_sharpness(self):
        df = pd.DataFrame(
            {
                "cell_id": [3, 3],
                "z_stack_index": [4, 5],
                "x": [0.0, 0.0],
                "y": [0.0, 0.0],
                "area": [200, 200],  # tie on area
                "sharpness": [0.8, 1.0],  # z=5 sharper
            }
        )
        out = select_representative(df, _cfg(sharpness_gate_fraction=0.7))
        chosen = out[out["selected"]]
        assert int(chosen.iloc[0]["z_stack_index"]) == 5

    def test_empty_frame(self):
        df = pd.DataFrame(
            columns=["cell_id", "z_stack_index", "x", "y", "area", "sharpness"]
        )
        out = select_representative(df, _cfg())
        assert "selected" in out.columns
        assert out.empty

    def test_sharpness_metric_picks_sharpest_not_largest(self):
        # both slices pass the gate; area-chooser would pick z=4 (larger), but the
        # sharpness chooser picks z=5 (sharper). Same frame, opposite winners.
        df = pd.DataFrame(
            {
                "cell_id": [7, 7],
                "z_stack_index": [4, 5],
                "x": [0.0, 0.0],
                "y": [0.0, 0.0],
                "area": [200, 150],
                "sharpness": [0.8, 1.0],
            }
        )
        by_area = select_representative(df, _cfg(selection_metric="area"))
        by_sharp = select_representative(df, _cfg(selection_metric="sharpness"))
        assert int(by_area[by_area["selected"]].iloc[0]["z_stack_index"]) == 4
        assert int(by_sharp[by_sharp["selected"]].iloc[0]["z_stack_index"]) == 5

    def test_sharpness_tie_broken_by_area(self):
        df = pd.DataFrame(
            {
                "cell_id": [8, 8],
                "z_stack_index": [4, 5],
                "x": [0.0, 0.0],
                "y": [0.0, 0.0],
                "area": [200, 150],  # z=4 larger
                "sharpness": [1.0, 1.0],  # tie on sharpness
            }
        )
        out = select_representative(df, _cfg(selection_metric="sharpness"))
        chosen = out[out["selected"]]
        assert int(chosen.iloc[0]["z_stack_index"]) == 4


# ─── compute_slice_metrics — leakage guard ───────────────────────────────────
class TestComputeSliceMetricsLeakage:
    def test_signature_takes_only_bf(self):
        params = list(inspect.signature(compute_slice_metrics).parameters)
        assert params == ["tracked_stack_3d", "bf_stack_3d", "config"]
        assert not any(
            tok in p.lower() for p in params for tok in ("mcherry", "target")
        )

    def test_metrics_depend_only_on_the_given_bf(self):
        tracked = np.zeros((2, 20, 20), dtype=np.int32)
        tracked[0, 4:14, 4:14] = 1
        tracked[1, 4:14, 4:14] = 1
        rng = np.random.default_rng(1)
        bf_a = rng.integers(0, 1000, size=(2, 20, 20)).astype(np.uint16)
        bf_b = np.zeros((2, 20, 20), dtype=np.uint16)  # flat -> zero sharpness
        cfg = _cfg()
        ma = compute_slice_metrics(tracked, bf_a, cfg)
        mb = compute_slice_metrics(tracked, bf_b, cfg)
        assert ma["sharpness"].sum() > 0
        assert mb["sharpness"].sum() == pytest.approx(0.0)


# ─── link_slices — short-cell survival (min_track_length=1) ───────────────────
class TestLinkSlices:
    def test_single_and_two_slice_cells_survive(self):
        stack = np.zeros((3, 40, 40), dtype=np.int32)
        # cell A: z0 and z1 at same location (two-slice)
        stack[0, 5:11, 5:11] = 1
        stack[1, 5:11, 5:11] = 1
        # cell B: z0 only (single-slice)
        stack[0, 25:31, 25:31] = 2
        tracked = link_slices(stack, _cfg())
        labels = set(np.unique(tracked)) - {0}
        assert len(labels) == 2  # both cells kept, none dropped as a "stub"
        # the single-slice cell is present at exactly one z
        z_counts = [int((tracked[z] > 0).any()) for z in range(3)]
        assert z_counts[0] == 1 and z_counts[2] == 0


# ─── apply_blur_filter (pre-link absolute blur-map filter) ───────────────────
class TestApplyBlurFilter:
    def test_drops_blurry_cell_keeps_sharp_cell(self):
        # two cells; blur heatmap high (blurry) over cell 1, low (sharp) over cell 2.
        masks = np.zeros((2, 30, 30), dtype=np.int32)
        masks[0, 2:8, 2:8] = 1  # cell 1
        masks[0, 20:26, 20:26] = 2  # cell 2
        masks[1, 2:8, 2:8] = 1  # cell 1 also on z1
        heatmap = np.zeros((2, 30, 30), dtype=np.float64)
        heatmap[:, 2:8, 2:8] = 0.9  # cell 1: blurry (mean 0.9 >= 0.5 -> drop)
        heatmap[:, 20:26, 20:26] = 0.1  # cell 2: sharp (mean 0.1 < 0.5 -> keep)
        cfg = _cfg(blur_threshold=0.5, blur_invert_threshold=False)
        out = apply_blur_filter(masks, heatmap, cfg)
        # cell 1 removed everywhere, cell 2 retained
        assert not (out == 1).any()
        assert (out == 2).any()

    def test_shape_mismatch_raises(self):
        masks = np.zeros((2, 30, 30), dtype=np.int32)
        heatmap = np.zeros((3, 30, 30), dtype=np.float64)
        with pytest.raises(ValueError, match="shape mismatch"):
            apply_blur_filter(masks, heatmap, _cfg())


# ─── write_filtered_masks ────────────────────────────────────────────────────
class TestWriteFilteredMasks:
    def test_one_z_per_cell_with_offset(self, tmp_path):
        tracked = np.zeros((3, 20, 20), dtype=np.int32)
        tracked[0, 2:8, 2:8] = 1
        tracked[1, 2:8, 2:8] = 1  # cell 1 present at z0 and z1
        tracked[0, 12:18, 12:18] = 2  # cell 2 at z0
        selection = pd.DataFrame(
            {
                "cell_id": [1, 1, 2],
                "z_stack_index": [0, 1, 0],
                "selected": [False, True, True],  # cell1 -> z1, cell2 -> z0
            }
        )
        cfg = _cfg(
            output_dir=str(tmp_path),
            output_label_format="tif",
            z_index_offset=1,
        )
        written = write_filtered_masks(tracked, selection, "pMF5V1_A01_t1", cfg)
        assert len(written) == 3  # one file per plane
        final_2d = tmp_path / "final_2d"
        # z0 -> file _z1: only cell 2
        z1 = load_labels(final_2d / "pMF5V1_A01_t1_z1_pred_mask.tif")
        assert set(np.unique(z1)) == {0, 2}
        # z1 -> file _z2: only cell 1
        z2 = load_labels(final_2d / "pMF5V1_A01_t1_z2_pred_mask.tif")
        assert set(np.unique(z2)) == {0, 1}
        # z2 -> file _z3: empty
        z3 = load_labels(final_2d / "pMF5V1_A01_t1_z3_pred_mask.tif")
        assert set(np.unique(z3)) == {0}


# ─── write_all_slice_masks (diagnostic per-slice mode) ───────────────────────
class TestWriteAllSliceMasks:
    def test_every_cell_at_all_its_z(self, tmp_path):
        tracked = np.zeros((3, 20, 20), dtype=np.int32)
        tracked[0, 2:8, 2:8] = 1
        tracked[1, 2:8, 2:8] = 1  # cell 1 at z0 AND z1
        tracked[0, 12:18, 12:18] = 2  # cell 2 at z0 only
        cfg = _cfg(
            output_dir=str(tmp_path), output_label_format="tif", z_index_offset=1
        )
        written = write_all_slice_masks(tracked, "pMF5V1_A01_t1", cfg)
        assert len(written) == 3
        final_2d = tmp_path / "final_2d"
        z1 = load_labels(final_2d / "pMF5V1_A01_t1_z1_pred_mask.tif")  # stack z0
        z2 = load_labels(final_2d / "pMF5V1_A01_t1_z2_pred_mask.tif")  # stack z1
        # cell 1 present at BOTH z1 and z2 (unlike the selected-only writer)
        assert set(np.unique(z1)) == {0, 1, 2}
        assert set(np.unique(z2)) == {0, 1}


# ─── run — tiny end-to-end ───────────────────────────────────────────────────
def _write_synthetic_stack(masks_dir: Path, bf_dir: Path, well: str) -> None:
    stack = np.zeros((3, 30, 30), dtype=np.uint16)
    stack[0, 4:10, 4:10] = 1
    stack[1, 4:10, 4:10] = 1  # cell 1 spans z0, z1
    stack[0, 20:26, 20:26] = 2  # cell 2 at z0 only
    tifffile.imwrite(masks_dir / f"pMF5V1_{well}_t1_pred_mask_3d.tif", stack)
    rng = np.random.default_rng(abs(hash(well)) % (2**32))
    bf = rng.integers(0, 4000, size=(3, 30, 30)).astype(np.uint16)
    tifffile.imwrite(bf_dir / f"pMF5V1_{well}_t1_BF_3d.tif", bf)


class TestRunEndToEnd:
    def test_two_stacks(self, tmp_path):
        masks_dir = tmp_path / "masks_3d"
        bf_dir = tmp_path / "3d_data"
        out_dir = tmp_path / "inference_unfiltered_area"
        masks_dir.mkdir()
        bf_dir.mkdir()
        for well in ("A01", "A02"):
            _write_synthetic_stack(masks_dir, bf_dir, well)

        cfg = _cfg(
            input_masks_dir=str(masks_dir),
            bf_3d_dir=str(bf_dir),
            output_dir=str(out_dir),
            mask_pattern="*_pred_mask_3d.tif",
            output_label_format="tif",
            n_jobs=1,
        )
        combined = run(cfg)

        assert (out_dir / "selection.csv").exists()
        assert list(combined.columns) == SELECTION_COLUMNS
        # 2 cells/stack x 2 stacks selected; 3 candidate rows/stack
        assert int(combined["selected"].sum()) == 4
        assert len(combined) == 6
        assert set(combined["sample_id"]) == {"A01", "A02"}
        assert set(combined["timepoint"]) == {"1"}
        # z-labels are stack index + offset (1) -> {1, 2, 3}
        assert set(combined["z_index"]) <= {"1", "2", "3"}
        # each (sample_id, cell_id) has exactly one selected slice
        sel = combined[combined["selected"]]
        assert (sel.groupby(["sample_id", "cell_id"]).size() == 1).all()
