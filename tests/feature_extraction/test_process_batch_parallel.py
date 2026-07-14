"""Tests for file-level parallelization of ``FeatureExtractionPipeline.process_batch``.

Covers the refactor that fans the outer file loop out across worker processes
(``--n-jobs`` = number of concurrent files) while forcing inner per-cell
parallelism to 1:

- sequential (n_jobs=1) and parallel (n_jobs>1) produce identical results,
- individual per-image CSVs are still written under parallel execution,
- worker error records are aggregated back into ``pipeline.error_files``,
- scPortrait is always routed to the sequential path (kept on one GPU).
"""

import numpy as np
import pandas as pd
import cv2
from unittest.mock import MagicMock

from src.feature_extraction.feature_extraction_pipeline import FeatureExtractionPipeline


def _make_dataset(tmp_path, n_images=4):
    """Create ``n_images`` synthetic BF images + two-cell masks; return dirs."""
    img_dir = tmp_path / "imgs"
    msk_dir = tmp_path / "msks"
    img_dir.mkdir()
    msk_dir.mkdir()

    rng = np.random.RandomState(0)
    for i in range(n_images):
        image = (rng.rand(64, 64) * 255).astype(np.uint8)
        mask = np.zeros((64, 64), dtype=np.uint16)
        mask[10:25, 10:25] = 1
        mask[30:50, 30:55] = 2
        cv2.imwrite(str(img_dir / f"s{i}_BF.tif"), image)
        cv2.imwrite(str(msk_dir / f"s{i}_pred_mask.tif"), mask)

    return img_dir, msk_dir


def _pipeline(tmp_path, n_jobs, save_individual=False, subdir="out"):
    return FeatureExtractionPipeline(
        config={
            "method": "regionprops",
            "n_jobs": n_jobs,
            "output": {
                "save_individual_files": save_individual,
                "create_subdirs": False,
            },
        },
        output_dir=str(tmp_path / subdir),
    )


def _run(pipeline, img_dir, msk_dir):
    return pipeline.process_batch(
        img_dir,
        msk_dir,
        image_patterns=["*_BF.tif"],
        mask_patterns=["*_pred_mask.tif"],
    )


def _canon(df):
    """Sort rows/columns so two runs can be compared regardless of ordering."""
    cols = sorted(df.columns)
    return df[cols].sort_values(cols).reset_index(drop=True)


class TestSequentialVsParallelEquivalence:
    def test_same_results(self, tmp_path):
        img_dir, msk_dir = _make_dataset(tmp_path, n_images=4)

        seq = _run(_pipeline(tmp_path, n_jobs=1, subdir="seq"), img_dir, msk_dir)
        par = _run(_pipeline(tmp_path, n_jobs=2, subdir="par"), img_dir, msk_dir)

        assert not seq.empty
        assert seq.shape == par.shape
        # 4 images * 2 cells each
        assert len(seq) == 8
        pd.testing.assert_frame_equal(_canon(seq), _canon(par))


class TestIndividualFileSaving:
    def test_parallel_writes_one_csv_per_image(self, tmp_path):
        img_dir, msk_dir = _make_dataset(tmp_path, n_images=3)
        pipeline = _pipeline(tmp_path, n_jobs=2, save_individual=True, subdir="indiv")
        _run(pipeline, img_dir, msk_dir)

        csvs = sorted(pipeline.output_dir.glob("*_features.csv"))
        assert len(csvs) == 3


class TestErrorAggregation:
    def test_worker_errors_returned_to_parent(self, tmp_path):
        img_dir, msk_dir = _make_dataset(tmp_path, n_images=2)
        pipeline = _pipeline(tmp_path, n_jobs=2, subdir="err")

        good_img = img_dir / "s0_BF.tif"
        good_mask = msk_dir / "s0_pred_mask.tif"
        missing_mask = msk_dir / "does_not_exist_pred_mask.tif"

        pairs = [(good_img, good_mask), (good_img, missing_mask)]
        all_features, processed = pipeline._process_pairs_parallel(pairs, n_workers=2)

        # Only the valid pair yields features; the batch does not crash.
        assert processed == 1
        assert len(all_features) == 1
        # The missing-mask error made it back to the parent process.
        assert any("does_not_exist" in str(path) for path, _ in pipeline.error_files)


class TestWorkerResolution:
    def test_zero_and_none_mean_sequential(self, tmp_path):
        assert _pipeline(tmp_path, n_jobs=0)._resolve_file_workers() == 1
        assert _pipeline(tmp_path, n_jobs=None)._resolve_file_workers() == 1

    def test_positive_passthrough(self, tmp_path):
        assert _pipeline(tmp_path, n_jobs=3)._resolve_file_workers() == 3

    def test_negative_means_all_cores(self, tmp_path):
        assert _pipeline(tmp_path, n_jobs=-1)._resolve_file_workers() >= 1


class TestScportraitStaysSequential:
    def test_scportrait_routes_to_sequential(self, tmp_path):
        img_dir, msk_dir = _make_dataset(tmp_path, n_images=2)
        pipeline = FeatureExtractionPipeline(
            config={
                "method": "scportrait",
                "n_jobs": 4,  # would trigger parallel for a CPU method
                "output": {"save_individual_files": False},
            },
            output_dir=str(tmp_path / "sc"),
        )
        seq_mock = MagicMock(return_value=([], 0))
        par_mock = MagicMock(return_value=([], 0))
        pipeline._process_pairs_sequential = seq_mock
        pipeline._process_pairs_parallel = par_mock

        _run(pipeline, img_dir, msk_dir)

        seq_mock.assert_called_once()
        par_mock.assert_not_called()
