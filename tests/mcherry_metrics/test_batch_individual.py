"""Tests for per-image mCherry CSV output (save_individual_files flag)."""

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.mcherry_metrics.config import ExtractionConfig
from src.mcherry_metrics.core.batch import MetricsExtractor
from src.mcherry_metrics.core.extractor import InstanceMetricsExtractor
from src.mcherry_metrics.io.exporters import INSTANCE_METRICS_COLUMNS


def _fake_metrics_row(image_name: str) -> pd.DataFrame:
    """A minimal finalized per-image metrics frame satisfying the contract."""
    row = {column: 0 for column in INSTANCE_METRICS_COLUMNS}
    row["image"] = image_name
    row["cell_id"] = 1
    return pd.DataFrame([row])


def _extractor() -> InstanceMetricsExtractor:
    # n_jobs=1 keeps process_single_image in-process so it can be patched.
    return InstanceMetricsExtractor(config=ExtractionConfig(n_jobs=1))


def test_process_batch_writes_per_image_csv(tmp_path):
    ex = _extractor()
    imgs = [
        tmp_path / "pMF5V1_C09_t101_z1_mCherry.tif",
        tmp_path / "pMF5V1_C09_t101_z2_mCherry.tif",
    ]
    lbls = [tmp_path / "m1.tif", tmp_path / "m2.tif"]

    def fake_single(image_path, _label_path):
        return None, _fake_metrics_row(Path(image_path).name)

    out = tmp_path / "split_data"
    with patch.object(ex, "process_single_image", side_effect=fake_single):
        combined = ex.process_batch_images(imgs, lbls, individual_output_dir=out)

    f1 = out / "pMF5V1_C09_t101_z1_mCherry_metrics.csv"
    f2 = out / "pMF5V1_C09_t101_z2_mCherry_metrics.csv"
    assert f1.exists() and f2.exists()
    # per-image CSV carries the contract columns and one row
    per_image = pd.read_csv(f1)
    assert "cell_id" in per_image.columns
    assert len(per_image) == 1
    # combined table still returned with both images
    assert len(combined) == 2


def test_process_batch_no_individual_dir_writes_nothing(tmp_path):
    ex = _extractor()
    imgs = [tmp_path / "pMF5V1_C09_t101_z1_mCherry.tif"]
    lbls = [tmp_path / "m1.tif"]

    def fake_single(image_path, _label_path):
        return None, _fake_metrics_row(Path(image_path).name)

    with patch.object(ex, "process_single_image", side_effect=fake_single):
        ex.process_batch_images(imgs, lbls, individual_output_dir=None)

    assert list(tmp_path.glob("*_metrics.csv")) == []


@pytest.mark.parametrize("save_individual", [True, False])
def test_run_passes_individual_dir_per_flag(tmp_path, save_individual):
    cfg = ExtractionConfig(n_jobs=1, save_individual_files=save_individual)
    extractor = MetricsExtractor(config=cfg)
    output_csv = tmp_path / "instance_metrics.csv"

    with patch.object(
        extractor.instance_extractor,
        "process_batch_images",
        return_value=pd.DataFrame(),
    ) as mock_batch:
        extractor.run(
            mcherry_paths=[tmp_path / "a_mCherry.tif"],
            mask_paths=[tmp_path / "a.tif"],
            output_csv=output_csv,
        )

    kwargs = mock_batch.call_args.kwargs
    if save_individual:
        assert kwargs["individual_output_dir"] == tmp_path / "split_data"
    else:
        assert kwargs["individual_output_dir"] is None
