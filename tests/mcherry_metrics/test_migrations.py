"""Tests for the label_id -> cell_id CSV header migration helper."""

import pandas as pd
import pytest

from src.mcherry_metrics.io.migrations import migrate_label_id_header


def _write_csv(path, header):
    rows = ",".join(header)
    path.write_text(f"{rows}\nfoo.tif,mask.tif,1,42\nfoo.tif,mask.tif,2,17\n")


def test_migrates_label_id_header(tmp_path):
    csv_path = tmp_path / "metrics.csv"
    _write_csv(csv_path, ["image_path", "label_path", "label_id", "area"])

    assert migrate_label_id_header(csv_path, dry_run=False) is True

    df = pd.read_csv(csv_path)
    assert "cell_id" in df.columns
    assert "label_id" not in df.columns
    # values are unchanged
    assert df["cell_id"].tolist() == [1, 2]
    assert df["area"].tolist() == [42, 17]


def test_dry_run_reports_but_does_not_write(tmp_path):
    csv_path = tmp_path / "metrics.csv"
    _write_csv(csv_path, ["image_path", "label_path", "label_id", "area"])

    assert migrate_label_id_header(csv_path, dry_run=True) is True
    # header untouched on dry-run
    assert "label_id" in pd.read_csv(csv_path).columns


def test_idempotent_on_already_migrated(tmp_path):
    csv_path = tmp_path / "metrics.csv"
    _write_csv(csv_path, ["image_path", "label_path", "cell_id", "area"])

    assert migrate_label_id_header(csv_path, dry_run=False) is False
    assert "cell_id" in pd.read_csv(csv_path).columns


def test_raises_when_both_columns_present(tmp_path):
    csv_path = tmp_path / "metrics.csv"
    csv_path.write_text("label_id,cell_id,area\n1,1,42\n")

    with pytest.raises(ValueError, match="both"):
        migrate_label_id_header(csv_path, dry_run=False)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        migrate_label_id_header(tmp_path / "nope.csv", dry_run=True)
