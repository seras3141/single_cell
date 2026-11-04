"""Tests for feature_to_mcherry.provenance."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.feature_to_mcherry.provenance import write_run_provenance


@dataclass
class _Config:
    output_dir: str
    alpha: float = 1.0


def test_provenance_records_the_config_commit_and_job(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "12345")

    path = write_run_provenance(_Config(output_dir="x"), tmp_path / "out")

    record = json.loads(path.read_text())
    assert record["config"] == {"output_dir": "x", "alpha": 1.0}
    assert record["slurm_job_id"] == "12345"
    assert len(record["git_commit"]) == 40
    assert isinstance(record["git_dirty"], bool)
    assert record["written_at"]


def test_provenance_without_slurm_or_git_still_writes(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(
        "src.feature_to_mcherry.provenance._git_commit", lambda repo_root: None
    )

    record = json.loads(write_run_provenance(_Config("x"), tmp_path).read_text())

    assert record["slurm_job_id"] is None
    assert record["git_commit"] is None


def test_a_config_that_is_not_a_dataclass_is_recorded_as_its_repr(
    tmp_path: Path,
) -> None:
    record = json.loads(write_run_provenance({"alpha": 1}, tmp_path).read_text())

    assert record["config"] == {"repr": "{'alpha': 1}"}


def test_a_config_that_cannot_be_copied_does_not_raise(tmp_path: Path) -> None:
    import threading

    @dataclass
    class _Locked:
        lock: object

    assert write_run_provenance(_Locked(threading.Lock()), tmp_path) is None
