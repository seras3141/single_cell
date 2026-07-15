"""Unit tests for run_manifest: core stage tracking plus the Phase-2
DVC/Git provenance linking.

Run from the repo root:
    VIRTUAL_ENV="" uv run pytest tests/dataset_analysis/test_run_manifest.py -q
"""

import json
import os  # noqa: F401  (used via __import__ in the atomic-write test)
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from src.dataset_analysis import run_manifest as rm
from src.dataset_analysis.run_manifest import (
    MANIFEST_FILENAME,
    STAGE_ORDER,
    RunManifest,
    StageRecord,
    complete_stage_with_dvc,
    create_or_load_manifest,
    current_git_commit,
    find_manifests,
    resolve_dvc_output,
)

DUMMY_CONFIG = {"model": {"type": "cyto3"}, "flow_threshold": 0.4}


# ─── create ───────────────────────────────────────────────────────────────────

def test_create_writes_valid_json(tmp_path):
    manifest = RunManifest.create(str(tmp_path), "/data/raw", DUMMY_CONFIG)

    manifest_file = tmp_path / MANIFEST_FILENAME
    assert manifest_file.exists()

    data = json.loads(manifest_file.read_text())
    assert data["experiment_id"] == tmp_path.name
    assert data["input_dir"] == "/data/raw"
    assert "config_hash" in data
    assert set(data["stages"].keys()) == set(STAGE_ORDER)
    for stage_data in data["stages"].values():
        assert stage_data["status"] == "not_started"


def test_create_uses_provided_experiment_id(tmp_path):
    manifest = RunManifest.create(str(tmp_path), "/raw", DUMMY_CONFIG, experiment_id="my_exp")
    assert manifest.experiment_id == "my_exp"


# ─── stage transitions ────────────────────────────────────────────────────────

def test_start_stage_sets_running(tmp_path):
    manifest = RunManifest.create(str(tmp_path), "/raw", DUMMY_CONFIG)
    manifest.start_stage("prepare")

    assert manifest.stages["prepare"].status == "running"
    assert manifest.stages["prepare"].started_at is not None


def test_complete_stage(tmp_path):
    manifest = RunManifest.create(str(tmp_path), "/raw", DUMMY_CONFIG)
    manifest.start_stage("prepare")
    manifest.complete_stage("prepare", output_dir=str(tmp_path / "split_data"))

    record = manifest.stages["prepare"]
    assert record.status == "completed"
    assert record.completed_at is not None
    assert "split_data" in record.output_dir


def test_fail_stage(tmp_path):
    manifest = RunManifest.create(str(tmp_path), "/raw", DUMMY_CONFIG)
    manifest.start_stage("segment-2d")
    manifest.fail_stage("segment-2d", error="CUDA out of memory")

    record = manifest.stages["segment-2d"]
    assert record.status == "failed"
    assert record.error == "CUDA out of memory"
    assert record.completed_at is not None


def test_skip_stage(tmp_path):
    manifest = RunManifest.create(str(tmp_path), "/raw", DUMMY_CONFIG)
    manifest.skip_stage("track", reason="Not implemented")

    record = manifest.stages["track"]
    assert record.status == "skipped"
    assert record.metadata == {"reason": "Not implemented"}


def test_start_stage_captures_config_snapshot(tmp_path):
    manifest = RunManifest.create(str(tmp_path), "/raw", DUMMY_CONFIG)
    snap = {"model_type": "cyto3", "flow_threshold": 0.4}
    manifest.start_stage("segment-2d", config=snap)

    assert manifest.stages["segment-2d"].config_snapshot == snap


# ─── next_steps ───────────────────────────────────────────────────────────────

def test_next_steps_returns_nonterminal_in_order(tmp_path):
    manifest = RunManifest.create(str(tmp_path), "/raw", DUMMY_CONFIG)
    manifest.complete_stage("prepare")
    manifest.complete_stage("segment-2d")
    manifest.skip_stage("track")

    remaining = manifest.next_steps()
    assert remaining == ["mcherry", "extract"]


def test_next_steps_empty_when_all_terminal(tmp_path):
    manifest = RunManifest.create(str(tmp_path), "/raw", DUMMY_CONFIG)
    for stage in STAGE_ORDER:
        manifest.complete_stage(stage)

    assert manifest.next_steps() == []


# ─── round-trip ───────────────────────────────────────────────────────────────

def test_load_from_output_dir_round_trips(tmp_path):
    original = RunManifest.create(str(tmp_path), "/raw", DUMMY_CONFIG, experiment_id="exp_rt")
    original.start_stage("prepare")
    original.complete_stage("prepare", output_dir=str(tmp_path / "prep"))

    reloaded = RunManifest.load_from_output_dir(str(tmp_path))

    assert reloaded.experiment_id == "exp_rt"
    assert reloaded.stages["prepare"].status == "completed"
    assert reloaded.stages["prepare"].output_dir == str(tmp_path / "prep")
    assert reloaded.stages["segment-2d"].status == "not_started"


# ─── atomic write ─────────────────────────────────────────────────────────────

def test_atomic_write_no_corrupt_file_on_crash(tmp_path, monkeypatch):
    """Simulate a crash mid-write: .tmp file is removed; original is untouched."""
    manifest = RunManifest.create(str(tmp_path), "/raw", DUMMY_CONFIG)
    manifest.complete_stage("prepare")

    original_replace = __import__("os").replace

    def crashing_replace(src, dst):
        # Remove the tmp file to simulate crash before rename
        Path(src).unlink(missing_ok=True)
        raise OSError("simulated crash")

    monkeypatch.setattr("os.replace", crashing_replace)

    with pytest.raises(OSError):
        manifest.save()

    # The tmp file should be cleaned up
    tmp_file = tmp_path / (MANIFEST_FILENAME + ".tmp")
    assert not tmp_file.exists()

    # The original manifest is still valid (written by complete_stage before crash)
    loaded = RunManifest.load_from_output_dir(str(tmp_path))
    assert loaded.stages["prepare"].status == "completed"


# ─── find_manifests ───────────────────────────────────────────────────────────

def test_find_manifests_locates_nested(tmp_path):
    for name in ("exp_a", "exp_b", "exp_c"):
        d = tmp_path / name
        d.mkdir()
        RunManifest.create(str(d), "/raw", DUMMY_CONFIG, experiment_id=name)

    found = find_manifests(str(tmp_path))
    ids = {m.experiment_id for m in found}
    assert ids == {"exp_a", "exp_b", "exp_c"}


def test_find_manifests_empty_dir(tmp_path):
    assert find_manifests(str(tmp_path)) == []


# ─── STAGE_ORDER: mcherry ─────────────────────────────────────────────────────

def test_mcherry_stage_order_position():
    idx_track = STAGE_ORDER.index("track")
    idx_mcherry = STAGE_ORDER.index("mcherry")
    idx_extract = STAGE_ORDER.index("extract")
    assert idx_track < idx_mcherry < idx_extract


def test_next_steps_includes_mcherry(tmp_path):
    manifest = RunManifest.create(str(tmp_path), "/raw", DUMMY_CONFIG)
    for stage in ["prepare", "segment-2d", "track"]:
        manifest.complete_stage(stage)
    assert manifest.next_steps() == ["mcherry", "extract"]


# ─── create_or_load_manifest ──────────────────────────────────────────────────

def test_create_or_load_creates_when_missing(tmp_path):
    manifest = create_or_load_manifest(str(tmp_path), "/raw", DUMMY_CONFIG)
    assert (tmp_path / MANIFEST_FILENAME).exists()
    assert manifest.stages["prepare"].status == "not_started"


def test_create_or_load_loads_existing(tmp_path):
    m1 = create_or_load_manifest(str(tmp_path), "/raw", DUMMY_CONFIG)
    m1.complete_stage("prepare")

    m2 = create_or_load_manifest(str(tmp_path), "/raw", DUMMY_CONFIG)
    assert m2.stages["prepare"].status == "completed"
    assert m2.stages["segment-2d"].status == "not_started"


def test_create_or_load_incremental_stages(tmp_path):
    """Simulates run_preprocessing then run_inference each updating the same manifest."""
    m1 = create_or_load_manifest(str(tmp_path), "/raw", DUMMY_CONFIG)
    m1.start_stage("prepare")
    m1.complete_stage("prepare", output_dir=str(tmp_path / "split_data"))

    m2 = create_or_load_manifest(str(tmp_path), "/raw", DUMMY_CONFIG)
    m2.start_stage("segment-2d")
    m2.complete_stage("segment-2d", output_dir=str(tmp_path / "masks"))

    final = create_or_load_manifest(str(tmp_path), "/raw", DUMMY_CONFIG)
    assert final.stages["prepare"].status == "completed"
    assert final.stages["segment-2d"].status == "completed"
    assert final.stages["track"].status == "not_started"


# ═══════════════════════════════════════════════════════════════════════════ #
# Phase-2 DVC / Git provenance linking
# ═══════════════════════════════════════════════════════════════════════════ #

# ─── StageRecord / RunManifest schema round-trips ──────────────────────────────

def test_stage_record_dvc_fields_roundtrip():
    rec = StageRecord(
        status="completed", output_dir="out", dvc_hash="abc123", dvc_path="out"
    )
    d = rec.to_dict()
    assert d["dvc_hash"] == "abc123"
    assert d["dvc_path"] == "out"
    assert StageRecord.from_dict(d) == rec


def test_stage_record_to_dict_drops_none_dvc_fields():
    # A record with no DVC info must not emit null dvc keys.
    d = StageRecord(status="running").to_dict()
    assert "dvc_hash" not in d
    assert "dvc_path" not in d


def test_manifest_new_fields_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(rm, "current_git_commit", lambda repo_root=None: "deadbee")
    m = RunManifest.create(str(tmp_path), input_dir="in", config={"a": 1})
    m.complete_stage("prepare", output_dir="out", dvc_hash="h1.dir", dvc_path="out")
    m.save()

    reloaded = RunManifest.load_from_output_dir(str(tmp_path))
    assert reloaded.config_git_commit == "deadbee"
    assert reloaded.stages["prepare"].dvc_hash == "h1.dir"
    assert reloaded.stages["prepare"].dvc_path == "out"
    assert reloaded.to_dict()["config_git_commit"] == "deadbee"


def test_backward_compat_old_manifest_loads(tmp_path):
    """A manifest predating the new keys loads with None defaults (no crash)."""
    old = {
        "experiment_id": "EXP",
        "input_dir": "in",
        "output_dir": str(tmp_path),
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "config_hash": "x",
        # no config_git_commit; stages carry no dvc_hash/dvc_path
        "stages": {"prepare": {"status": "completed", "output_dir": "out"}},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(old))

    m = RunManifest.load(str(path))
    assert m.config_git_commit is None
    assert m.stages["prepare"].dvc_hash is None
    assert m.stages["prepare"].dvc_path is None
    # unknown/new keys tolerated on the way in and back out
    assert m.to_dict()["config_git_commit"] is None


# ─── resolve_dvc_output ────────────────────────────────────────────────────────

def test_resolve_dvc_output_untracked_path(tmp_path):
    assert resolve_dvc_output(str(tmp_path / "nope")) == (None, None)


def test_resolve_dvc_output_from_dvc_pointer(tmp_path):
    target = tmp_path / "processed_summary"
    pointer = tmp_path / "processed_summary.dvc"
    pointer.write_text(
        yaml.safe_dump({"outs": [{"md5": "b07508fd.dir", "path": "processed_summary"}]})
    )
    md5, path = resolve_dvc_output(str(target))
    assert md5 == "b07508fd.dir"  # .dir suffix preserved verbatim
    assert path == str(target)


def test_resolve_dvc_output_from_dvc_lock(tmp_path):
    (tmp_path / "myout").write_text("data")
    (tmp_path / "dvc.lock").write_text(
        yaml.safe_dump(
            {"stages": {"prepare": {"outs": [{"path": "myout", "md5": "1234abcd"}]}}}
        )
    )
    md5, path = resolve_dvc_output(str(tmp_path / "myout"))
    assert md5 == "1234abcd"
    assert path == str(tmp_path / "myout")


def test_resolve_dvc_output_never_raises_on_bad_dvc(tmp_path):
    (tmp_path / "x.dvc").write_text(":::not: valid: yaml: [")
    assert resolve_dvc_output(str(tmp_path / "x")) == (None, None)


# ─── current_git_commit — best-effort, never raises ────────────────────────────

def test_current_git_commit_absent_returns_none(monkeypatch):
    def _boom(*a, **k):
        raise FileNotFoundError("git not installed")

    monkeypatch.setattr("subprocess.run", _boom)
    assert current_git_commit() is None


def test_current_git_commit_clean_tree(monkeypatch):
    def _fake(cmd, **kw):
        if "status" in cmd:
            return SimpleNamespace(returncode=0, stdout="")
        return SimpleNamespace(returncode=0, stdout="abcdef\n")

    monkeypatch.setattr("subprocess.run", _fake)
    assert current_git_commit() == "abcdef"


def test_current_git_commit_dirty_tree(monkeypatch):
    def _fake(cmd, **kw):
        if "status" in cmd:
            return SimpleNamespace(returncode=0, stdout=" M file.py\n")
        return SimpleNamespace(returncode=0, stdout="abcdef\n")

    monkeypatch.setattr("subprocess.run", _fake)
    assert current_git_commit() == "abcdef-dirty"


def test_current_git_commit_nonzero_returns_none(monkeypatch):
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **k: SimpleNamespace(returncode=128, stdout=""),
    )
    assert current_git_commit() is None


# ─── complete_stage_with_dvc + summary live-resolution ─────────────────────────

def test_complete_stage_with_dvc_attaches_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(rm, "current_git_commit", lambda repo_root=None: None)
    m = RunManifest.create(str(tmp_path), input_dir="in", config={})
    # sibling pointer so resolution succeeds
    (tmp_path / "out").mkdir()
    (tmp_path / "out.dvc").write_text(
        yaml.safe_dump({"outs": [{"md5": "feed42", "path": "out"}]})
    )
    complete_stage_with_dvc(m, "prepare", output_dir=str(tmp_path / "out"))
    assert m.stages["prepare"].status == "completed"
    assert m.stages["prepare"].dvc_hash == "feed42"


def test_complete_stage_with_dvc_untracked_is_none_not_fatal(tmp_path, monkeypatch):
    monkeypatch.setattr(rm, "current_git_commit", lambda repo_root=None: None)
    m = RunManifest.create(str(tmp_path), input_dir="in", config={})
    complete_stage_with_dvc(m, "prepare", output_dir=str(tmp_path / "untracked"))
    assert m.stages["prepare"].status == "completed"
    assert m.stages["prepare"].dvc_hash is None


def test_summary_live_resolves_without_mutating(tmp_path, monkeypatch):
    monkeypatch.setattr(rm, "current_git_commit", lambda repo_root=None: "cafe123")
    m = RunManifest.create(str(tmp_path), input_dir="in", config={})
    (tmp_path / "out").mkdir()
    (tmp_path / "out.dvc").write_text(
        yaml.safe_dump({"outs": [{"md5": "abcd1234ef56", "path": "out"}]})
    )
    # complete WITHOUT DVC info stored (simulates the usual pre-dvc-add ordering)
    m.complete_stage("prepare", output_dir=str(tmp_path / "out"))
    assert m.stages["prepare"].dvc_hash is None  # nothing stored on the record

    out = m.summary()
    assert "config commit: cafe123" in out
    assert "dvc:abcd1234ef56" in out  # resolved live for display
    # display-only: the record itself must not have been mutated
    assert m.stages["prepare"].dvc_hash is None
