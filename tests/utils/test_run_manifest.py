"""Unit tests for run_manifest, incl. the Phase-2 DVC/Git provenance linking.

Run from the repo root:
    VIRTUAL_ENV="" uv run pytest tests/utils/test_run_manifest.py -q
"""

from types import SimpleNamespace

import yaml

from src.dataset_analysis import run_manifest as rm
from src.dataset_analysis.run_manifest import (
    RunManifest,
    StageRecord,
    complete_stage_with_dvc,
    current_git_commit,
    resolve_dvc_output,
)


# --------------------------------------------------------------------------- #
# StageRecord / RunManifest schema round-trips
# --------------------------------------------------------------------------- #
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
    path.write_text(__import__("json").dumps(old))

    m = RunManifest.load(str(path))
    assert m.config_git_commit is None
    assert m.stages["prepare"].dvc_hash is None
    assert m.stages["prepare"].dvc_path is None
    # unknown/new keys tolerated on the way in and back out
    assert m.to_dict()["config_git_commit"] is None


# --------------------------------------------------------------------------- #
# resolve_dvc_output
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# current_git_commit — best-effort, never raises
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# complete_stage_with_dvc + summary live-resolution
# --------------------------------------------------------------------------- #
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
