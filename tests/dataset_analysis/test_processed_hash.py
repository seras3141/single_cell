"""Unit tests for the Tier-2 processed folder-hash logic (Phase 4).

Run: VIRTUAL_ENV="" uv run --extra dev pytest tests/dataset_analysis/test_processed_hash.py -q
"""

from scripts.hash_processed_tier2 import aggregate_hash, hash_folder


def _line(md5: str, relpath: str) -> str:
    return f"{md5}  {relpath}"


def test_aggregate_hash_deterministic_and_order_independent():
    a = [_line("aaa", "./x.tif"), _line("bbb", "./sub/y.zarr")]
    assert aggregate_hash(a) == aggregate_hash(list(reversed(a)))  # sorted internally


def test_aggregate_hash_changes_on_nested_file_change():
    base = [_line("aaa", "./split_data/a.tif"), _line("bbb", "./inference/cellpose_sam/m.zarr")]
    changed = [_line("aaa", "./split_data/a.tif"), _line("ccc", "./inference/cellpose_sam/m.zarr")]
    assert aggregate_hash(base) != aggregate_hash(changed)


def test_aggregate_hash_changes_on_new_model_subfolder():
    base = [_line("aaa", "./inference/cellpose_sam/m.zarr")]
    with_new_model = base + [_line("ddd", "./inference/new_model/m.zarr")]
    assert aggregate_hash(base) != aggregate_hash(with_new_model)


def test_aggregate_hash_empty_stable():
    assert aggregate_hash([]) == aggregate_hash([])


def test_hash_folder_missing_returns_none(tmp_path):
    assert hash_folder(tmp_path / "does_not_exist") is None


def test_hash_folder_real_dir(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.tif").write_bytes(b"hello")
    (tmp_path / "sub" / "b.zarr").write_bytes(b"world")
    res = hash_folder(tmp_path, jobs=2)
    assert res is not None
    assert res["nfiles"] == 2
    assert res["total_bytes"] == 10  # 5 + 5
    assert len(res["folder_md5"]) == 32

    # a nested change flips the folder hash
    before = res["folder_md5"]
    (tmp_path / "sub" / "b.zarr").write_bytes(b"WORLD!")
    assert hash_folder(tmp_path, jobs=2)["folder_md5"] != before


def test_hash_folder_empty_dir_no_spurious_line(tmp_path):
    """A folder with zero regular files must hash as empty (no spurious xargs md5sum line)."""
    (tmp_path / "empty").mkdir()
    res = hash_folder(tmp_path / "empty", jobs=2)
    assert res is not None
    assert res["nfiles"] == 0 and res["n_symlinks"] == 0 and res["total_bytes"] == 0
    assert res["folder_md5"] == aggregate_hash([])  # deterministic empty-folder hash


def test_hash_folder_symlink_only_dir(tmp_path):
    """A symlink-only dir (no regular file) counts symlinks only — no spurious file entry."""
    (tmp_path / "raw.tif").write_bytes(b"x")
    d = tmp_path / "links"
    d.mkdir()
    (d / "a.tif").symlink_to(tmp_path / "raw.tif")
    res = hash_folder(d, jobs=2)
    assert res["nfiles"] == 1 and res["n_symlinks"] == 1  # 1 symlink, 0 real files


def test_hash_folder_symlinks_by_target(tmp_path):
    """split_data-style symlink trees: hashed by target path (selection), not target bytes."""
    target_dir = tmp_path / "raw"
    target_dir.mkdir()
    (target_dir / "img.tif").write_bytes(b"rawbytes")
    link_dir = tmp_path / "split_data"
    link_dir.mkdir()
    (link_dir / "sel.tif").symlink_to(target_dir / "img.tif")
    (link_dir / "dataset_split.json").write_bytes(b"{}")  # one real file

    res = hash_folder(link_dir, jobs=2)
    assert res["nfiles"] == 2 and res["n_symlinks"] == 1  # 1 symlink + 1 real file
    before = res["folder_md5"]

    # changing the TARGET's bytes must NOT change the folder hash (content ≠ selection)
    (target_dir / "img.tif").write_bytes(b"different-raw-bytes-entirely")
    assert hash_folder(link_dir, jobs=2)["folder_md5"] == before

    # re-pointing the symlink (a selection change) MUST change the hash
    (target_dir / "img2.tif").write_bytes(b"other")
    (link_dir / "sel.tif").unlink()
    (link_dir / "sel.tif").symlink_to(target_dir / "img2.tif")
    assert hash_folder(link_dir, jobs=2)["folder_md5"] != before
