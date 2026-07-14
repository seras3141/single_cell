#!/usr/bin/env python3
"""Tier-2 processed-data folder hashing (Phase 4, hash-only — no bytes stored).

For each experiment's large processed stage subfolders (split_data/, 3d_data/,
blur_heatmaps/, inference/, inference_tracked/, inference_scportrait/) compute a single
folder-state hash and write a small git-tracked JSON index. The folder hash is the md5 of
the sorted per-file md5 listing (computed recursively via `md5sum`), so it changes if any
nested tif/zarr/file changes, or a file/subfolder is added/removed (e.g. a new model under
inference/<model>). No bytes are copied or committed — only the hashes.

This mirrors Phase 1's raw md5-manifest approach (safe, read-only) rather than declaring the
already-produced data as DVC stage outs (which `dvc repro` could delete/regenerate). Small
"stats" units (processed_summary/, mcherry_metrics/, manifest.json, loose CSVs) are byte-
versioned by DVC in Phase 1 and are intentionally NOT hashed here.

Usage (via SLURM — reads all bytes of the processed data):
    python scripts/hash_processed_tier2.py \
        --processed-root "data/MF5V1_processed Timelapse samples 19.03.2024" \
        --out data_versioning/processed_hashes/processed_hash_index.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Large per-stage output subfolders to hash (hash-only, no bytes stored).
DEFAULT_SUBFOLDERS = [
    "split_data",
    "3d_data",
    "blur_heatmaps",
    "inference",
    "inference_tracked",
    "inference_scportrait",
]
# Subfolders whose immediate children are per-model dirs (also hashed individually,
# so a new model or a change within one model's outputs is attributable).
MODEL_LEVEL = {"inference", "inference_tracked", "inference_scportrait"}

DEFAULT_EXPERIMENTS = [
    "HD1509 MF5V1 0-72h 23-02-26",
    "SA110 MF5V1 0-72h 13-02-26",
    "HD1883 MF5V1 0-72h 20-03-26",
    "Ew2-1 MF5V1 0-72h 06-03-26",
    "Ew2-2 MF5V1 072h 17-04-26",
]


def aggregate_hash(md5sum_lines: List[str]) -> str:
    """Folder-state hash = md5 of the sorted per-file md5 listing.

    Pure function (order-independent via sort) so it is deterministic and testable:
    identical file sets → identical hash; any added/removed/changed file → different hash.
    """
    joined = "\n".join(sorted(md5sum_lines))
    return hashlib.md5(joined.encode()).hexdigest()


def hash_folder(path: Path, jobs: int = 8) -> Optional[Dict[str, object]]:
    """Return {folder_md5, nfiles, n_symlinks, total_bytes} for `path`, or None if absent.

    Recursive, relative-path (location-independent), never stores bytes. Two entry kinds:
      * real files  -> content md5 (via `find -type f | xargs md5sum`);
      * symlinks    -> their target path (via `find -type l`), NOT the target's bytes.
    `split_data/` is a tree of symlinks into the raw data (the train/test *selection*); its
    hash therefore tracks which files are selected + where they point, while the target
    image *content* is integrity-tracked separately by Phase 1's raw md5 manifests. This
    also avoids re-reading ~10 GB of raw per experiment through the links.
    """
    if not path.is_dir():
        return None
    # real files -> content md5 (relative paths), parallel
    find = subprocess.Popen(
        ["find", ".", "-type", "f", "-print0"],
        cwd=str(path),
        stdout=subprocess.PIPE,
    )
    md5 = subprocess.run(
        ["xargs", "-0", "-P", str(jobs), "-n", "64", "md5sum"],
        cwd=str(path),
        stdin=find.stdout,
        capture_output=True,
        text=True,
    )
    find.wait()
    content_lines = [ln for ln in md5.stdout.splitlines() if ln.strip()]
    # symlinks -> record "link:<target>  <relpath>" (selection state, no byte read)
    links = subprocess.run(
        ["find", ".", "-type", "l", "-printf", "link:%l  %p\n"],
        cwd=str(path),
        capture_output=True,
        text=True,
    )
    link_lines = [ln for ln in links.stdout.splitlines() if ln.strip()]
    lines = content_lines + link_lines
    total_bytes = 0
    for dirpath, _dirs, files in os.walk(path):
        for f in files:
            try:
                total_bytes += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass  # broken symlink / race — skip
    return {
        "folder_md5": aggregate_hash(lines),
        "nfiles": len(lines),
        "n_symlinks": len(link_lines),
        "total_bytes": total_bytes,
    }


def hash_experiment(exp_dir: Path, subfolders: List[str], jobs: int) -> Dict[str, object]:
    """Hash each present Tier-2 subfolder of one experiment; recurse into model dirs."""
    result: Dict[str, object] = {}
    for sub in subfolders:
        sub_path = exp_dir / sub
        folder = hash_folder(sub_path, jobs=jobs)
        if folder is None:
            continue  # missing subfolder — skip gracefully
        entry: Dict[str, object] = dict(folder)
        if sub in MODEL_LEVEL:
            models: Dict[str, object] = {}
            for model_dir in sorted(p for p in sub_path.iterdir() if p.is_dir()):
                mh = hash_folder(model_dir, jobs=jobs)
                if mh is not None:
                    models[model_dir.name] = mh
            if models:
                entry["models"] = models
        result[sub] = entry
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Tier-2 processed folder hashing (hash-only)")
    parser.add_argument("--processed-root", required=True, help="Root containing experiment dirs")
    parser.add_argument("--out", required=True, help="Output JSON index path")
    parser.add_argument("--experiments", nargs="*", default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--subfolders", nargs="*", default=DEFAULT_SUBFOLDERS)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Update only the given --experiments in an existing index, keeping the rest "
        "(for per-experiment re-tracking). Without it, --out is written from scratch.",
    )
    args = parser.parse_args()

    root = Path(args.processed_root)
    experiments: Dict[str, object] = {}
    if args.merge and Path(args.out).exists():
        with open(args.out) as f:
            experiments = (json.load(f) or {}).get("experiments", {})
        print(f"MERGE: loaded {len(experiments)} existing experiment entries from {args.out}")
    for exp in args.experiments:
        exp_dir = root / exp
        if not exp_dir.is_dir():
            print(f"SKIP missing experiment: {exp_dir}")
            continue
        print(f"=== hashing {exp} ===")
        experiments[exp] = hash_experiment(exp_dir, args.subfolders, args.jobs)
        for sub, e in experiments[exp].items():
            extra = f" ({len(e['models'])} models)" if isinstance(e, dict) and "models" in e else ""
            print(f"    {sub:22s} md5={e['folder_md5']} nfiles={e['nfiles']}{extra}")

    index = {
        "description": (
            "Tier-2 processed folder hashes (hash-only; no bytes stored/committed). "
            "folder_md5 = md5 of the sorted recursive per-file md5 listing — changes on "
            "any nested file change or added/removed file/subfolder (e.g. a new model)."
        ),
        "generated_by": "scripts/hash_processed_tier2.py",
        "processed_root": str(root),
        "experiments": experiments,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"=== wrote {out_path} ({len(experiments)} experiments) ===")


if __name__ == "__main__":
    main()
