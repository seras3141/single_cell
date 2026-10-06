"""Expected-vs-found inventory for a downloaded raw share tree.

Download completeness compares disk against the ``*.filelist.txt`` manifest
written by ``data/download_data.py --manifest-dir``.  Acquisition completeness
compares disk against a full dense grid (timepoint x wavelength x z-plane).

Summaries are keyed on the wavelength index (``w1``/``w2``/``w3``), not channel
names: the name mapping exists in two divergent copies in this repo.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

import pandas as pd

# e.g. t11_E07_s1_w2_z13.tif
FRAME_RE = re.compile(
    r"^t(?P<t>\d+)_(?P<well>[A-Z]\d+)_s(?P<s>\d+)_w(?P<w>\d+)_z(?P<z>\d+)\.tif$"
)

DEFAULT_EXPECTED_TIMEPOINTS = tuple(range(1, 41))
DEFAULT_EXPECTED_WAVELENGTHS = (1, 2, 3)
DEFAULT_EXPECTED_Z = tuple(range(1, 21))
PROJECTION_Z = 0

# Measured from pixel statistics; matches EXPERIMENT_WAVELENGTH_MAPPINGS in
# src/utils/file_utils.py. Explicit because the duplicate in
# src/cell_activity_labeler/utils/file_utils.py has HD1509/HD1883 transposed.
MEASURED_BF_WAVELENGTH: Dict[str, int] = {
    "Ew2-1": 3,
    "Ew2-2": 3,
    "HD1509": 1,
    "HD1883": 3,
    "SA110": 1,
}

MAX_INLINE_MISSING = 20


def parse_frame_name(name: str) -> Optional[Dict[str, Union[int, str]]]:
    """Parse a raw frame filename, or return None if it does not match."""
    match = FRAME_RE.match(name)
    if match is None:
        return None
    return {
        "time_point": int(match.group("t")),
        "well_id": match.group("well"),
        "site": int(match.group("s")),
        "wavelength": int(match.group("w")),
        "z_index": int(match.group("z")),
    }


def scan_frames(folder: Union[str, Path]) -> pd.DataFrame:
    """Inventory the raw frames directly inside *folder* (non-recursive)."""
    root = Path(folder)
    records = []
    for path in sorted(root.glob("*.tif")):
        parsed = parse_frame_name(path.name)
        if parsed is None:
            continue
        parsed["file_name"] = path.name
        parsed["size_bytes"] = path.stat().st_size
        records.append(parsed)
    if not records:
        return pd.DataFrame(
            columns=[
                "time_point",
                "well_id",
                "site",
                "wavelength",
                "z_index",
                "file_name",
                "size_bytes",
            ]
        )
    return pd.DataFrame.from_records(records)


def _stage(expected: Sequence[str], found: Iterable[str]) -> Dict[str, object]:
    """Count expected/found/missing names, inlining the first few missing."""
    found_set = set(found)
    missing = [name for name in expected if name not in found_set]
    return {
        "expected": len(expected),
        "found": len(expected) - len(missing),
        "missing": len(missing),
        "missing_entries": missing[:MAX_INLINE_MISSING],
        "missing_truncated": max(0, len(missing) - MAX_INLINE_MISSING),
    }


def _dense_grid_names(
    wells: Sequence[str],
    timepoints: Sequence[int],
    wavelength: int,
    sites: Sequence[int],
    z_planes: Sequence[int],
) -> List[str]:
    return [
        f"t{t}_{well}_s{s}_w{wavelength}_z{z}.tif"
        for well in wells
        for t in timepoints
        for s in sites
        for z in z_planes
    ]


def build_raw_summary(
    folder: Union[str, Path],
    manifest_path: Optional[Union[str, Path]] = None,
    expected_timepoints: Sequence[int] = DEFAULT_EXPECTED_TIMEPOINTS,
    expected_wavelengths: Sequence[int] = DEFAULT_EXPECTED_WAVELENGTHS,
    expected_z: Optional[Sequence[int]] = None,
) -> Dict[str, object]:
    """Summarize download and acquisition completeness for one raw folder.

    *expected_z* defaults to ``z0`` for a projection folder, else ``z1..z20``.
    A ``download`` block is added only when *manifest_path* exists.
    """
    if len(expected_timepoints) == 0:
        raise ValueError("expected_timepoints must not be empty")
    root = Path(folder)
    frames = scan_frames(root)
    is_projection = root.name.endswith("_Projection")
    if expected_z is None:
        expected_z = (PROJECTION_Z,) if is_projection else DEFAULT_EXPECTED_Z
    if len(expected_z) == 0:
        raise ValueError("expected_z must not be empty")

    on_disk = set(frames["file_name"]) if not frames.empty else set()
    summary: Dict[str, object] = {
        "folder": root.name,
        "is_projection": is_projection,
        "files_on_disk": len(on_disk),
        "bytes_on_disk": int(frames["size_bytes"].sum()) if not frames.empty else 0,
    }

    manifest: List[str] = []
    if manifest_path is not None and Path(manifest_path).is_file():
        manifest = [
            line.strip()
            for line in Path(manifest_path).read_text().splitlines()
            if line.strip()
        ]
        summary["download"] = _stage(manifest, on_disk)
        summary["manifest"] = str(manifest_path)

    # Axes from disk alone would make an empty folder look complete
    wells_set = set(frames["well_id"].unique()) if not frames.empty else set()
    sites_set = {int(s) for s in frames["site"].unique()} if not frames.empty else set()
    for name in manifest:
        parsed = parse_frame_name(name)
        if parsed is not None:
            wells_set.add(str(parsed["well_id"]))
            sites_set.add(int(parsed["site"]))
    wells = sorted(wells_set)
    sites = sorted(sites_set) or [1]
    acquisition: Dict[str, object] = {}
    for wavelength in expected_wavelengths:
        expected_names = _dense_grid_names(
            wells, expected_timepoints, wavelength, sites, expected_z
        )
        block = _stage(expected_names, on_disk)
        if not frames.empty:
            present = frames[frames["wavelength"] == wavelength]
            block["timepoints_present"] = sorted(
                int(t) for t in present["time_point"].unique()
            )
        else:
            block["timepoints_present"] = []
        # Full list for the CSV; stripped from the JSON
        block["_missing_all"] = [name for name in expected_names if name not in on_disk]
        acquisition[f"w{wavelength}"] = block
    summary["acquisition"] = acquisition
    summary["wells"] = wells
    summary["sites"] = sites
    summary["expected_timepoints"] = list(expected_timepoints)
    summary["expected_z"] = list(expected_z)
    return summary


def missing_frames_table(summary: Dict[str, object], folder_name: str) -> pd.DataFrame:
    """Long-form table of every missing frame, with a ``reason`` per row.

    ``timepoint_not_acquired`` means the whole timepoint is absent for that
    wavelength; ``gap_within_timepoint`` means a single frame is absent.
    """
    rows = []
    acquisition = summary.get("acquisition", {})
    if not isinstance(acquisition, dict):
        return pd.DataFrame(columns=["folder", "wavelength", "file_name", "reason"])
    for key, block in acquisition.items():
        if not isinstance(block, dict):
            continue
        present = set(block.get("timepoints_present") or [])
        for name in block.get("_missing_all") or []:
            parsed = parse_frame_name(name)
            timepoint = parsed["time_point"] if parsed else None
            rows.append(
                {
                    "folder": folder_name,
                    "wavelength": key,
                    "file_name": name,
                    "reason": (
                        "gap_within_timepoint"
                        if timepoint in present
                        else "timepoint_not_acquired"
                    ),
                }
            )
    return pd.DataFrame(rows, columns=["folder", "wavelength", "file_name", "reason"])


def _public(value):
    """Recursively drop underscore-prefixed keys before JSON serialization."""
    if isinstance(value, dict):
        return {k: _public(v) for k, v in value.items() if not str(k).startswith("_")}
    if isinstance(value, list):
        return [_public(item) for item in value]
    return value


def write_raw_summary(
    summaries: Sequence[Dict[str, object]],
    out_dir: Union[str, Path],
) -> Dict[str, Path]:
    """Write ``raw_summary.json``, ``raw_inventory.csv`` and ``raw_missing.csv``."""
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)

    summary_path = target / "raw_summary.json"
    summary_path.write_text(json.dumps(_public(list(summaries)), indent=2) + "\n")

    inventory_rows = []
    missing_frames = []
    for summary in summaries:
        folder_name = str(summary.get("folder", ""))
        inventory_rows.append(
            {
                "folder": folder_name,
                "files_on_disk": summary.get("files_on_disk", 0),
                "bytes_on_disk": summary.get("bytes_on_disk", 0),
                "wells": len(summary.get("wells", []) or []),
                "download_missing": (summary.get("download") or {}).get("missing"),
                **{
                    f"{key}_missing": block.get("missing")
                    for key, block in (summary.get("acquisition") or {}).items()
                    if isinstance(block, dict)
                },
            }
        )
        missing_frames.append(missing_frames_table(summary, folder_name))

    inventory_path = target / "raw_inventory.csv"
    pd.DataFrame(inventory_rows).to_csv(inventory_path, index=False)

    missing_path = target / "raw_missing.csv"
    combined = (
        pd.concat(missing_frames, ignore_index=True)
        if missing_frames
        else pd.DataFrame(columns=["folder", "wavelength", "file_name", "reason"])
    )
    combined.to_csv(missing_path, index=False)

    return {
        "summary": summary_path,
        "inventory": inventory_path,
        "missing": missing_path,
    }


def format_summary_table(summaries: Sequence[Dict[str, object]]) -> str:
    """Render a compact console table over *summaries*."""
    lines = [
        f"{'folder':52s} {'on disk':>8s} {'dl gap':>7s} "
        f"{'w1 miss':>8s} {'w2 miss':>8s} {'w3 miss':>8s}"
    ]
    for summary in summaries:
        acquisition = summary.get("acquisition") or {}
        download = summary.get("download") or {}
        gap = download.get("missing", "-") if isinstance(download, dict) else "-"

        def miss(key: str) -> str:
            block = acquisition.get(key)
            return str(block.get("missing")) if isinstance(block, dict) else "-"

        lines.append(
            f"{str(summary.get('folder', ''))[:52]:52s} "
            f"{summary.get('files_on_disk', 0):8d} {str(gap):>7s} "
            f"{miss('w1'):>8s} {miss('w2'):>8s} {miss('w3'):>8s}"
        )
    return "\n".join(lines)
