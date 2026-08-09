"""Small aggregation helpers over the raw incarta ``split_data`` feature CSVs.

Used only where a result CSV doesn't already carry the aggregate needed (e.g. mean
cell area by z-slice) — prefer an existing small result file over these where one
exists, since these read every per-(well, timepoint, z) CSV in a directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def mean_area_by_z(feature_dir: Path, pattern: str = "*.csv") -> Optional[pd.Series]:
    """Mean ``area`` per ``z_index``, across every feature CSV in ``feature_dir``."""
    csv_paths = sorted(Path(feature_dir).glob(pattern))
    if not csv_paths:
        return None

    frames = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path, usecols=lambda c: c in ("area", "z_index"))
        if "area" in df.columns and "z_index" in df.columns:
            frames.append(df[["z_index", "area"]])
    if not frames:
        return None

    combined = pd.concat(frames, ignore_index=True)
    return combined.groupby("z_index")["area"].mean().sort_index()


def cell_count(feature_dir: Path, pattern: str = "*.csv") -> Optional[int]:
    """Total row count across every feature CSV in ``feature_dir`` (one row/cell)."""
    csv_paths = sorted(Path(feature_dir).glob(pattern))
    if not csv_paths:
        return None
    return sum(len(pd.read_csv(csv_path, usecols=[0])) for csv_path in csv_paths)
