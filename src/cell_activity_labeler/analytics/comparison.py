"""Cross-sample comparison plots for labeled results."""

from __future__ import annotations

from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_activity_ratio_by_sample(
    labeled_df: pd.DataFrame, output_path: Path
) -> Path | None:
    """Save average activity ratio by sample when sample metadata is present."""
    if "sample" not in labeled_df.columns or labeled_df["sample"].dropna().empty:
        return None

    sample_summary = (
        labeled_df.groupby("sample", dropna=False)["is_active"]
        .mean()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(sample_summary.index.astype(str), sample_summary.values, color="#4f7d44")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Activity ratio")
    ax.set_title("Activity ratio by sample")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def save_plate_activity_heatmap(
    labeled_df: pd.DataFrame, output_path: Path
) -> Path | None:
    """Save a well-plate heatmap when sample names look like well IDs."""
    if "sample" not in labeled_df.columns:
        return None

    pattern = re.compile(r"^(?P<row>[A-Za-z])(?P<col>\d{2})$")
    sample_summary = (
        labeled_df.groupby("sample", dropna=False)["is_active"].mean().reset_index()
    )
    matches = sample_summary["sample"].astype(str).map(pattern.match)
    if matches.isna().any() or any(match is None for match in matches):
        return None

    sample_summary["row"] = [match.group("row").upper() for match in matches]
    sample_summary["col"] = [int(match.group("col")) for match in matches]
    rows = sorted(sample_summary["row"].unique())
    cols = sorted(sample_summary["col"].unique())
    heatmap = np.full((len(rows), len(cols)), np.nan)

    row_index = {row: index for index, row in enumerate(rows)}
    col_index = {col: index for index, col in enumerate(cols)}
    for _, row in sample_summary.iterrows():
        heatmap[row_index[row["row"]], col_index[row["col"]]] = row["is_active"]

    fig, ax = plt.subplots(figsize=(max(6, len(cols) * 0.6), max(4, len(rows) * 0.6)))
    image = ax.imshow(heatmap, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(cols)), [str(col) for col in cols])
    ax.set_yticks(range(len(rows)), rows)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_title("Plate activity heatmap")
    fig.colorbar(image, ax=ax, label="Activity ratio")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


__all__ = ["save_activity_ratio_by_sample", "save_plate_activity_heatmap"]