"""Matplotlib plots for dataset summary notebooks."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

ROWS = list("ABCDEFGHIJKLMNOP")
COLS = list(range(1, 25))

LABELS: Dict[str, str] = {
    "Doxorubicin": "DOX",
    "Eprenetapopt": "EPR",
    "Navitoclax": "NAV",
    "Selinexor": "SEL",
    "Venetoclax": "VEN",
    "Staurosporine": "STA",
    "Benzethonium Chloride": "BZC",
    "DMSO": "DMSO",
}


def _select_plate_time(
    dataframe: pd.DataFrame,
    plate_id: Optional[str] = None,
    time_point: Optional[int] = None,
) -> Tuple[pd.DataFrame, Optional[str], Optional[int]]:
    if dataframe.empty:
        return dataframe, plate_id, time_point

    selected_plate = (
        plate_id or sorted(dataframe["plate_id"].dropna().astype(str).unique())[0]
    )
    subset = dataframe[dataframe["plate_id"] == selected_plate]
    selected_time = time_point
    if selected_time is None and "time_point" in subset:
        selected_time = int(
            sorted(subset["time_point"].dropna().astype(int).unique())[0]
        )
    if selected_time is not None and "time_point" in subset:
        subset = subset[subset["time_point"] == selected_time]

    return subset, selected_plate, selected_time


def _annotation_label(row: pd.Series) -> str:
    control = row.get("control")
    if isinstance(control, str) and control:
        return LABELS.get(control, control[:4].upper())
    drug = row.get("drug")
    if isinstance(drug, str) and drug:
        return LABELS.get(drug, drug[:3].upper())
    return ""


def _format_plate_axes(ax: plt.Axes, title: str) -> None:
    ax.set_xticks(np.arange(len(COLS)))
    ax.set_xticklabels(COLS, fontsize=7)
    ax.set_yticks(np.arange(len(ROWS)))
    ax.set_yticklabels(ROWS, fontsize=8)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_title(title)
    ax.set_xticks(np.arange(-0.5, len(COLS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ROWS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)


def _observed_wells(inventory: pd.DataFrame) -> pd.DataFrame:
    if inventory.empty:
        return inventory
    return inventory.drop_duplicates(["plate_id", "time_point", "well_id"])


def plot_plate_coverage(
    inventory: pd.DataFrame,
    plate_id: Optional[str] = None,
    time_point: Optional[int] = None,
) -> plt.Figure:
    """Plot a full-plate observed-subset coverage heatmap."""
    subset, selected_plate, selected_time = _select_plate_time(
        _observed_wells(inventory), plate_id, time_point
    )
    grid = np.zeros((len(ROWS), len(COLS)), dtype=int)
    labels = [["" for _ in COLS] for _ in ROWS]
    content_codes = {"drug": 1, "control": 2, "empty": 3, "unmapped": 4}

    for row in subset.itertuples():
        row_idx = ROWS.index(row.row)
        col_idx = int(row.col) - 1
        grid[row_idx, col_idx] = content_codes.get(str(row.content), 4)
        labels[row_idx][col_idx] = _annotation_label(pd.Series(row._asdict()))

    cmap = ListedColormap(["#d8dadd", "#80b1d3", "#fdb462", "#b3de69", "#fb8072"])
    fig, ax = plt.subplots(figsize=(14, 7))
    image = ax.imshow(grid, cmap=cmap, vmin=0, vmax=4)
    _format_plate_axes(
        ax,
        f"Observed Plate Coverage: {selected_plate or 'no plate'}"
        + (f" t{selected_time}" if selected_time is not None else ""),
    )
    for y in range(len(ROWS)):
        for x in range(len(COLS)):
            if labels[y][x]:
                ax.text(x, y, labels[y][x], ha="center", va="center", fontsize=6)
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, ticks=[0, 1, 2, 3, 4])
    cbar.ax.set_yticklabels(["not observed", "drug", "control", "empty", "unmapped"])
    fig.tight_layout()
    return fig


def plot_channel_completeness(
    completeness: pd.DataFrame,
    expected_channels: Sequence[str],
    plate_id: Optional[str] = None,
    time_point: Optional[int] = None,
) -> plt.Figure:
    """Plot channel completeness for observed wells."""
    subset, selected_plate, selected_time = _select_plate_time(
        completeness, plate_id, time_point
    )
    grid = np.full((len(ROWS), len(COLS)), np.nan)
    expected_count = max(len(expected_channels), 1)

    for row in subset.itertuples():
        row_idx = ROWS.index(row.row)
        col_idx = int(row.col) - 1
        grid[row_idx, col_idx] = float(row.n_channels_present) / expected_count

    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#d8dadd")
    fig, ax = plt.subplots(figsize=(14, 7))
    image = ax.imshow(grid, cmap=cmap, vmin=0, vmax=1)
    _format_plate_axes(
        ax,
        f"Channel Completeness: {selected_plate or 'no plate'}"
        + (f" t{selected_time}" if selected_time is not None else ""),
    )
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Fraction of expected channels present")
    fig.tight_layout()
    return fig


def plot_z_completeness(
    completeness: pd.DataFrame,
    plate_id: Optional[str] = None,
    time_point: Optional[int] = None,
) -> plt.Figure:
    """Plot core z-slice completeness for observed wells."""
    subset, selected_plate, selected_time = _select_plate_time(
        completeness, plate_id, time_point
    )
    grid = np.full((len(ROWS), len(COLS)), np.nan)

    for row in subset.itertuples():
        row_idx = ROWS.index(row.row)
        col_idx = int(row.col) - 1
        expected = max(int(row.n_expected_core_z), 1)
        grid[row_idx, col_idx] = float(row.n_core_z_present) / expected

    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#d8dadd")
    fig, ax = plt.subplots(figsize=(14, 7))
    image = ax.imshow(grid, cmap=cmap, vmin=0, vmax=1)
    _format_plate_axes(
        ax,
        f"Core Z-Slice Completeness: {selected_plate or 'no plate'}"
        + (f" t{selected_time}" if selected_time is not None else ""),
    )
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Fraction of expected z1-z20 present")
    fig.tight_layout()
    return fig


def plot_control_distribution(inventory: pd.DataFrame) -> plt.Figure:
    """Plot observed control-well counts."""
    observed = _observed_wells(inventory)
    fig, ax = plt.subplots(figsize=(8, 4))
    if observed.empty or "control" not in observed:
        ax.text(0.5, 0.5, "No observed controls", ha="center", va="center")
        ax.axis("off")
        return fig

    counts = observed.dropna(subset=["control"])["control"].value_counts()
    if counts.empty:
        ax.text(0.5, 0.5, "No observed controls", ha="center", va="center")
        ax.axis("off")
        return fig

    counts.plot(kind="bar", ax=ax, color="#fdb462")
    ax.set_ylabel("Observed wells")
    ax.set_xlabel("Control")
    ax.set_title("Observed Control Distribution")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


def plot_drug_distribution(inventory: pd.DataFrame) -> plt.Figure:
    """Plot observed drug/concentration well counts."""
    observed = _observed_wells(inventory)
    drug_rows = (
        observed[observed.get("content") == "drug"] if not observed.empty else observed
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    if drug_rows.empty:
        ax.text(0.5, 0.5, "No observed drug wells", ha="center", va="center")
        ax.axis("off")
        return fig

    pivot = (
        drug_rows.groupby(["drug", "concentration_uM"])["well_id"]
        .nunique()
        .unstack(fill_value=0)
        .sort_index()
    )
    image = ax.imshow(pivot.to_numpy(), cmap="Blues")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(col) for col in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Concentration (uM)")
    ax.set_ylabel("Drug")
    ax.set_title("Observed Drug Treatment Distribution")
    for y in range(pivot.shape[0]):
        for x in range(pivot.shape[1]):
            ax.text(x, y, int(pivot.iat[y, x]), ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Observed wells")
    fig.tight_layout()
    return fig
