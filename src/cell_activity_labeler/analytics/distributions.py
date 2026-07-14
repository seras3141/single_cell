"""Distribution plots for labeled activity outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def save_intensity_overlay(
    labeled_df: pd.DataFrame, metric: str, output_path: Path
) -> Path:
    """Save an active-vs-dead intensity overlay plot."""
    fig, ax = plt.subplots(figsize=(8, 5))
    active_values = labeled_df.loc[labeled_df["is_active"], metric].dropna()
    dead_values = labeled_df.loc[~labeled_df["is_active"], metric].dropna()

    if len(active_values) > 1:
        sns.kdeplot(active_values, fill=True, alpha=0.4, label="active", ax=ax)
    elif len(active_values) == 1:
        ax.axvline(active_values.iloc[0], color="tab:green", label="active")

    if len(dead_values) > 1:
        sns.kdeplot(dead_values, fill=True, alpha=0.4, label="dead", ax=ax)
    elif len(dead_values) == 1:
        ax.axvline(dead_values.iloc[0], color="tab:red", label="dead")

    ax.set_title(f"{metric} by cell status")
    ax.set_xlabel(metric)
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(alpha=0.2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


__all__ = ["save_intensity_overlay"]