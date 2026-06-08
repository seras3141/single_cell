"""Summary generation for labeled activity outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..core.summaries import generate_activity_summary


def build_label_summary(labeled_df: pd.DataFrame) -> pd.DataFrame:
    """Return the per-image label summary table."""
    return generate_activity_summary(labeled_df)


def write_label_summary_csv(summary_df: pd.DataFrame, output_path: Path) -> Path:
    """Write the per-image label summary CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False)
    return output_path


__all__ = ["build_label_summary", "write_label_summary_csv"]