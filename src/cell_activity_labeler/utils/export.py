"""Export helpers for activity classification outputs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff
from tqdm import tqdm

from cell_activity_labeler.core.classifier import create_activity_labeled_image
from cell_activity_labeler.utils.df_utils import add_meta_info
from cell_activity_labeler.utils.io import (
    find_activity_from_mcherry_path,
    find_brightfield_from_mcherry_path,
)


def save_threshold_config(config, output_dir: Path | str) -> Path:
    """Save a threshold configuration JSON file."""
    if config is None:
        raise ValueError("No threshold configuration is available to save.")

    config_path = Path(output_dir) / "threshold_config.json"
    config.save(config_path)
    return config_path


def validate_label_dict(label_dict: dict[str, int]) -> bool:
    """Validate activity label mappings used for exported activity images."""
    if not isinstance(label_dict, dict):
        raise ValueError("label_dict must be a dictionary mapping class names to label values.")
    if "active" not in label_dict or "dead" not in label_dict:
        raise ValueError("label_dict must contain mappings for both 'active' and 'dead'.")
    for key, value in label_dict.items():
        if not isinstance(value, int):
            raise ValueError(f"Label value for key '{key}' is not an integer.")
    return True


def save_output_images(
    image_paths,
    label_paths,
    metrics_df: pd.DataFrame,
    output_dir: Path | str,
    label_dict: dict[str, int] | None = None,
    out_widget=None,
) -> list[Path]:
    """Save activity-labeled images from classification results."""
    label_is_bin = False
    if label_dict is not None:
        validate_label_dict(label_dict)
        label_is_bin = True

    activity_dir = Path(output_dir)
    activity_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    saved = 0
    skipped = 0

    def _print(message: str) -> None:
        if out_widget is None:
            print(message)
        else:
            with out_widget:
                print(message)

    pairs = list(zip(image_paths, label_paths))
    progress = tqdm(pairs, desc="Saving activity labels")
    for img_path, lbl_path in progress:
        img_p = Path(img_path)
        img_name = img_p.name
        try:
            if lbl_path is None or not Path(lbl_path).exists():
                _print(f"Skipping {img_name}: label file not found")
                skipped += 1
                continue

            lbl = tiff.imread(str(lbl_path)).astype(np.int32, copy=False)
            classification = metrics_df[metrics_df["image"] == img_name].copy()
            if classification.empty:
                _print(f"Skipping {img_name}: no classification data available")
                skipped += 1
                continue

            activity_labels = create_activity_labeled_image(
                lbl,
                classification,
                label_dict=label_dict,
            )
            activity_suffix = "_activity_bin" if label_is_bin else "_activity"
            activity_path = find_activity_from_mcherry_path(
                img_p,
                activity_suffix=activity_suffix,
                activity_dir=activity_dir,
                must_exist=False,
            )
            if activity_path is None:
                raise ValueError(f"Could not derive activity output path for {img_name}")

            tiff.imwrite(
                str(activity_path),
                activity_labels.astype(np.int32),
                metadata={"is_bin": label_is_bin},
            )
            saved += 1
            saved_paths.append(activity_path)
        except Exception as exc:
            _print(f"Error processing {img_name}: {exc}")
            skipped += 1

    _print(f"Done. Saved: {saved}, Skipped/Errors: {skipped}. Files written to: {activity_dir}")
    return saved_paths


def enrich_summary_with_paths(
    summary: pd.DataFrame,
    image_paths,
    label_paths,
    activity_paths,
    output_dir: Path | str,
) -> pd.DataFrame:
    """Add source and output image paths to a per-image summary."""
    output_dir = Path(output_dir)
    mcherry_by_name = {Path(path).name: Path(path) for path in image_paths or []}
    label_by_name = {
        Path(img_path).name: (label_paths[index] if index < len(label_paths or []) else None)
        for index, img_path in enumerate(image_paths or [])
    }
    activity_by_name = {Path(path).name: Path(path) for path in activity_paths or []}

    def _build_path_row(img_name):
        mcherry_path = mcherry_by_name.get(img_name)

        brightfield_path = None
        if mcherry_path is not None:
            try:
                candidate = find_brightfield_from_mcherry_path(mcherry_path)
                brightfield_path = str(candidate) if candidate and candidate.exists() else None
            except Exception:
                brightfield_path = None

        label_path = label_by_name.get(img_name)
        activity_path = None
        if mcherry_path is not None:
            candidate = find_activity_from_mcherry_path(
                mcherry_path,
                activity_suffix="_activity_bin",
                activity_dir=output_dir,
                must_exist=False,
            )
            if candidate is not None and not candidate.exists():
                candidate = activity_by_name.get(candidate.name)
            activity_path = str(candidate) if candidate else None

        return pd.Series(
            {
                "mcherry_path": str(mcherry_path) if mcherry_path else None,
                "brightfield_path": brightfield_path,
                "label_path": str(label_path) if label_path is not None else None,
                "activity_path": activity_path,
            }
        )

    path_cols = summary["image"].apply(_build_path_row)
    return pd.concat([summary.copy(), path_cols], axis=1)


def save_activity_classification(
    metrics_df: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path | str,
    threshold_config,
    image_paths=None,
    label_paths=None,
    activity_paths=None,
    file_handler=None,
    out_widget=None,
) -> dict[str, Path]:
    """Save per-instance, per-image, and overall classification outputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    threshold_metric = getattr(threshold_config, "metric", "N/A")
    threshold_method = getattr(threshold_config, "method", "N/A")
    output_prefix = f"cell_activity_{threshold_metric}_{threshold_method}"

    instance_output = output_dir / f"{output_prefix}_per_instance.csv"
    metrics_df.to_csv(instance_output, index=False)

    summary = enrich_summary_with_paths(
        summary,
        image_paths=image_paths or [],
        label_paths=label_paths or [],
        activity_paths=activity_paths or [],
        output_dir=output_dir,
    )
    if "ID" not in summary.columns and file_handler is not None:
        summary = add_meta_info(summary, file_handler=file_handler)

    summary_output = output_dir / f"{output_prefix}_summary.csv"
    summary.to_csv(summary_output, index=False)

    overall_stats = {
        "metric_used": threshold_metric,
        "threshold_method": threshold_method,
        "total_images": len(summary),
        "total_cells": len(metrics_df),
        "total_active": int(metrics_df["is_active"].sum()),
        "total_dead": int((~metrics_df["is_active"]).sum()),
        "overall_activity_rate": float(metrics_df["is_active"].mean()),
        "avg_cells_per_image": float(summary["n_instances"].mean()),
        "avg_activity_rate_per_image": float(summary["activity_ratio"].mean()),
    }
    stats_output = output_dir / f"{output_prefix}_overall_stats.json"
    stats_output.write_text(json.dumps(overall_stats, indent=2))

    if out_widget is None:
        _print_export_summary(instance_output, summary_output, stats_output, overall_stats)
    else:
        with out_widget:
            _print_export_summary(instance_output, summary_output, stats_output, overall_stats)

    return {
        "instance_csv": instance_output,
        "summary_csv": summary_output,
        "overall_stats_json": stats_output,
    }


def _print_export_summary(
    instance_output: Path,
    summary_output: Path,
    stats_output: Path,
    overall_stats: dict[str, object],
) -> None:
    print("\nFiles saved:")
    print(f"- Per-instance results: {instance_output}")
    print(f"- Per-image summary: {summary_output}")
    print(f"- Overall statistics: {stats_output}")
    print(
        "\n=== CELL ACTIVITY CLASSIFICATION RESULTS ===\n"
        f"Metric used: {overall_stats['metric_used']}\n"
        f"Threshold method: {overall_stats['threshold_method']}\n"
        f"Total images processed: {overall_stats['total_images']}\n"
        f"Total cells analyzed: {overall_stats['total_cells']}\n"
        f"Active cells: {overall_stats['total_active']} "
        f"({overall_stats['overall_activity_rate']:.1%})\n"
        f"Dead cells: {overall_stats['total_dead']} "
        f"({1 - overall_stats['overall_activity_rate']:.1%})"
    )
