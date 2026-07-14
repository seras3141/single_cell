"""CLI frontend for milestone-2 activity labeling."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ...analytics import build_label_summary, run_labeling_analytics
from ...config import LabelingConfig, ThresholdParams
from ...core import ThresholdInstanceLabeler, save_activity_images
from ...io import load_metrics_csv, write_labeled_instances, write_label_summary
from ..base import AbstractFrontend


class CLIFrontend(AbstractFrontend):
    """Non-interactive labeling frontend backed by argparse."""

    def __init__(
        self,
        config: LabelingConfig,
        analytics_dir: Path | None = None,
        write_analytics: bool = True,
        write_activity_images: bool = True,
    ) -> None:
        self._config = config
        self.analytics_dir = analytics_dir
        self.write_analytics = write_analytics
        self.write_activity_images = write_activity_images

    def load_metrics(self, source: Path) -> pd.DataFrame:
        return load_metrics_csv(source)

    def configure_labeling(self) -> LabelingConfig:
        return self._config

    def run_labeling(
        self, metrics_df: pd.DataFrame, config: LabelingConfig
    ) -> pd.DataFrame:
        labeler = ThresholdInstanceLabeler(config)
        return labeler.run(metrics_df)

    def show_analytics(self, labeled_df: pd.DataFrame) -> None:
        if not self.write_analytics or self.analytics_dir is None:
            return
        run_labeling_analytics(labeled_df, self.analytics_dir)

    def export_results(self, labeled_df: pd.DataFrame, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_labeled_instances(labeled_df, output_dir / "labeled_instances.csv")
        summary_df = build_label_summary(labeled_df)
        write_label_summary(summary_df, output_dir / "label_summary.csv")

        if self.write_activity_images:
            save_activity_images(labeled_df, output_dir / "activity_images")

    def run(self, metrics_csv: Path, output_dir: Path) -> pd.DataFrame:
        metrics_df = self.load_metrics(metrics_csv)
        config = self.configure_labeling()
        labeled_df = self.run_labeling(metrics_df, config)
        self.export_results(labeled_df, output_dir)
        self.show_analytics(labeled_df)
        return labeled_df


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Label cells from a precomputed instance_metrics.csv file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--metrics-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--metric", default="percentile_90")
    parser.add_argument(
        "--method",
        default="otsu",
        choices=["otsu", "yen", "li", "triangle", "percentile", "manual"],
    )
    parser.add_argument("--per-image", action="store_true")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Manual threshold value when --method manual is used.",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=90.0,
        help="Percentile value when --method percentile is used.",
    )
    parser.add_argument("--no-analytics", action="store_true")
    parser.add_argument("--skip-activity-images", action="store_true")
    return parser


def _build_config_from_args(args: argparse.Namespace) -> LabelingConfig:
    params = ThresholdParams(
        percentile=args.percentile if args.method == "percentile" else None,
        manual_value=args.threshold if args.method == "manual" else None,
    )
    return LabelingConfig(
        metric=args.metric,
        method=args.method,
        per_image=args.per_image,
        params=params,
    )


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    config = _build_config_from_args(args)
    frontend = CLIFrontend(
        config=config,
        analytics_dir=args.output_dir / "analytics",
        write_analytics=not args.no_analytics,
        write_activity_images=not args.skip_activity_images,
    )
    frontend.run(args.metrics_csv, args.output_dir)


__all__ = ["CLIFrontend", "build_arg_parser", "main"]