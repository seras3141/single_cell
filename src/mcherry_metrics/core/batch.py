"""Batch orchestration for mCherry metrics extraction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..analytics import generate_standard_outputs, write_metrics_summary
from ..config import ExtractionConfig
from ..io import build_file_handler, discover_mcherry_images, resolve_label_paths
from ..io.exporters import write_instance_metrics
from ..io.loaders import should_keep_image
from .extractor import InstanceMetricsExtractor


class MetricsExtractor:
    """High-level extractor that runs batch extraction and writes outputs.

    Parameters
    ----------
    config : ExtractionConfig
        Extraction configuration.
    file_handler : optional
        Metadata parser for filenames.
    """

    def __init__(self, config: ExtractionConfig, file_handler=None):
        self.config = config
        self.file_handler = file_handler or build_file_handler()
        self.instance_extractor = InstanceMetricsExtractor(
            config=config,
            file_handler=self.file_handler,
        )

    def run(
        self,
        mcherry_paths: list[Path],
        mask_paths: list[Path | None] | None = None,
        output_csv: Path | None = None,
        summary_csv: Path | None = None,
        analytics_dir: Path | None = None,
        show_progress: bool = False,
    ) -> pd.DataFrame:
        """Run batch extraction for explicit image and label path lists."""
        metrics_df = self.instance_extractor.process_batch_images(
            img_paths=mcherry_paths,
            lbl_paths=mask_paths,
            show_progress=show_progress,
            n_jobs=self.config.n_jobs,
        )

        if output_csv is not None and not metrics_df.empty:
            write_instance_metrics(metrics_df, output_csv)

        if summary_csv is not None:
            write_metrics_summary(metrics_df, summary_csv)

        if analytics_dir is not None and self.config.write_analytics:
            generate_standard_outputs(
                metrics_df,
                analytics_dir,
                processed_image_paths=mcherry_paths,
            )

        return metrics_df


def run_extraction(
    mcherry_dir: Path,
    output_dir: Path,
    config: ExtractionConfig | None = None,
    mask_dir: Path | None = None,
    image_pattern: str = "*_mCherry.tif",
    label_suffix: str = "_Cells",
    wavelength_mappings: dict[int, str] | None = None,
    plate_number: str | None = None,
) -> pd.DataFrame:
    """Run milestone-1 extraction from directories.

    Parameters
    ----------
    mcherry_dir : Path
        Directory containing input mCherry images.
    output_dir : Path
        Directory receiving CSV and analytics outputs.
    config : ExtractionConfig, optional
        Extraction configuration. Defaults to ``ExtractionConfig()``.
    mask_dir : Path, optional
        Directory containing label masks. Defaults to ``mcherry_dir``.
    image_pattern : str
        Glob used to discover input images.
    label_suffix : str
        Filename suffix used to resolve label paths.
    wavelength_mappings : dict[int, str], optional
        Channel mapping overrides for metadata parsing.
    plate_number : str, optional
        Plate identifier override for metadata parsing.

    Returns
    -------
    pd.DataFrame
        Instance-level metrics table.
    """
    config = config or ExtractionConfig()
    file_handler = build_file_handler(
        wavelength_mappings=wavelength_mappings,
        plate_number=plate_number,
    )

    image_paths = discover_mcherry_images(mcherry_dir, image_pattern)
    if not image_paths:
        raise ValueError(
            f"No mCherry images matched pattern {image_pattern!r} in {mcherry_dir}"
        )

    image_paths = [
        image_path
        for image_path in image_paths
        if should_keep_image(image_path, file_handler, config.exclude_z0)
    ]
    label_paths = resolve_label_paths(image_paths, mask_dir, label_suffix)

    output_dir.mkdir(parents=True, exist_ok=True)
    extractor = MetricsExtractor(config=config, file_handler=file_handler)
    return extractor.run(
        mcherry_paths=image_paths,
        mask_paths=label_paths,
        output_csv=output_dir / "instance_metrics.csv",
        summary_csv=output_dir / "metrics_summary.csv",
        analytics_dir=output_dir,
        show_progress=False,
    )