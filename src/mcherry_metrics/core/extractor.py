"""Instance-level mCherry metric extraction."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff
from joblib import Parallel, delayed
from skimage import measure
from tqdm import tqdm

from ..config import ExtractionConfig
from ..io.exporters import finalize_metrics_dataframe
from ..io.loaders import ensure_2d, extract_image_metadata, find_label_from_mcherry_path
from ..io.loaders import build_file_handler
from .preprocessing import ImagePreprocessor

logger = logging.getLogger(__name__)


class InstanceMetricsExtractor:
    """Compute per-instance mCherry intensity statistics.

    Parameters
    ----------
    config : ExtractionConfig
        Extraction configuration.
    file_handler : optional
        File handler used to extract sample metadata from filenames.
    """

    def __init__(
        self,
        config: ExtractionConfig,
        file_handler=None,
    ):
        self.config = config
        self.file_handler = file_handler or build_file_handler()
        self.preprocessor = ImagePreprocessor(config)

    def compute_basic_metrics(
        self,
        label_array: np.ndarray,
        intensity_image: np.ndarray,
    ) -> pd.DataFrame:
        """Compute area and basic intensity statistics for each label."""
        props = measure.regionprops_table(
            label_array,
            intensity_image=intensity_image,
            properties=(
                "label",
                "area",
                "mean_intensity",
                "max_intensity",
                "min_intensity",
            ),
        )
        return pd.DataFrame(props)

    def compute_sum_intensity(
        self,
        metrics_df: pd.DataFrame,
        label_array: np.ndarray,
        intensity_image: np.ndarray,
    ) -> pd.DataFrame:
        """Compute per-instance intensity sums."""
        labels = metrics_df["label"].to_numpy(dtype=np.int32)
        sums = np.bincount(
            label_array.ravel(),
            weights=intensity_image.ravel(),
            minlength=int(labels.max()) + 1,
        )
        metrics_df["sum_intensity"] = sums[labels]
        return metrics_df

    def compute_percentiles(
        self,
        metrics_df: pd.DataFrame,
        label_array: np.ndarray,
        intensity_image: np.ndarray,
    ) -> pd.DataFrame:
        """Compute contract and caller-requested percentiles for each label."""
        labels = metrics_df["label"].to_numpy(dtype=np.int32)
        foreground_mask = label_array > 0
        if not np.any(foreground_mask):
            for percentile in self.config.effective_percentiles:
                metrics_df[f"percentile_{percentile}"] = 0.0
            return metrics_df

        grouped = pd.Series(
            intensity_image[foreground_mask].ravel(),
            index=label_array[foreground_mask].ravel(),
        )
        quantiles = grouped.groupby(level=0).quantile(
            np.array(self.config.effective_percentiles) / 100.0
        )
        quantiles = quantiles.unstack(level=1).reindex(labels)

        for percentile in self.config.effective_percentiles:
            metrics_df[f"percentile_{percentile}"] = quantiles[
                percentile / 100.0
            ].to_numpy()

        return metrics_df

    def process_single_image(
        self,
        img_path: str | Path,
        lbl_path: str | Path | None,
    ) -> tuple[str | None, pd.DataFrame | None]:
        """Extract metrics for one image/label pair."""
        image_path = Path(img_path)
        if lbl_path is None:
            return f"No label found for {image_path.name}", None

        label_path = Path(lbl_path)
        image = ensure_2d(tiff.imread(str(image_path)))
        labels = ensure_2d(tiff.imread(str(label_path)))

        image = image.astype(np.float32, copy=False)
        labels = labels.astype(np.int32, copy=False)

        if image.shape != labels.shape:
            return (
                f"Shape mismatch for {image_path.name}: {image.shape} vs {labels.shape}",
                None,
            )

        image = self.preprocessor.preprocess(image)
        metrics_df = self.compute_basic_metrics(labels, image)

        if metrics_df.empty:
            return None, None

        metrics_df = self.compute_sum_intensity(metrics_df, labels, image)
        metrics_df = self.compute_percentiles(metrics_df, labels, image)
        metrics_df = metrics_df.rename(columns={"label": "cell_id"})
        metrics_df = metrics_df[metrics_df["area"] >= self.config.min_area_px].copy()

        if metrics_df.empty:
            return None, None

        metadata = extract_image_metadata(image_path, self.file_handler)
        metrics_df["image_path"] = str(image_path.resolve())
        metrics_df["label_path"] = str(label_path.resolve())
        metrics_df["sample_id"] = metadata["sample_id"]
        metrics_df["z_index"] = metadata["z_index"]
        metrics_df["timepoint"] = metadata["timepoint"]
        metrics_df["image"] = image_path.name
        metrics_df["sample"] = metadata["sample_id"]
        metrics_df["time"] = metadata["timepoint"]
        metrics_df["ID"] = metadata["unique_id"]
        return None, finalize_metrics_dataframe(metrics_df)

    def process_batch_images(
        self,
        img_paths: list[str | Path],
        lbl_paths: list[str | Path | None] | None = None,
        show_progress: bool = True,
        n_jobs: int | None = None,
    ) -> pd.DataFrame:
        """Extract metrics for a batch of images."""
        if lbl_paths is None:
            lbl_paths = [find_label_from_mcherry_path(Path(path)) for path in img_paths]

        if len(img_paths) != len(lbl_paths):
            raise ValueError("image paths and label paths must have the same length")

        worker_count = self.config.n_jobs if n_jobs is None else n_jobs
        pairs = list(zip(img_paths, lbl_paths))

        if worker_count == 1:
            results = [
                self.process_single_image(image_path, label_path)
                for image_path, label_path in tqdm(
                    pairs,
                    total=len(pairs),
                    disable=not show_progress,
                )
            ]
        else:
            results = Parallel(n_jobs=worker_count)(
                delayed(self.process_single_image)(image_path, label_path)
                for image_path, label_path in tqdm(
                    pairs,
                    total=len(pairs),
                    disable=not show_progress,
                )
            )

        records = []
        for warning, result in results:
            if warning is not None:
                logger.warning("%s", warning)
            if result is not None:
                records.append(result)

        if not records:
            return pd.DataFrame(columns=finalize_metrics_dataframe(pd.DataFrame()).columns)

        return pd.concat(records, ignore_index=True)