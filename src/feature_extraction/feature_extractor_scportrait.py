"""
Feature extraction using scportrait package.

This module provides feature extraction functions using the scportrait library,
following the official workflow for single-cell image featurization.

For brightfield-only workflows, use CytosolOnlySegmentationCellpose (whole-image
Cellpose cyto3 segmentation) with MLClusterClassifier for deep feature extraction.
"""

import logging
from pathlib import Path
from typing import Any, List, Dict, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; safe for headless/script use
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scportrait.pipeline.project import Project
from scportrait.pipeline.featurization import CellFeaturizer, MLClusterClassifier
from scportrait.pipeline.extraction import HDF5CellExtraction
from scportrait.pipeline.segmentation.workflows import (
    ShardedCytosolSegmentationCellpose,
    CytosolOnlySegmentationCellpose,
)
from scportrait.pipeline.selection import LMDSelection

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _plot_segmentation(project: Project, plots_dir: Path, label: str) -> None:
    """Save a segmentation overview figure.

    Renders the input brightfield image alongside the cytosol mask. Falls back
    gracefully if the expected sdata keys are absent.

    Parameters
    ----------
    project : Project
        Segmented scPortrait project.
    plots_dir : Path
        Directory where the figure will be saved.
    label : str
        Used in the figure title and output file name.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"Segmentation – {label}")
    try:
        sdata = project.sdata
        img_keys = list(sdata.images.keys())
        if img_keys:
            img = np.array(sdata.images[img_keys[0]])
            if img.ndim == 3:
                img = img[0]
            axes[0].imshow(img, cmap="gray")
            axes[0].set_title("Brightfield input")
            axes[0].axis("off")
        label_keys = list(sdata.labels.keys())
        if label_keys:
            mask = np.array(sdata.labels[label_keys[0]])
            axes[1].imshow(mask, cmap="nipy_spectral", interpolation="nearest")
            axes[1].set_title(f"Mask – {label_keys[0]} ({mask.max()} cells)")
            axes[1].axis("off")
    except Exception as exc:
        log.warning("Could not render segmentation figure: %s", exc)
    plt.tight_layout()
    out = plots_dir / f"{label}_segmentation.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("Segmentation figure saved: %s", out)


def _plot_extraction(project: Project, plots_dir: Path, label: str) -> None:
    """Save a grid of extracted single-cell crops (up to 64).

    Reads cells from the HDF5 archive produced by HDF5CellExtraction.

    Parameters
    ----------
    project : Project
        Extracted scPortrait project.
    plots_dir : Path
        Directory where the figure will be saved.
    label : str
        Used in the figure title and output file name.
    """
    import h5py

    h5_candidates = sorted(Path(project.directory).rglob("*.h5"))
    if not h5_candidates:
        log.warning("No HDF5 extraction file found; skipping extraction plot.")
        return
    try:
        with h5py.File(h5_candidates[0], "r") as f:
            def _find_crops(group):
                for key in group:
                    item = group[key]
                    if isinstance(item, h5py.Dataset) and item.ndim >= 3:
                        return item
                    if isinstance(item, h5py.Group):
                        result = _find_crops(item)
                        if result is not None:
                            return result
                return None

            dataset = _find_crops(f)
            if dataset is None:
                log.warning("Could not locate crop dataset; skipping extraction plot.")
                return
            n_cells = dataset.shape[0]
            n_show = min(64, n_cells)
            crops = dataset[:n_show]
            if crops.ndim == 4:
                crops = crops[:, 0]  # channel_selection=0 → brightfield channel
    except Exception as exc:
        log.warning("Could not read HDF5 crops: %s", exc)
        return

    grid_size = int(np.ceil(np.sqrt(n_show)))
    fig, axes = plt.subplots(
        grid_size, grid_size, figsize=(grid_size * 1.5, grid_size * 1.5)
    )
    fig.suptitle(f"Extracted crops (first {n_show} of {n_cells}) – {label}")
    axes = np.array(axes).reshape(-1)
    for idx, ax in enumerate(axes):
        if idx < n_show:
            ax.imshow(crops[idx], cmap="gray", interpolation="nearest")
        ax.axis("off")
    plt.tight_layout()
    out = plots_dir / f"{label}_extraction.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("Extraction figure saved: %s", out)


def _plot_featurization(project: Project, plots_dir: Path, label: str) -> None:
    """Save a feature-distribution histogram for the MLClusterClassifier output.

    Parameters
    ----------
    project : Project
        Featurized scPortrait project.
    plots_dir : Path
        Directory where the figure will be saved.
    label : str
        Used in the figure title and output file name.
    """
    try:
        sdata = project.sdata
        table_keys = [k for k in sdata.tables.keys() if "MLClusterClassifier" in k]
        if not table_keys:
            table_keys = list(sdata.tables.keys())
        if not table_keys:
            log.warning("No featurization table found in sdata; skipping feature plot.")
            return
        df = sdata.tables[table_keys[0]].to_df()
        if df.empty:
            log.warning("Featurization table is empty; skipping feature plot.")
            return
        cell_means = df.mean(axis=1)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(cell_means, bins=40, color="steelblue", edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Mean encoder activation")
        ax.set_ylabel("Cell count")
        ax.set_title(f"Feature distribution ({len(df)} cells) – {label}")
        plt.tight_layout()
        out = plots_dir / f"{label}_featurization.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        log.info("Featurization figure saved: %s", out)
    except Exception as exc:
        log.warning("Could not render featurization figure: %s", exc)


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------

def get_scportrait_features(
    image_paths: List[str],
    channel_names: List[str],
    config_path: str,
    project_location: str,
    overwrite: bool = True,
    debug: bool = False,
    segmentation_f: Any = CytosolOnlySegmentationCellpose,
    extraction_f: Any = HDF5CellExtraction,
    featurization_f: Any = MLClusterClassifier,
    selection_f: Optional[Any] = None,
    mask_path: Optional[str] = None,
    plots_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run scportrait workflow to extract features from single-cell images.

    Defaults to CytosolOnlySegmentationCellpose (whole-image Cellpose cyto3)
    and MLClusterClassifier for brightfield-only featurization. Pass
    segmentation_f=ShardedCytosolSegmentationCellpose for large tiled images.

    CytosolOnlySegmentationCellpose requires exactly 2 input channels. For
    BF-only inputs, pass the same BF path twice with distinct channel names
    (e.g. ["brightfield", "brightfield_ch1"]).

    Parameters
    ----------
    image_paths : list of str
        List of image file paths, one per channel (same FOV). For BF-only
        inputs, duplicate the single BF path to satisfy the 2-channel
        requirement of CytosolOnlySegmentationCellpose.
    channel_names : list of str
        Names of channels in order.
    config_path : str
        Path to scportrait config YAML.
    project_location : str
        Directory for scportrait project output.
    overwrite : bool, optional
        Overwrite existing project files (default True).
    debug : bool, optional
        Enable debug mode (default False).
    segmentation_f : callable, optional
        Segmentation workflow class. Defaults to CytosolOnlySegmentationCellpose
        (whole-image, suitable for standard FOV sizes).
    extraction_f : callable, optional
        Extraction workflow class.
    featurization_f : callable, optional
        Featurization workflow class. Defaults to MLClusterClassifier, which
        produces encoder embeddings for each single-cell crop.
    selection_f : callable or None, optional
        LMD selection workflow class. Pass None (default) to skip selection.
    mask_path : str or None, optional
        Path to a pre-computed segmentation mask TIF. Currently unused.
        # TODO: if mask_path is provided, skip scPortrait segmentation and
        # inject the external mask directly into the project's sdata so that
        # extract() and featurize() use the user-supplied labels instead.
    plots_dir : str or None, optional
        Directory to save diagnostic figures. If None, no figures are saved.

    Returns
    -------
    pd.DataFrame
        DataFrame of extracted features per cell.
    """
    _plots = Path(plots_dir) if plots_dir else None
    if _plots is not None:
        _plots.mkdir(parents=True, exist_ok=True)

    project_kwargs: Dict[str, Any] = {
        "segmentation_f": segmentation_f,
        "extraction_f": extraction_f,
        "featurization_f": featurization_f,
    }
    if selection_f is not None:
        project_kwargs["selection_f"] = selection_f

    project = Project(
        project_location,
        config_path=config_path,
        overwrite=overwrite,
        debug=debug,
        **project_kwargs,
    )

    log.info("Loading input images...")
    project.load_input_from_tif_files(image_paths, channel_names=channel_names)
    log.info("Input loaded (%d channel(s)).", len(image_paths))

    log.info("Running segmentation (%s)...", segmentation_f.__name__)
    project.segment()
    log.info("Segmentation complete.")
    if _plots is not None:
        _plot_segmentation(project, _plots, Path(project_location).name)

    log.info("Running extraction (image_size from config)...")
    project.extract()
    log.info("Extraction complete.")
    if _plots is not None:
        _plot_extraction(project, _plots, Path(project_location).name)

    log.info("Running featurization (%s)...", featurization_f.__name__)
    project.featurize(overwrite=True)
    log.info("Featurization complete.")
    if _plots is not None:
        _plot_featurization(project, _plots, Path(project_location).name)

    # Retrieve the MLClusterClassifier results table from sdata.
    classifier_keys = [k for k in project.sdata.tables.keys() if "MLClusterClassifier" in k]
    if not classifier_keys:
        raise RuntimeError("No MLClusterClassifier results found in project.sdata.tables.")
    result_key = classifier_keys[0]
    results = project.sdata.tables[result_key].to_df()
    results["scportrait_cell_id"] = project.sdata.tables[result_key].obs["scportrait_cell_id"]
    return results


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="scPortrait BF-only single-cell feature extraction."
    )
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    # Example: brightfield-only extraction from Plate 2426 subset.
    # CytosolOnlySegmentationCellpose requires 2 input channels; duplicate the
    # BF TIF so both channel slots receive brightfield data.
    base_dir = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "Plate 2426_new_preprocessed_2D_split"
        / "subset"
    )
    bf_tif = str(base_dir / "p2426_B01_z10_BF.tif")
    image_paths = [bf_tif, bf_tif]
    channel_names = ["brightfield", "brightfield_ch1"]

    project_location = str(
        Path(__file__).resolve().parents[2] / "tmp" / "scportrait_bf_only_project"
    )
    config_path = str(Path(__file__).resolve().parent / "config_scportrait.yaml")
    plots_dir = str(Path(project_location) / "plots")

    features = get_scportrait_features(
        image_paths=image_paths,
        channel_names=channel_names,
        config_path=config_path,
        project_location=project_location,
        overwrite=True,
        debug=args.debug,
        segmentation_f=CytosolOnlySegmentationCellpose,
        extraction_f=HDF5CellExtraction,
        featurization_f=MLClusterClassifier,
        plots_dir=plots_dir,
    )

    print("Extracted features:")
    print(features.head())
