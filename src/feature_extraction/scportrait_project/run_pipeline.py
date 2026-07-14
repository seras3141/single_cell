"""
run_pipeline.py – scPortrait brightfield-only single-cell feature extraction.

Processes all *_BF.tif images found in the input directory through the full
scPortrait pipeline:
  1. CytosolOnlySegmentationCellpose  – whole-image Cellpose cyto3 segmentation
  2. HDF5CellExtraction               – 128-px single-cell crop archiving
  3. MLClusterClassifier              – deep encoder featurization (BF channel)

Usage
-----
  python run_pipeline.py [--debug]

Outputs are written to a per-sample subdirectory under PROJECT_DIR.
Diagnostic figures are saved to <project_dir>/<sample>/plots/.
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; no display required
import matplotlib.pyplot as plt
import numpy as np

from scportrait.pipeline.project import Project
from scportrait.pipeline.segmentation.workflows import CytosolOnlySegmentationCellpose
from scportrait.pipeline.extraction import HDF5CellExtraction
from scportrait.pipeline.featurization import CellFeaturizer, MLClusterClassifier

# ---------------------------------------------------------------------------
# Paths – edit INPUT_DIR and PROJECT_DIR to match your environment.
# ---------------------------------------------------------------------------

# Directory containing *_BF.tif files to process.
INPUT_DIR: Path = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "Plate 2426_new_preprocessed_2D_split"
    / "subset"
)

# Root output directory; one subdirectory is created per input sample.
PROJECT_DIR: Path = (
    Path(__file__).resolve().parents[1] / "tmp" / "scportrait_bf_pipeline"
)

# Path to the scPortrait config YAML (same directory as this script).
CONFIG_PATH: Path = Path(__file__).resolve().parent / "config.yml"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _plot_segmentation(project: Project, plots_dir: Path, sample_name: str) -> None:
    """Save a segmentation overview figure for one sample.

    Renders the input brightfield image overlaid with cytosol mask contours.
    Falls back gracefully if the expected sdata keys are absent.

    Parameters
    ----------
    project : Project
        Initialised and segmented scPortrait project.
    plots_dir : Path
        Directory where the figure will be saved.
    sample_name : str
        Used in the figure title and output file name.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"Segmentation – {sample_name}")

    try:
        sdata = project.sdata

        # Retrieve the raw input image (channel 0 = brightfield).
        img_keys = list(sdata.images.keys())
        if img_keys:
            img = np.array(sdata.images[img_keys[0]])
            # sdata images may be (C, Y, X); take channel 0.
            if img.ndim == 3:
                img = img[0]
            axes[0].imshow(img, cmap="gray")
            axes[0].set_title("Brightfield input")
            axes[0].axis("off")

        # Retrieve cytosol segmentation mask.
        label_keys = list(sdata.labels.keys())
        if label_keys:
            # CytosolOnly produces a single label layer; take the first.
            mask = np.array(sdata.labels[label_keys[0]])
            axes[1].imshow(mask, cmap="nipy_spectral", interpolation="nearest")
            axes[1].set_title(f"Mask – {label_keys[0]} ({mask.max()} cells)")
            axes[1].axis("off")
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not render segmentation figure: %s", exc)

    plt.tight_layout()
    out = plots_dir / f"{sample_name}_segmentation.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("  Segmentation figure saved: %s", out)


def _plot_extraction(project: Project, plots_dir: Path, sample_name: str) -> None:
    """Save a grid of extracted single-cell crops for one sample.

    Reads cells from the HDF5 archive produced by HDF5CellExtraction and
    renders up to 64 crops in an 8×8 grid.

    Parameters
    ----------
    project : Project
        Initialised and extracted scPortrait project.
    plots_dir : Path
        Directory where the figure will be saved.
    sample_name : str
        Used in the figure title and output file name.
    """
    import h5py

    # HDF5CellExtraction writes to <project_dir>/extraction/cells.h5 (typical path).
    h5_candidates = sorted(Path(project.directory).rglob("*.h5"))
    if not h5_candidates:
        log.warning("No HDF5 extraction file found; skipping extraction plot.")
        return

    h5_path = h5_candidates[0]
    try:
        with h5py.File(h5_path, "r") as f:
            # Standard scPortrait HDF5 layout: /single_cells/<cell_id>/...
            # The exact key depends on the scPortrait version; we look for
            # a dataset whose first axis is the number of cells.
            def _find_crops(group: h5py.Group):
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
                log.warning("Could not locate crop dataset in %s", h5_path)
                return

            n_cells = dataset.shape[0]
            n_show = min(64, n_cells)
            # crops shape: (N, C, H, W) or (N, H, W); take channel 0.
            crops = dataset[:n_show]
            if crops.ndim == 4:
                crops = crops[:, 0]  # channel_selection=0 → brightfield channel

    except Exception as exc:  # noqa: BLE001
        log.warning("Could not read HDF5 crops: %s", exc)
        return

    grid_size = int(np.ceil(np.sqrt(n_show)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 1.5, grid_size * 1.5))
    fig.suptitle(f"Extracted crops (first {n_show} of {n_cells}) – {sample_name}")
    axes = np.array(axes).reshape(-1)
    for idx, ax in enumerate(axes):
        if idx < n_show:
            ax.imshow(crops[idx], cmap="gray", interpolation="nearest")
        ax.axis("off")

    plt.tight_layout()
    out = plots_dir / f"{sample_name}_extraction.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("  Extraction figure saved: %s", out)


def _plot_featurization(project: Project, plots_dir: Path, sample_name: str) -> None:
    """Save a feature-distribution figure for the MLClusterClassifier output.

    Retrieves the encoder embedding table from sdata and plots the distribution
    of the mean embedding value per cell as a histogram.

    Parameters
    ----------
    project : Project
        Initialised and featurized scPortrait project.
    plots_dir : Path
        Directory where the figure will be saved.
    sample_name : str
        Used in the figure title and output file name.
    """
    try:
        sdata = project.sdata
        # MLClusterClassifier results are stored in sdata.tables with a key
        # containing the classifier class name.
        table_keys = [k for k in sdata.tables.keys() if "MLClusterClassifier" in k]
        if not table_keys:
            # Fall back: any table present.
            table_keys = list(sdata.tables.keys())
        if not table_keys:
            log.warning("No featurization table found in sdata; skipping feature plot.")
            return

        table = sdata.tables[table_keys[0]]
        df = table.to_df()
        if df.empty:
            log.warning("Featurization table is empty; skipping feature plot.")
            return

        # Plot the mean embedding value per cell as a distribution summary.
        cell_means = df.mean(axis=1)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(cell_means, bins=40, color="steelblue", edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Mean encoder activation")
        ax.set_ylabel("Cell count")
        ax.set_title(f"Feature distribution ({len(df)} cells) – {sample_name}")
        plt.tight_layout()
        out = plots_dir / f"{sample_name}_featurization.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        log.info("  Featurization figure saved: %s", out)
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not render featurization figure: %s", exc)


# ---------------------------------------------------------------------------
# Per-sample pipeline
# ---------------------------------------------------------------------------

def _run_sample(bf_path: Path, debug: bool) -> None:
    """Run the full scPortrait pipeline for a single brightfield image.

    Creates a project subdirectory named after the image stem under PROJECT_DIR,
    runs segmentation, extraction, and featurization, and saves diagnostic
    plots to <project_dir>/plots/.

    Parameters
    ----------
    bf_path : Path
        Path to the brightfield TIFF file.
    debug : bool
        Enable scPortrait debug mode (verbose logging, intermediate saves).
    """
    sample_name = bf_path.stem
    sample_project_dir = PROJECT_DIR / sample_name
    plots_dir = sample_project_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("Sample: %s", sample_name)
    log.info("Input:  %s", bf_path)
    log.info("Output: %s", sample_project_dir)
    log.info("=" * 60)

    # Instantiate the project with the three workflow classes.
    # CytosolOnlySegmentationCellpose: whole-image Cellpose cytosol segmentation
    #   – no tiling needed for standard microscopy FOV sizes.
    # HDF5CellExtraction: archives 128-px crops per detected cell.
    # MLClusterClassifier: produces a latent embedding per cell using the
    #   forward encoder pass (channel_selection=0 → brightfield channel).
    project = Project(
        str(sample_project_dir),
        config_path=str(CONFIG_PATH),
        overwrite=True,
        debug=debug,
        segmentation_f=CytosolOnlySegmentationCellpose,
        extraction_f=HDF5CellExtraction,
        # featurization_f=MLClusterClassifier,    # <-- Needs custom featurizer for BF-only input; change to CellFeaturizer for a simple baseline.
        featurization_f=CellFeaturizer,  # <-- change this
    )

    # --- Load input --------------------------------------------------------
    # For BF-only, a single TIF file constitutes the full channel stack.
    log.info("[1/5] Loading input image...")
    project.load_input_from_tif_files(
        file_paths=[str(bf_path), str(bf_path)],  # same file twice
        channel_names=["Brightfield_cyto", "Brightfield_nuc"]
    )
    log.info("  Input loaded.")

    # --- Segmentation ------------------------------------------------------
    log.info("[2/5] Running segmentation (CytosolOnlySegmentationCellpose)...")
    project.segment()
    log.info("  Segmentation complete.")
    _plot_segmentation(project, plots_dir, sample_name)

    # --- Extraction --------------------------------------------------------
    log.info("[3/5] Running extraction (HDF5CellExtraction, image_size=128)...")
    project.extract()
    log.info("  Extraction complete.")
    _plot_extraction(project, plots_dir, sample_name)

    # --- Featurization -----------------------------------------------------
    log.info("[4/5] Running featurization (CellFeaturizer, channel=0)...")
    project.featurize(overwrite=True)
    log.info("  Featurization complete.")
    _plot_featurization(project, plots_dir, sample_name)

    log.info("[5/5] Pipeline finished for sample: %s", sample_name)
    log.info("  Plots saved to: %s", plots_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> None:
    """Discover brightfield images and run the scPortrait pipeline for each.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments. Expected attributes:
        - debug (bool): enable scPortrait debug mode.
    """
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)

    # Glob all brightfield TIF files in the input directory (non-recursive).
    # Only *_BF.tif files are processed; fluorescence (*_w2.tif, *_Cells.tif)
    # are intentionally excluded for this BF-only featurization workflow.
    bf_files = sorted(INPUT_DIR.glob("*_BF.tif"))
    if not bf_files:
        log.error("No *_BF.tif files found in %s", INPUT_DIR)
        sys.exit(1)

    log.info("Found %d brightfield image(s) in %s", len(bf_files), INPUT_DIR)
    log.info("Project root: %s", PROJECT_DIR)
    log.info("Config:       %s", CONFIG_PATH)

    for bf_path in bf_files:
        _run_sample(bf_path, debug=args.debug)

    log.info("All samples processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="scPortrait brightfield-only single-cell feature extraction pipeline."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable scPortrait debug mode (verbose output, intermediate saves).",
    )
    run_pipeline(parser.parse_args())
