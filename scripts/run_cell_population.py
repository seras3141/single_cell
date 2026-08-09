#!/usr/bin/env python
"""CLI: segmentation-only cell-population-over-time for one experiment.

Reads the segmentation-derived ``instance_metrics.csv`` (mCherry-free columns only),
computes per-(well, timepoint) distinct-cell count + coverage fraction, annotates
wells via the plate layout, and writes a CSV + per-well curves (DMSO overlaid).

See docs/dataset_analysis/plan_cell_population_over_time.md.
"""
import argparse
import glob
import logging
from pathlib import Path

import tifffile

from src.dataset_analysis.cell_population import compute_cell_population, plot_population
from src.dataset_analysis.layout import load_plate_layout
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def _infer_fov_pixels(final_2d_dir: Path) -> int:
    """Infer FOV (H*W) from the first final_2d mask found."""
    hits = sorted(glob.glob(str(final_2d_dir / "*_pred_mask.tif")))
    if not hits:
        raise FileNotFoundError(f"no *_pred_mask.tif under {final_2d_dir} to infer FOV")
    arr = tifffile.imread(hits[0])
    h, w = arr.shape[-2], arr.shape[-1]
    return int(h) * int(w)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment-dir", required=True,
                    help="processed experiment dir (contains mcherry_metrics/ + inference_tracked/)")
    ap.add_argument("--model", default="cellpose_sam")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dmso-well", default=None, help="e.g. N11 (HD/SA) or M11 (Ew2)")
    ap.add_argument("--layout", default="config/MF5v1_plate_layout.json")
    ap.add_argument("--fov-pixels", type=int, default=None,
                    help="override FOV size; default inferred from a final_2d mask")
    args = ap.parse_args()

    setup_logging()
    exp = Path(args.experiment_dir)
    metrics = exp / "mcherry_metrics" / args.model / "instance_metrics.csv"
    final_2d = exp / "inference_tracked" / args.model / "final_2d"
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fov = args.fov_pixels or _infer_fov_pixels(final_2d)
    layout = load_plate_layout(args.layout) if Path(args.layout).exists() else None
    logger.info("cell population: %s (fov=%d, dmso=%s)", metrics, fov, args.dmso_well)

    df = compute_cell_population(metrics, fov_pixels=fov, layout=layout,
                                 dmso_well=args.dmso_well)
    csv_path = out / "cell_population.csv"
    df.to_csv(csv_path, index=False)
    logger.info("wrote %s (%d well x timepoint rows, %d wells)",
                csv_path, len(df), df["sample_id"].nunique())

    label = exp.name.split(" ")[0]
    plot_population(df, "n_cells", out / "n_cells_over_time.png",
                    title=f"{label} — distinct cells / timepoint", dmso_well=args.dmso_well)
    plot_population(df, "coverage_fraction", out / "coverage_over_time.png",
                    title=f"{label} — coverage fraction / timepoint", dmso_well=args.dmso_well)
    logger.info("wrote n_cells_over_time.png + coverage_over_time.png -> %s", out)


if __name__ == "__main__":
    main()
