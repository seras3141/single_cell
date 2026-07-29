"""Representative-slice-per-cell selection (the ``inference_filtered/`` branch).

An alternative to the tracked pipeline: link raw ``masks_3d`` slices across z with
trackpy (as a z-linker only, ``min_track_length=1``), then per physical cell apply a
blur gate + a configurable max-area / max-sharpness chooser (``selection_metric``) to
keep ONE representative z-slice. Only the selected
slices' masks are written (per-z 2D, each cell at exactly one z, label = ``cell_id``),
plus one auditable ``selection.csv``. Feature/mCherry extraction then run unchanged
against the output -> one row per cell.

Selection is computed on the **brightfield channel only** (never mCherry — that would
leak the downstream regression target). See
``docs/feature_to_mcherry/plan_representative_slice_per_cell.md``.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.ndimage import binary_erosion, laplace
from skimage.measure import regionprops

from src.postprocessing.blur_filtering import BlurFilter
from src.postprocessing.cell_tracking import CellTracker3D
from src.utils.blur_measure import get_or_compute_blur_heatmap
from src.utils.config_schemas import (
    FilterConfig,
    RepresentativeSliceConfig,
    TrackingConfig,
)
from src.utils.file_utils import ConfigurableFileHandler
from src.utils.image_utils import LABEL_FORMATS, load_image, load_labels, save_labels

logger = logging.getLogger(__name__)

# selection.csv key columns match feature_to_mcherry's CELL_KEY (names hardcoded here
# rather than imported, to keep postprocessing independent of that downstream module).
KEY_COLUMNS = ["sample_id", "timepoint", "z_index", "cell_id"]
SELECTION_COLUMNS = KEY_COLUMNS + ["x", "y", "area", "sharpness", "selected"]
METRIC_COLUMNS = ["cell_id", "z_stack_index", "x", "y", "area", "sharpness"]


def _count_detections(mask_stack_3d: np.ndarray) -> int:
    """Total per-slice detections (label values are per-slice, so count per z)."""
    return int(
        sum(
            np.count_nonzero(np.unique(mask_stack_3d[z]))
            for z in range(mask_stack_3d.shape[0])
        )
    )


def _blur_heatmap_for_stack(
    bf_path: Path, prefix: str, config: RepresentativeSliceConfig
) -> np.ndarray:
    """Read the cached blur heatmap for this stack, or compute it from the BF stack.

    Faithful to the tracked pipeline: ``get_or_compute_blur_heatmap`` reads the cached
    ``{prefix}_BF_3d_blur_heatmap.tif`` (already [0,1]-normalized) if present, else
    computes it (normalized) from the BF stack with the same patch/stride as
    ``postprocessing_config.yaml`` (32/8).
    """
    cached = None
    if config.blur_heatmap_dir:
        cached = str(
            Path(config.blur_heatmap_dir) / f"{prefix}_BF_3d_blur_heatmap.tif"
        )
    return get_or_compute_blur_heatmap(
        bf_path, blur_path=cached, patch_size=32, stride_size=8, normalize=True
    )


def apply_blur_filter(
    mask_stack_3d: np.ndarray,
    blur_heatmap_3d: np.ndarray,
    config: RepresentativeSliceConfig,
) -> np.ndarray:
    """Drop blurry detections (absolute blur-map threshold) BEFORE linking.

    Mirrors the tracked pipeline's ``filter_before_tracking`` step: a per-detection mean
    over the blur heatmap is thresholded (``blur_threshold``, ``invert_threshold``) and
    failing detections are zeroed. This removes junk/off-focus *cells* from the
    population (the per-cell relative blur gate in the selector never does this).
    """
    if blur_heatmap_3d.shape != mask_stack_3d.shape:
        raise ValueError(
            "blur heatmap shape mismatch: "
            f"{blur_heatmap_3d.shape} vs masks {mask_stack_3d.shape}"
        )
    blur_filter = BlurFilter(
        FilterConfig(
            blur_threshold=config.blur_threshold,
            invert_threshold=config.blur_invert_threshold,
        )
    )
    filtered_stack, _ = blur_filter.filter_3d_stack_fast(
        mask_stack_3d, blur_heatmap_3d, n_jobs=1
    )
    return filtered_stack.astype(mask_stack_3d.dtype)


def link_slices(
    mask_stack_3d: np.ndarray, config: RepresentativeSliceConfig
) -> np.ndarray:
    """Link raw per-z labels across z into a z-consistent ``cell_id`` stack.

    Uses ``CellTracker3D`` purely as a z-linker with ``min_track_length=1`` forced so
    that short (1-2 slice) cells are kept (the tracker default of 3 drops ~58% of
    cells). ``min_area``/``max_area`` gate detections before linking (the tracker
    filters by them in ``extract_cell_properties``). No blur hard-filter is applied.
    """
    tracking_config = TrackingConfig(
        search_range=config.search_range,
        memory=config.memory,
        min_track_length=1,
        min_area=config.min_area,
        max_area=config.max_area,
    )
    return CellTracker3D(tracking_config).track_cells(mask_stack_3d)


def _interior_sharpness(
    bf_bbox_crop: np.ndarray, region_mask: np.ndarray, erosion_px: int
) -> float:
    """Variance of the Laplacian over a cell's interior (a focus measure).

    Computes the Laplacian on the RAW brightfield bbox crop (real neighbours, no
    zeroing) and takes the variance only over the eroded interior of the region — so
    it measures focus, not the mask boundary (a zeroed crop is edge-dominated).
    """
    lap = laplace(bf_bbox_crop.astype(np.float64))
    if erosion_px > 0:
        interior = binary_erosion(region_mask, iterations=erosion_px)
        if not interior.any():
            interior = region_mask
    else:
        interior = region_mask
    return float(np.var(lap[interior]))


def compute_slice_metrics(
    tracked_stack_3d: np.ndarray,
    bf_stack_3d: np.ndarray,
    config: RepresentativeSliceConfig,
) -> pd.DataFrame:
    """Per-(cell_id, z) area + interior sharpness from the BF channel only.

    ``bf_stack_3d`` is the brightfield stack — this function must never be given the
    mCherry channel (leakage guard). Returns a DataFrame with ``METRIC_COLUMNS``.
    """
    if tracked_stack_3d.ndim != 3 or bf_stack_3d.ndim != 3:
        raise ValueError(
            "tracked_stack_3d and bf_stack_3d must be 3D (z, y, x); got "
            f"{tracked_stack_3d.shape} and {bf_stack_3d.shape}"
        )
    if tracked_stack_3d.shape != bf_stack_3d.shape:
        raise ValueError(
            "tracked/BF shape mismatch: "
            f"{tracked_stack_3d.shape} vs {bf_stack_3d.shape}"
        )

    rows: List[dict] = []
    for z in range(tracked_stack_3d.shape[0]):
        labels_2d = tracked_stack_3d[z]
        bf_2d = bf_stack_3d[z]
        for region in regionprops(labels_2d):
            if region.area < config.min_area:
                continue
            sharpness = _interior_sharpness(
                bf_2d[region.slice], region.image, config.sharpness_erosion_px
            )
            centroid_y, centroid_x = region.centroid
            rows.append(
                {
                    "cell_id": int(region.label),
                    "z_stack_index": int(z),
                    "x": float(centroid_x),
                    "y": float(centroid_y),
                    "area": int(region.area),
                    "sharpness": sharpness,
                }
            )
    return pd.DataFrame(rows, columns=METRIC_COLUMNS)


def select_representative(
    metrics_df: pd.DataFrame, config: RepresentativeSliceConfig
) -> pd.DataFrame:
    """Pick one representative z-slice per cell: blur gate, then a max-metric chooser.

    For each ``cell_id``: keep slices with ``sharpness >= gate_fraction * (that cell's
    max sharpness)``, then choose the largest ``config.selection_metric`` (``area`` or
    ``sharpness``) among them; ties broken by the *other* metric, then z nearest the
    cell's area-weighted centroid z. A boolean ``selected`` column is added (all
    candidate rows are retained for audit). ``selection_metric="area"`` (the default)
    reproduces the original ``[area, sharpness, zdist]`` ordering exactly.
    """
    if metrics_df.empty:
        out = metrics_df.copy()
        out["selected"] = pd.Series(dtype=bool)
        return out

    df = metrics_df.copy()
    grouped = df.groupby("cell_id")

    # gate: sharpness >= f * (per-cell max sharpness). The max always passes, so every
    # cell has >= 1 gated candidate.
    df["_smax"] = grouped["sharpness"].transform("max")
    df["_gated"] = df["sharpness"] >= config.sharpness_gate_fraction * df["_smax"]

    # area-weighted centroid z per cell, for the final tie-break.
    df["_wz"] = df["area"] * df["z_stack_index"]
    sums = df.groupby("cell_id").agg(_wz_sum=("_wz", "sum"), _a_sum=("area", "sum"))
    awz = sums["_wz_sum"] / sums["_a_sum"].replace(0, np.nan)
    df["_awz"] = df["cell_id"].map(awz)
    df["_zdist"] = (df["z_stack_index"] - df["_awz"]).abs()

    # chooser: primary = selection_metric, secondary = the other metric, then zdist.
    primary = config.selection_metric
    secondary = "sharpness" if primary == "area" else "area"
    gated = df[df["_gated"]].sort_values(
        by=[primary, secondary, "_zdist"], ascending=[False, False, True]
    )
    winners = gated.groupby("cell_id", sort=False).head(1).index
    df["selected"] = df.index.isin(winners)

    return df.drop(columns=["_smax", "_gated", "_wz", "_awz", "_zdist"])


def write_filtered_masks(
    tracked_stack_3d: np.ndarray,
    selection_df: pd.DataFrame,
    prefix: str,
    config: RepresentativeSliceConfig,
) -> List[Path]:
    """Write per-z 2D masks with each cell present at exactly its selected z.

    Output name: ``{prefix}_z{z + z_index_offset}_pred_mask{ext}``. Every plane is
    written (including empties) so extraction finds a mask for every real BF plane.
    Hand-rolled (not ``split_3d_to_2d``, which hardcodes ``z+1`` / tif) to honour
    ``z_index_offset`` and ``output_label_format``.
    """
    final_2d_dir = Path(config.output_dir) / "final_2d"
    final_2d_dir.mkdir(parents=True, exist_ok=True)
    ext = LABEL_FORMATS[config.output_label_format]

    out = np.zeros_like(tracked_stack_3d)
    for row in selection_df[selection_df["selected"]].itertuples(index=False):
        z = int(row.z_stack_index)
        cell_id = int(row.cell_id)
        out[z][tracked_stack_3d[z] == cell_id] = cell_id

    written: List[Path] = []
    for z in range(tracked_stack_3d.shape[0]):
        z_label = z + config.z_index_offset
        path = final_2d_dir / f"{prefix}_z{z_label}_pred_mask{ext}"
        save_labels(out[z], path)
        written.append(path)
    return written


def write_all_slice_masks(
    tracked_stack_3d: np.ndarray, prefix: str, config: RepresentativeSliceConfig
) -> List[Path]:
    """Write per-z 2D masks with EVERY cell at ALL its z (diagnostic per-slice mode).

    Unlike :func:`write_filtered_masks` (each cell at its selected z), this writes the
    full linked stack split per-z — so downstream extraction yields the per-slice table
    (multiple rows per cell) for the same-mask per-slice-vs-one-per-cell comparison.
    """
    final_2d_dir = Path(config.output_dir) / "final_2d"
    final_2d_dir.mkdir(parents=True, exist_ok=True)
    ext = LABEL_FORMATS[config.output_label_format]

    written: List[Path] = []
    for z in range(tracked_stack_3d.shape[0]):
        z_label = z + config.z_index_offset
        path = final_2d_dir / f"{prefix}_z{z_label}_pred_mask{ext}"
        save_labels(tracked_stack_3d[z], path)
        written.append(path)
    return written


def _parse_stack_keys(
    mask_path: Path, handler: ConfigurableFileHandler
) -> Tuple[str, str, str]:
    """Return (sample_id, timepoint, prefix) parsed from a 3D mask filename.

    prefix is the canonical ``p<plate>_<well>_t<tp>`` stem (used to locate the BF stack
    and to name the per-z output masks). Raises ValueError if any field is unparseable.
    """
    name = mask_path.name
    sample_id = handler.extract_sample_id(name)
    timepoint = handler.extract_time_point(name)
    plate = handler.extract_plate_number(str(mask_path))
    if not sample_id or timepoint == "unknown" or not plate or plate == "unknown":
        raise ValueError(
            f"Cannot parse sample_id/timepoint/plate from {name} "
            f"(got sample_id={sample_id!r}, timepoint={timepoint!r}, plate={plate!r})"
        )
    prefix = f"p{plate}_{sample_id}_t{timepoint}"
    return sample_id, timepoint, prefix


def process_stack(
    mask_path: Path, config: RepresentativeSliceConfig
) -> pd.DataFrame:
    """Process one (well, timepoint) 3D mask stack -> its selection rows.

    Links across z, computes BF metrics, selects one slice per cell, writes the per-z
    masks, and returns the per-(cell,z) selection frame with ``SELECTION_COLUMNS``.
    Returns an empty frame (and logs) if the stack can't be processed.
    """
    mask_path = Path(mask_path)
    handler = ConfigurableFileHandler()
    try:
        sample_id, timepoint, prefix = _parse_stack_keys(mask_path, handler)
    except ValueError as exc:
        logger.warning("Skipping %s: %s", mask_path.name, exc)
        return pd.DataFrame(columns=SELECTION_COLUMNS)

    bf_path = Path(config.bf_3d_dir) / f"{prefix}_BF_3d.tif"
    if not bf_path.exists():
        logger.warning("No BF 3D stack for %s (expected %s); skipping", prefix, bf_path)
        return pd.DataFrame(columns=SELECTION_COLUMNS)

    masks = load_labels(mask_path).astype(np.int32)
    bf = load_image(bf_path)
    if masks.shape != bf.shape:
        logger.warning(
            "Shape mismatch for %s: masks %s vs BF %s; skipping",
            prefix,
            masks.shape,
            bf.shape,
        )
        return pd.DataFrame(columns=SELECTION_COLUMNS)

    if config.enable_blur_filter:
        heatmap = _blur_heatmap_for_stack(bf_path, prefix, config)
        n_before = _count_detections(masks)
        masks = apply_blur_filter(masks, heatmap, config)
        logger.info(
            "%s: blur filter kept %d/%d per-slice detections",
            prefix,
            _count_detections(masks),
            n_before,
        )

    tracked = link_slices(masks, config)
    metrics = compute_slice_metrics(tracked, bf, config)
    if metrics.empty:
        logger.warning("No cells found for %s; skipping", prefix)
        return pd.DataFrame(columns=SELECTION_COLUMNS)

    selection = select_representative(metrics, config)
    if config.emit_all_slices:
        write_all_slice_masks(tracked, prefix, config)
    else:
        write_filtered_masks(tracked, selection, prefix, config)

    frame = selection.copy()
    frame["sample_id"] = str(sample_id)
    frame["timepoint"] = str(timepoint)
    frame["z_index"] = (frame["z_stack_index"] + config.z_index_offset).astype(str)
    frame["cell_id"] = frame["cell_id"].astype(str)
    n_selected = int(frame["selected"].sum())
    logger.info(
        "%s: %d cells (%d candidate slices) -> %d representative masks",
        prefix,
        n_selected,
        len(frame),
        n_selected,
    )
    return frame[SELECTION_COLUMNS]


def _validate_paths(config: RepresentativeSliceConfig) -> None:
    """Stage-entry path validation (deferred from Phase 1 config schema)."""
    for label, directory in (
        ("input_masks_dir", config.input_masks_dir),
        ("bf_3d_dir", config.bf_3d_dir),
    ):
        if not directory:
            raise ValueError(f"representative_slice.{label} is not set")
        if not Path(directory).is_dir():
            raise ValueError(
                f"representative_slice.{label} does not exist: {directory}"
            )
    if not config.output_dir:
        raise ValueError("representative_slice.output_dir is not set")


def run(config: RepresentativeSliceConfig) -> pd.DataFrame:
    """Run representative-slice selection over every stack in ``input_masks_dir``.

    Writes per-z 2D masks under ``output_dir/final_2d/`` and one combined
    ``output_dir/selection.csv``. Returns the combined selection frame.
    """
    start = time.time()
    _validate_paths(config)
    out_dir = Path(config.output_dir)
    (out_dir / "final_2d").mkdir(parents=True, exist_ok=True)
    selection_csv = out_dir / "selection.csv"

    if selection_csv.exists() and not config.overwrite_existing:
        logger.warning(
            "selection.csv exists at %s; set overwrite_existing=true to regenerate. "
            "Skipping.",
            selection_csv,
        )
        return pd.read_csv(selection_csv)

    mask_paths = sorted(Path(config.input_masks_dir).glob(config.mask_pattern))
    if not mask_paths:
        raise ValueError(
            f"No masks matching {config.mask_pattern!r} in {config.input_masks_dir}"
        )

    if config.n_jobs is not None:
        n_jobs = config.n_jobs
    else:
        n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    logger.info("Processing %d stacks with n_jobs=%d", len(mask_paths), n_jobs)

    frames = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(process_stack)(mask_path, config) for mask_path in mask_paths
    )
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if frames:
        combined = pd.concat(frames, ignore_index=True)
    else:
        combined = pd.DataFrame(columns=SELECTION_COLUMNS)

    combined.to_csv(selection_csv, index=False)
    n_selected = int(combined["selected"].sum()) if not combined.empty else 0
    logger.info(
        "Wrote %s: %d rows, %d selected cells across %d stacks (%.1fs)",
        selection_csv,
        len(combined),
        n_selected,
        len(frames),
        time.time() - start,
    )
    return combined
