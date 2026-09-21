"""Per-(well, timepoint) population metrics measured on the z0 projection masks.

The z0 masks are raw Cellpose output, written *before* 3D combine, blur filtering and
tracking — the stages suspected of destroying detections. Measuring them answers whether
the distinct-cell-count collapse behind ``t_cross`` reproduces at the earliest possible
point in the pipeline.

Two numbers per (well, timepoint) do the work:

* ``n_objects`` — distinct non-zero labels. Deliberately **not** named ``n_cells``: this
  counts projection objects from raw inference, whereas ``cell_population.n_cells``
  counts 3D-linked tracked identities. They are different quantities at different
  pipeline stages and must never be quoted against each other (see the plan's §6.4).
  Only the *normalised shape over time* is comparable.
* ``coverage_fraction`` — non-zero pixels over the frame. On a projection this is the
  cleanest confluence signal available: it cannot double-count across z, and genuine
  confluence must drive it up.

Their ratio, ``mean_object_area_fraction``, is the discriminator. Merging (real
aggregation) conserves foreground while reducing the count, so it must inflate mean
object size; detection loss reduces both and leaves the ratio flat.

Rows that could not be measured, or whose label width sits on an integer boundary, stay
in the table for auditability but must be removed with :func:`drop_unmeasurable` before
anything consumes the counts.

See ``docs/dataset_analysis/plan_z0_projection_count_check.md``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from src.dataset_analysis.layout import get_well_annotation
from src.dataset_analysis.z0_tree import iter_z0_entries
from src.utils.image_utils import load_labels

logger = logging.getLogger(__name__)

#: Filename suffix identifying a segmentation mask. Without this filter, pointing at
#: ``split_data/`` (which sits beside the masks in the staged tree and holds
#: ``_BF``/``_mCherry``/``_FlipGFP`` intensity images) would measure grey levels as
#: labels and return plausible-looking nonsense instead of an error.
MASK_SUFFIX = "pred_mask"

#: Label-width boundaries. A distinct count or max label sitting on one of these is
#: indistinguishable from an integer-width wrap and must not be reported as a
#: measurement (plan §6.2). z0 is the densest plane in every stack, so it is the most
#: likely place for such a problem to surface.
SATURATION_BOUNDARIES = {
    "uint8": 255,
    "uint16": 65535,
    "uint32": 4294967295,
}

#: The count-side leg of the same check: a wrap that happens to miss the max-label
#: boundary still lands the *object count* on one of these.
SATURATION_COUNTS = frozenset({255, 256, 65535, 65536})

OUTPUT_COLUMNS = (
    "sample_id",
    "timepoint",
    "ti",
    "n_objects",
    "coverage_fraction",
    "mean_object_area_fraction",
    "max_label",
    "dtype",
    "dtype_saturated",
    "fov_pixels",
    "error",
)


def measure_z0_mask(mask_path: Union[str, Path]) -> Dict[str, Any]:
    """Measure one z0 mask: object count, coverage, mean object size, width audit.

    Label 0 is background and is excluded from the count. Labels may be non-contiguous
    after relabeling, so the count comes from :func:`numpy.unique`, never ``max()``.

    ``mean_object_area_fraction`` is a fraction of the field of view, **not** pixels —
    multiply by ``fov_pixels`` to compare against a pixel area such as
    ``instance_metrics.area``.

    Returns:
        Dict with ``n_objects``, ``coverage_fraction``, ``mean_object_area_fraction``,
        ``max_label``, ``dtype``, ``dtype_saturated`` and ``fov_pixels``.
        ``mean_object_area_fraction`` is NaN for an all-background frame.

    Raises:
        ValueError: If the mask is not 2-D after squeezing. Measuring a volume would
            silently divide coverage by the plane count, and coverage is the plan's
            primary confluence signal — a wrong answer, not a cosmetic problem.
    """
    arr = np.squeeze(load_labels(mask_path))
    if arr.ndim != 2:
        raise ValueError(
            f"expected a 2-D projection mask, got shape {arr.shape}: {mask_path}"
        )

    labels = np.unique(arr)
    has_background = bool(labels.size) and labels[0] == 0
    n_objects = int(labels.size - (1 if has_background else 0))
    max_label = int(labels[-1]) if labels.size else 0

    fov_pixels = int(arr.size)
    coverage = float(np.count_nonzero(arr)) / fov_pixels if fov_pixels else float("nan")
    mean_area = coverage / n_objects if n_objects else float("nan")

    dtype_name = str(arr.dtype)
    boundary = SATURATION_BOUNDARIES.get(dtype_name)
    saturated = (boundary is not None and max_label >= boundary) or (
        n_objects in SATURATION_COUNTS
    )

    return {
        "n_objects": n_objects,
        "coverage_fraction": coverage,
        "mean_object_area_fraction": mean_area,
        "max_label": max_label,
        "dtype": dtype_name,
        "dtype_saturated": saturated,
        "fov_pixels": fov_pixels,
    }


def _failed_measurement(message: str) -> Dict[str, Any]:
    """A measurement row standing in for a mask that could not be read."""
    return {
        "n_objects": float("nan"),
        "coverage_fraction": float("nan"),
        "mean_object_area_fraction": float("nan"),
        "max_label": float("nan"),
        "dtype": None,
        "dtype_saturated": False,
        "fov_pixels": float("nan"),
        "error": message,
    }


def compute_z0_population(
    masks_dir: Union[str, Path],
    layout: Optional[Mapping[str, Any]] = None,
    dmso_well: Optional[str] = None,
    z_index: int = 0,
    log_every: int = 100,
    mask_suffix: str = MASK_SUFFIX,
    wells: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Measure every z0 mask under ``masks_dir`` into one per-(well, timepoint) table.

    A mask that cannot be read yields a NaN row carrying the error message rather than
    aborting the sweep, so one truncated store does not cost a whole 1620-file run.

    Args:
        masks_dir: Directory of ``*_pred_mask.zarr`` masks — the **raw**
            ``inference/<model>/masks`` output, not a filtered or tracked tree.
        layout: Optional loaded plate layout used to annotate each well with its drug.
        dmso_well: Optional well id flagged as the vehicle reference.
        z_index: Plane to measure; defaults to the projection.
        log_every: Emit a progress line every N masks.
        mask_suffix: Filename suffix a file must carry to be measured.
        wells: Restrict to these well ids. ``None`` measures every well present.

    Returns:
        One row per (``sample_id``, ``timepoint``), sorted by well then time, with
        :data:`OUTPUT_COLUMNS` plus ``drug``/``is_dmso`` when those inputs are given.
        Pass it through :func:`drop_unmeasurable` before consuming the counts.

    Raises:
        FileNotFoundError: If ``masks_dir`` does not exist or holds no matching masks.
        ValueError: If two files map to the same (well, timepoint).
    """
    root = Path(masks_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"masks directory not found: {root}")

    wanted = {str(w).upper() for w in wells} if wells is not None else None
    entries = [
        (path, parsed)
        for path, parsed in iter_z0_entries(root, z_index=z_index)
        if parsed.suffix == mask_suffix and (wanted is None or parsed.well in wanted)
    ]
    if not entries:
        if wanted is not None:
            raise FileNotFoundError(
                f"no z{z_index} '{mask_suffix}' files for well(s) "
                f"{sorted(wanted)} under {root} — check the well ids; the directory "
                f"itself may be fine"
            )
        raise FileNotFoundError(
            f"no z{z_index} '{mask_suffix}' files under {root} — is this the "
            f"inference/<model>/masks directory?"
        )

    records: List[Dict[str, Any]] = []
    for i, (path, parsed) in enumerate(entries, start=1):
        try:
            measurement = measure_z0_mask(path)
            measurement["error"] = None
        except Exception as exc:  # one bad store must not cost the whole sweep
            logger.warning("could not measure %s: %s", path.name, exc)
            measurement = _failed_measurement(str(exc))
        # `parsed.timepoint` is already an int from the filename regex, so the
        # float-timepoint / `str.isdigit()` trap that produced ti = -1 elsewhere
        # (plan §6.8) cannot arise here.
        records.append(
            {
                "sample_id": parsed.well,
                "timepoint": parsed.timepoint,
                "ti": parsed.timepoint,
                **measurement,
            }
        )
        if log_every and i % log_every == 0:
            logger.info("measured %d/%d z%d masks", i, len(entries), z_index)

    df = pd.DataFrame(records, columns=list(OUTPUT_COLUMNS))

    duplicated = df.duplicated(["sample_id", "ti"], keep=False)
    if duplicated.any():
        offenders = df.loc[duplicated, ["sample_id", "ti"]].drop_duplicates()
        raise ValueError(
            f"{int(duplicated.sum())} rows share a (well, timepoint) key — the "
            f"directory holds more than one mask per frame: "
            f"{offenders.to_dict('records')[:5]}"
        )

    if layout is not None:
        df["drug"] = df["sample_id"].map(lambda w: _drug_for_well(w, layout))
    if dmso_well is not None:
        df["is_dmso"] = df["sample_id"].str.upper() == str(dmso_well).upper()

    n_failed = int(df["error"].notna().sum())
    if n_failed:
        logger.warning("%d/%d masks could not be measured", n_failed, len(df))
    n_saturated = int(df["dtype_saturated"].sum())
    if n_saturated:
        logger.warning(
            "%d/%d frames sit on an integer-width boundary — treat their counts as "
            "unmeasured, not as data (plan 6.2)",
            n_saturated,
            len(df),
        )

    return df.sort_values(["sample_id", "ti"]).reset_index(drop=True)


def _drug_for_well(well: str, layout: Mapping[str, Any]) -> Optional[str]:
    """Drug (or plate content) for a well id, or None when it cannot be annotated.

    Mirrors ``cell_population``'s annotation exactly so the two tables stack: a control
    well therefore reads ``"control"`` here, and its vehicle identity is carried by the
    separate ``is_dmso`` flag rather than by this column.
    """
    text = str(well)
    row, col = text[:1], text[1:]
    if not col.isdigit():
        return None
    try:
        annotation = get_well_annotation(row, int(col), layout)
    except Exception:  # unknown row/col — leave unannotated rather than fail the run
        return None
    return annotation.get("drug") or annotation.get("content")


def saturation_report(df: pd.DataFrame) -> pd.DataFrame:
    """Frames flagged by :data:`SATURATION_BOUNDARIES`, for the plan's 6.2 audit."""
    if "dtype_saturated" not in df.columns:
        raise ValueError("frame has no 'dtype_saturated' column")
    columns = ["sample_id", "timepoint", "n_objects", "max_label", "dtype"]
    return df.loc[df["dtype_saturated"], columns].reset_index(drop=True)


def drop_unmeasurable(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows whose counts must not be consumed as measurements.

    That is any frame flagged by the integer-width audit (plan §6.2, "never report such
    a value as a measurement") and any frame that could not be read at all. Call this
    before feeding the table to ``compute_collapse_metrics`` or any figure; the full
    table is kept on disk so the exclusions stay auditable.
    """
    for column in ("dtype_saturated", "error"):
        if column not in df.columns:
            raise ValueError(f"frame has no {column!r} column")
    keep = ~df["dtype_saturated"].fillna(False).astype(bool) & df["error"].isna()
    dropped = int((~keep).sum())
    if dropped:
        logger.info("excluding %d unmeasurable frame(s) of %d", dropped, len(df))
    return df.loc[keep].reset_index(drop=True)


def compute_multiz_population(
    masks_dir: Union[str, Path],
    z_indices: Sequence[int],
    wells: Optional[Sequence[str]] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Measure several z planes, returning one frame with a ``z_index`` column.

    Used to localise where detections are lost: the projection (z0) and the optical
    slices (z1..z20) are segmented independently, so measuring both says whether the
    loss is specific to the projection or already present in the raw planes.

    Args:
        masks_dir: Directory of masks — typically the **original** processed tree's
            ``inference/<model>/masks``, since the staged z0 tree holds only z0.
        z_indices: Planes to measure.
        wells: Restrict to these wells. A 21-plane sweep of every well is 6804 reads
            per experiment, so callers normally pass just the controls.
        **kwargs: Forwarded to :func:`compute_z0_population`.

    Returns:
        The per-plane tables concatenated, with ``z_index`` identifying each.

    Raises:
        FileNotFoundError: If no plane in ``z_indices`` yields any mask.
    """
    frames = []
    for z_index in z_indices:
        try:
            frame = compute_z0_population(
                masks_dir, z_index=z_index, wells=wells, **kwargs
            )
        except FileNotFoundError:
            logger.warning("no masks at z=%d under %s — skipped", z_index, masks_dir)
            continue
        frame["z_index"] = z_index
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(
            f"no masks at any of z={list(z_indices)} in {masks_dir}"
        )
    return pd.concat(frames, ignore_index=True)
