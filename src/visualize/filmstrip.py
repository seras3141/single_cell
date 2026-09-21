"""Side-by-side BF + segmentation-mask strips across timepoints.

Answers the one question the mask-derived numbers provably cannot (see the plan's §7):
are cells *visibly present but unlabelled* (detection loss), or genuinely fused into
masses the segmentation cannot separate (aggregation)? Both look identical in a count,
and largely identical in coverage; they look nothing alike to the eye.

The strip renders BF alone on the top row and BF + mask on the bottom, so a reader can
see what the segmentation had to work with before seeing what it produced. Panels are
annotated with the frame's own count and coverage, tying the picture to the table.

Layer rendering is delegated to :func:`headless_layers.render_layers` (matplotlib Agg,
label overlay with transparent background), so this module only handles frame selection,
cropping and layout.

See ``docs/dataset_analysis/plan_z0_projection_count_check.md``.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Union

import numpy as np

from src.dataset_analysis.z0_tree import parse_processed_name
from src.utils.image_utils import load_labels
from src.visualize.headless_layers import normalize_for_display, render_layers

logger = logging.getLogger(__name__)

#: Frames span 72 h across 351 indices; 1 index = 10 min.
MINUTES_PER_INDEX = 10.0
TI_TO_HOURS = MINUTES_PER_INDEX / 60.0

#: Requested timepoints the user is most likely to reach for. The acquisition grid is
#: ``1, 11, 21, ... 351``, so round numbers like 50/100 do not exist and are snapped.
DEFAULT_TIMEPOINTS = (1, 51, 101, 151, 201)

#: Timepoints straddling the early transition, where the collapse actually happens.
EARLY_TIMEPOINTS = (1, 11, 21, 31, 41)

#: Mask rendering. ``fill`` tints each label (good for "was this cell found?");
#: ``outline`` draws only boundaries, leaving the underlying signal visible — which is
#: what you want on mCherry, where a filled overlay hides the very intensity you are
#: trying to judge.
MASK_STYLE_FILL = "fill"
MASK_STYLE_OUTLINE = "outline"

#: Boundary colour, chosen to stay legible over the inferno ramp used for mCherry.
OUTLINE_COLOR = (0.0, 1.0, 1.0)

#: Width allocated to each column, in inches.
DEFAULT_PANEL_INCHES = 3.0

#: Output resolution. ``DEFAULT_PANEL_INCHES * DEFAULT_DPI`` = 450 px per panel,
#: which is
#: fine for a crop but downsamples a full 1024x1024 field by ~2.3x. Whole-field strips
#: should raise this to ~340 so each panel carries the field's own pixels.
DEFAULT_DPI = 150

#: Suffixes that make :func:`build_filmstrip` write lossy output and accept ``quality``.
JPEG_SUFFIXES = (".jpg", ".jpeg")


class Frame(NamedTuple):
    """One column of the strip."""

    timepoint: int
    requested: int
    bf_path: Path
    mask_path: Optional[Path]


def _index_by_timepoint(
    directory: Path, well: str, suffix: str, z_index: int
) -> Dict[int, Path]:
    """Map timepoint -> path for one well's files carrying ``suffix`` at ``z_index``.

    The well filter is applied *before* indexing: a stage folder holds all nine wells,
    so keying on timepoint alone would let each well overwrite the last and leave only
    the alphabetically final one reachable.
    """
    found: Dict[int, Path] = {}
    if not directory.is_dir():
        return found
    for path in sorted(directory.iterdir()):
        parsed = parse_processed_name(path.name)
        if parsed is None or parsed.z_index != z_index or parsed.suffix != suffix:
            continue
        if parsed.well != well:
            continue
        found[parsed.timepoint] = path
    return found


def available_wells(
    directory: Union[str, Path],
    z_index: int = 0,
    suffix: str = "pred_mask",
) -> List[str]:
    """Every well id present in ``directory`` at ``z_index``, sorted."""
    wells = set()
    root = Path(directory)
    if not root.is_dir():
        return []
    for path in root.iterdir():
        parsed = parse_processed_name(path.name)
        if parsed is not None and parsed.z_index == z_index and parsed.suffix == suffix:
            wells.add(parsed.well)
    return sorted(wells)


def condition_label(well: str, annotation: Optional[Mapping[str, Any]] = None) -> str:
    """``"Navitoclax 75 µM"`` / ``"DMSO"`` / the bare well id when unannotated.

    Control wells carry ``drug == "control"`` in the shared plate annotation, with the
    vehicle identity in ``is_dmso`` — so the flag is checked before the drug name.
    """
    if not annotation:
        return str(well)
    if bool(annotation.get("is_dmso")):
        return "DMSO"
    drug = annotation.get("drug")
    # `bool(float("nan"))` is True, so an `or` fallback would print the string
    # "nan" for a well pandas left unannotated.
    if isinstance(drug, float) and math.isnan(drug):
        return str(well)
    if drug in (None, "", "control"):
        return str(drug or well)
    concentration = annotation.get("concentration_uM")
    try:
        if concentration is not None and float(concentration) == float(concentration):
            return f"{drug} {float(concentration):g} \u00b5M"
    except (TypeError, ValueError):
        pass
    return str(drug)


def resolve_frames(
    split_dir: Union[str, Path],
    masks_dir: Union[str, Path],
    well: str,
    timepoints: Sequence[int] = DEFAULT_TIMEPOINTS,
    z_index: int = 0,
    channel: str = "BF",
    mask_suffix: str = "pred_mask",
) -> List[Frame]:
    """Pick the acquired frame nearest each requested timepoint, for one well.

    The acquisition grid steps by 10 (``1, 11, 21, ...``), so a requested ``50``
    snaps to ``51``. Substitutions are logged rather than applied silently.

    Raises:
        FileNotFoundError: If the well has no BF frames at ``z_index``.
    """
    well = str(well).upper()

    bf = _index_by_timepoint(Path(split_dir), well, channel, z_index)
    masks = _index_by_timepoint(Path(masks_dir), well, mask_suffix, z_index)
    if not bf:
        raise FileNotFoundError(
            f"no z{z_index} {channel} frames for well {well} under {split_dir}"
        )

    available = sorted(bf)
    frames: List[Frame] = []
    for requested in timepoints:
        nearest = min(available, key=lambda t: (abs(t - requested), t))
        if nearest != requested:
            logger.info(
                "t=%d not acquired; using nearest frame t=%d", requested, nearest
            )
        if any(f.timepoint == nearest for f in frames):
            logger.warning(
                "t=%d snaps to t=%d, already in the strip — column skipped",
                requested,
                nearest,
            )
            continue
        if nearest not in masks:
            logger.warning("no mask for %s t=%d; BF shown alone", well, nearest)
        frames.append(Frame(nearest, requested, bf[nearest], masks.get(nearest)))
    return frames


def _outline_rgba(
    mask: np.ndarray, color: "tuple[float, float, float]" = OUTLINE_COLOR
) -> np.ndarray:
    """Label boundaries as an RGBA overlay; everything that is not a boundary is clear.

    ``mode="outer"`` traces just outside each object, so touching cells in a dense clump
    still get separate visible borders instead of one merged blob.
    """
    from skimage.segmentation import find_boundaries

    boundaries = find_boundaries(mask, mode="outer")
    rgba = np.zeros((*mask.shape[-2:], 4), dtype=float)
    rgba[boundaries, 0] = color[0]
    rgba[boundaries, 1] = color[1]
    rgba[boundaries, 2] = color[2]
    rgba[boundaries, 3] = 1.0
    return rgba


def _centre_crop(array: np.ndarray, fraction: float) -> np.ndarray:
    """Centred crop keeping ``fraction`` of each axis; ``1.0`` returns the field."""
    if not 0 < fraction <= 1:
        raise ValueError(f"crop fraction must be in (0, 1], got {fraction}")
    if fraction == 1:
        return array
    height, width = array.shape[-2], array.shape[-1]
    new_h, new_w = int(height * fraction), int(width * fraction)
    top, left = (height - new_h) // 2, (width - new_w) // 2
    return array[..., top : top + new_h, left : left + new_w]


def build_filmstrip(
    frames: Sequence[Frame],
    out_png: Union[str, Path],
    title: str = "",
    crop: float = 1.0,
    show_raw_row: bool = True,
    annotations: Optional[Mapping[int, Mapping[str, Any]]] = None,
    mask_alpha: float = 0.45,
    mask_style: str = MASK_STYLE_FILL,
    base_cmap: str = "gray",
    dpi: int = DEFAULT_DPI,
    panel_inches: float = DEFAULT_PANEL_INCHES,
    quality: Optional[int] = None,
) -> Path:
    """Render ``frames`` as a strip and save it.

    Args:
        frames: Columns, in order, from :func:`resolve_frames`.
        out_png: Destination; parent directories are created.
        title: Figure title.
        crop: Centred crop fraction. The field is 1024x1024, so five full-field
            panels leave cells only a few pixels wide — crop to inspect morphology.
        show_raw_row: Draw a BF-only row above the overlaid row.
        annotations: ``{timepoint: {"n_objects": ..., "coverage_fraction": ...}}`` used
            for panel subtitles.
        mask_alpha: Overlay opacity (``fill`` style only).
        mask_style: :data:`MASK_STYLE_FILL` or :data:`MASK_STYLE_OUTLINE`.
        base_cmap: Colormap for the underlying image — ``"gray"`` for brightfield,
            ``"inferno"`` for mCherry.
        dpi: Output resolution. ``panel_inches * dpi`` is the pixels each panel gets, so
            a full 1024x1024 field needs ~340 dpi at the default panel size to be shown
            without downsampling away single cells.
        panel_inches: Width allocated to each column.
        quality: JPEG quality (1-95) when ``out_png`` has a ``.jpg``/``.jpeg`` suffix.
            Full-field strips are large; lossy output keeps them a few MB without
            costing the detail that matters here. Ignored for PNG.

    Raises:
        ValueError: If ``frames`` is empty, ``mask_style`` is unknown, or ``quality`` is
            given for a non-JPEG destination.
    """
    if not frames:
        raise ValueError("no frames to render")
    if mask_style not in (MASK_STYLE_FILL, MASK_STYLE_OUTLINE):
        raise ValueError(
            f"mask_style must be {MASK_STYLE_FILL!r} or {MASK_STYLE_OUTLINE!r}, "
            f"got {mask_style!r}"
        )
    is_jpeg = Path(out_png).suffix.lower() in JPEG_SUFFIXES
    if quality is not None and not is_jpeg:
        raise ValueError(
            f"quality is a JPEG setting but {Path(out_png).name} is not a JPEG — "
            f"give the destination a .jpg suffix or drop quality"
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows = 2 if show_raw_row else 1
    fig, axes = plt.subplots(
        n_rows,
        len(frames),
        figsize=(panel_inches * len(frames), (panel_inches + 0.2) * n_rows + 0.6),
        squeeze=False,
    )

    import tifffile

    for column, frame in enumerate(frames):
        bf = _centre_crop(np.squeeze(tifffile.imread(str(frame.bf_path))), crop)
        mask = None
        if frame.mask_path is not None:
            mask = _centre_crop(np.squeeze(load_labels(frame.mask_path)), crop)

        if show_raw_row:
            _draw_base(axes[0][column], bf, base_cmap)
            axes[0][column].set_title(_panel_title(frame, annotations), fontsize=9)
            _strip_axis(axes[0][column], keep_frame_for_label=column == 0)

        overlay_ax = axes[n_rows - 1][column]
        if mask_style == MASK_STYLE_OUTLINE:
            _draw_base(overlay_ax, bf, base_cmap)
            if mask is not None:
                overlay_ax.imshow(_outline_rgba(mask))
        else:
            render_layers(bf, mask=mask, ax=overlay_ax, mask_alpha=mask_alpha)
        if not show_raw_row:
            overlay_ax.set_title(_panel_title(frame, annotations), fontsize=9)
        _strip_axis(overlay_ax, keep_frame_for_label=column == 0)

    if show_raw_row:
        axes[0][0].set_ylabel("signal")
        axes[1][0].set_ylabel(
            "+ outlines" if mask_style == MASK_STYLE_OUTLINE else "+ mask"
        )
    fig.suptitle(title, fontsize=12)

    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    save_kwargs = {"dpi": dpi}
    if is_jpeg:
        save_kwargs["pil_kwargs"] = {"quality": quality if quality is not None else 85}
    fig.savefig(out, **save_kwargs)
    plt.close(fig)
    logger.info("wrote %s", out)
    return out


def _strip_axis(ax, keep_frame_for_label: bool = False) -> None:
    """Hide ticks and frame. ``set_axis_off()`` would also hide the y-label artist,
    which silently dropped the row captions from every strip."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    if not keep_frame_for_label:
        ax.set_axis_off()


def _draw_base(ax, image: np.ndarray, cmap: str) -> None:
    """Percentile-rescaled base image. ``render_layers`` hardcodes gray, so mCherry
    (which wants ``inferno``) cannot go through it."""
    ax.imshow(normalize_for_display(image), cmap=cmap)


def _panel_title(
    frame: Frame, annotations: Optional[Mapping[int, Mapping[str, Any]]]
) -> str:
    """``"t=51 (8.6 h)"`` plus count/coverage when a population table is supplied."""
    hours = (frame.timepoint - 1) * TI_TO_HOURS
    label = f"t={frame.timepoint} ({hours:.1f} h)"
    if frame.requested != frame.timepoint:
        label += f"\n[asked t={frame.requested}]"
    row = (annotations or {}).get(frame.timepoint)
    if row:
        label += (
            f"\nn={int(row['n_objects'])}  cov={float(row['coverage_fraction']):.3f}"
        )
    return label
