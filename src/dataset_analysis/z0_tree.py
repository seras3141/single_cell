"""Stage the z0 projection plane into a standalone tree (symlinks by default).

``z0`` is the 2D **projection**, not an optical slice: the main pipeline excludes it via
``z_min: 1`` (``config/config.yaml:30``) and this package already encodes it as
:data:`src.dataset_analysis.qc.DEFAULT_PROJECTION_Z_INDEX`. Because the projection is
segmented but never consumed downstream, its masks sit *upstream* of 3D combine, blur
filtering and tracking — the stages suspected of destroying detections.

This module mirrors the ``split_data/`` and ``inference/<model>/masks/`` folders of a
processed experiment into a parallel tree holding only the z0 files, preserving the
folder structure. Symlinks are the default: the payload is unchanged, there is one
source of truth, and the tree is rebuildable at no cost. Links point at the *processed*
tree (absolute but unresolved) — the processed ``split_data`` entries are themselves
symlinks into the raw group share's ``_Projection`` folder, so resolving them would
bypass the processed tree and silently retarget the raw share.

Filtering, tracking and 3D combining are deliberately absent — a single plane has no
stack to combine and no z-slices to select between, and per-timepoint population
statistics need no temporal identity. Note that skipping blur filtering also removes a
frame-rejection QC gate, so this tree includes frames the per-slice pipeline would have
discarded.

See ``docs/dataset_analysis/plan_z0_projection_count_check.md``.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from pathlib import Path
from typing import Iterator, List, NamedTuple, Optional, Sequence, Tuple, Union

import pandas as pd

from src.dataset_analysis.qc import DEFAULT_PROJECTION_Z_INDEX

logger = logging.getLogger(__name__)

#: Stage folders copied into the z0 tree, relative to an experiment directory.
#: ``inference`` is templated on the model name; ``split_data`` is flat.
#: Accepted materialization modes.
VALID_MODES = ("symlink", "copy")

SPLIT_DATA_DIR = "split_data"
INFERENCE_MASKS_TEMPLATE = "inference/{model}/masks"

#: Processed-tree filenames look like ``pMF5V1_C09_t101_z0_BF.tif`` or
#: ``pMF5V1_C09_t101_z0_pred_mask.zarr``. The raw-tree parser
#: (``inventory.parse_image_metadata``) cannot be reused: it expects the
#: wavelength-based raw convention (``w1``/``w2``/``w3``), not the channel-named
#: processed one.
_NAME_RE = re.compile(
    r"_(?P<well>[A-Za-z]\d{2,3})_t(?P<timepoint>\d+)_z(?P<z_index>\d+)_(?P<suffix>.+)$"
)

_SUMMARY_COLUMNS = (
    "experiment",
    "stage",
    "n_source",
    "n_z0",
    "n_created",
    "n_repaired",
    "n_existing",
)


class ProcessedName(NamedTuple):
    """Parsed components of a processed-tree filename."""

    well: str
    timepoint: int
    z_index: int
    suffix: str


def parse_processed_name(name: str) -> Optional[ProcessedName]:
    """Parse ``pMF5V1_C09_t101_z0_BF.tif`` into its parts, or ``None`` if unparseable.

    ``.zarr`` masks are directories; :attr:`Path.stem` strips the extension either
    way, so ``pMF5V1_C09_t101_z0_pred_mask.zarr`` yields ``suffix="pred_mask"``.
    """
    match = _NAME_RE.search(Path(name).stem)
    if not match:
        return None
    return ProcessedName(
        well=match.group("well").upper(),
        timepoint=int(match.group("timepoint")),
        z_index=int(match.group("z_index")),
        suffix=match.group("suffix"),
    )


def iter_z0_entries(
    directory: Union[str, Path],
    z_index: int = DEFAULT_PROJECTION_Z_INDEX,
) -> Iterator[Tuple[Path, ProcessedName]]:
    """Yield ``(path, parsed)`` for every direct child of ``directory`` at ``z_index``.

    Unparseable names are skipped with a debug log rather than raising: stage folders
    legitimately hold sidecars (``dataset_split.json``) alongside the image files.
    """
    root = Path(directory)
    if not root.is_dir():
        return
    for path in sorted(root.iterdir()):
        parsed = parse_processed_name(path.name)
        if parsed is None:
            logger.debug("skipping unparseable name: %s", path.name)
            continue
        if parsed.z_index == z_index:
            yield path, parsed


def _materialize(src: Path, dest: Path, mode: str) -> None:
    """Create ``dest`` from ``src`` as a symlink or a copy (files and .zarr dirs).

    Symlinks target ``src.absolute()``, deliberately *not* ``src.resolve()``: the
    processed ``split_data`` entries are themselves symlinks into the raw share, so
    resolving would point the z0 tree at raw data and detach it from the processed tree.
    """
    if mode == "symlink":
        os.symlink(src.absolute(), dest)
    else:
        if src.is_dir():
            shutil.copytree(src, dest)
        else:
            shutil.copy2(src, dest)


def _remove(dest: Path) -> None:
    """Delete ``dest`` whether it is a symlink, a file, or a directory tree."""
    if dest.is_symlink() or dest.is_file():
        dest.unlink()
    elif dest.is_dir():
        shutil.rmtree(dest)


def _dest_state(dest: Path, src: Path, mode: str) -> str:
    """Classify an existing destination as ``absent``, ``current``, or ``stale``.

    ``stale`` covers a link to the wrong target, a dangling link, and a kind mismatch
    (a symlink where a copy was asked for, or vice versa) — all of which an interrupted
    or reconfigured run leaves behind and which must be rebuilt rather than counted as
    staged. Copy-mode payloads are *not* content-verified; pass ``force`` to
    re-materialize them.
    """
    if not dest.exists() and not dest.is_symlink():
        return "absent"
    if dest.is_symlink():
        if mode != "symlink":
            return "stale"
        if Path(os.readlink(dest)) != src.absolute():
            return "stale"
        return "current" if dest.exists() else "stale"
    if mode == "symlink":
        return "stale"
    # Copy mode: a .zarr store replaced by a regular file (or the reverse) is a
    # kind mismatch, which the contract above says must be rebuilt.
    return "current" if dest.is_dir() == src.is_dir() else "stale"


def _stage_one_dir(
    source_dir: Path,
    dest_dir: Path,
    mode: str,
    z_index: int,
    dry_run: bool,
    force: bool,
) -> Tuple[int, int, int, int]:
    """Stage one stage-folder.

    Returns ``(n_source, n_created, n_repaired, n_existing)``. Stale entries are rebuilt
    rather than warned about, so re-running repairs a tree left half-written by an
    interrupted job instead of reporting it as complete.
    """
    n_source = sum(1 for _ in source_dir.iterdir()) if source_dir.is_dir() else 0
    entries = list(iter_z0_entries(source_dir, z_index=z_index))

    if not dry_run and entries:
        dest_dir.mkdir(parents=True, exist_ok=True)

    n_created = 0
    n_repaired = 0
    n_existing = 0
    for src, _ in entries:
        dest = dest_dir / src.name
        state = _dest_state(dest, src, mode)

        if state == "current" and not force:
            n_existing += 1
            continue

        if state == "absent":
            n_created += 1
        else:
            logger.info("rebuilding %s entry: %s", state, dest)
            n_repaired += 1
            if not dry_run:
                _remove(dest)

        if not dry_run:
            _materialize(src, dest, mode)

    return n_source, n_created, n_repaired, n_existing


def build_z0_tree(
    source_root: Union[str, Path],
    dest_root: Union[str, Path],
    experiments: Optional[Sequence[str]] = None,
    model: str = "cellpose_sam",
    mode: str = "symlink",
    z_index: int = DEFAULT_PROJECTION_Z_INDEX,
    dry_run: bool = False,
    force: bool = False,
) -> pd.DataFrame:
    """Mirror the z0 files of every experiment under ``source_root`` into ``dest_root``.

    Args:
        source_root: The processed tree, e.g.
            ``data/MF5V1_processed Timelapse samples 19.03.2024``.
        dest_root: Destination tree; created if absent.
        experiments: Experiment directory names to stage. ``None`` stages every
            directory found under ``source_root``.
        model: Segmentation model folder under ``inference/``.
        mode: ``"symlink"`` (default) or ``"copy"``.
        z_index: Plane to stage; defaults to the projection index (0).
        dry_run: Report what would be staged without touching the filesystem.
        force: Re-materialize entries that already look current. Needed for copy mode,
            whose payloads are not content-verified.

    Returns:
        One row per (experiment, stage) with source/z0/created/repaired/existing counts.

    Raises:
        FileNotFoundError: If ``source_root`` or a named experiment does not exist.
        ValueError: If ``mode`` is not one of :data:`VALID_MODES`.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")

    source = Path(source_root)
    if not source.is_dir():
        raise FileNotFoundError(f"source root not found: {source}")
    dest = Path(dest_root)

    # Staging onto the source would classify every source file as stale, delete it,
    # and replace it with a self-referential symlink — destroying the processed tree
    # on a single CLI typo.
    if source.resolve() == dest.resolve():
        raise ValueError(
            f"source_root and dest_root resolve to the same directory ({source}); "
            f"staging in place would destroy the source"
        )

    if experiments is None:
        names = sorted(p.name for p in source.iterdir() if p.is_dir())
    else:
        names = list(experiments)
        # Experiment dir names carry spaces and dates ("Ew2-1 MF5V1 0-72h 06-03-26"),
        # so a typo is easy and would otherwise be a silent no-op reading as success.
        absent = [n for n in names if not (source / n).is_dir()]
        if absent:
            raise FileNotFoundError(f"experiment(s) not found under {source}: {absent}")

    records: List[dict] = []
    for name in names:
        exp_src = source / name
        exp_dest = dest / name
        stages = {
            SPLIT_DATA_DIR: SPLIT_DATA_DIR,
            "inference_masks": INFERENCE_MASKS_TEMPLATE.format(model=model),
        }
        for stage_label, rel in stages.items():
            src_dir = exp_src / rel
            if not src_dir.is_dir():
                logger.warning("missing stage dir, skipped: %s", src_dir)
                continue
            n_source, n_created, n_repaired, n_existing = _stage_one_dir(
                src_dir, exp_dest / rel, mode, z_index, dry_run, force
            )
            records.append(
                {
                    "experiment": name,
                    "stage": stage_label,
                    "n_source": n_source,
                    "n_z0": n_created + n_repaired + n_existing,
                    "n_created": n_created,
                    "n_repaired": n_repaired,
                    "n_existing": n_existing,
                }
            )
            logger.info(
                "%s/%s: %d z0 of %d source (%d new, %d rebuilt, %d present)",
                name,
                stage_label,
                n_created + n_repaired + n_existing,
                n_source,
                n_created,
                n_repaired,
                n_existing,
            )

    return pd.DataFrame(records, columns=list(_SUMMARY_COLUMNS))
