"""Resolve BF -> mCherry / inference-mask sibling paths by filename convention."""

from pathlib import Path
from typing import Optional, Union

from src.inference.output_manager import OutputManager


def resolve_related_paths(
    bf_path: Union[str, Path],
    output_manager: OutputManager,
    bf_suffix: str = "_BF",
    mcherry_suffix: str = "_mCherry",
) -> dict:
    """Resolve the mCherry image and inference mask paths that correspond to a BF file.

    Returns {"bf": Path, "mcherry": Path | None, "mask": Path | None}; "mcherry"/"mask"
    are None when the corresponding file does not exist on disk (e.g. that channel wasn't
    acquired, or inference hasn't been run for this file yet) rather than raising.
    """
    bf_path = Path(bf_path)

    mcherry_path: Optional[Path] = None
    if bf_path.stem.endswith(bf_suffix):
        mcherry_stem = bf_path.stem[: -len(bf_suffix)] + mcherry_suffix
        candidate = bf_path.with_name(f"{mcherry_stem}{bf_path.suffix}")
        if candidate.exists():
            mcherry_path = candidate

    mask_candidate = output_manager.expected_mask_path(bf_path)
    mask_path = mask_candidate if mask_candidate.exists() else None

    return {"bf": bf_path, "mcherry": mcherry_path, "mask": mask_path}
