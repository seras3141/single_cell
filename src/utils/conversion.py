"""
Utilities for converting between 2D and 3D image formats.

This module provides functions to convert between 2D slice-based image datasets
and 3D volumetric image datasets, supporting both raw images and segmentation masks.
"""

import re
import numpy as np
import tifffile as tiff
from pathlib import Path
from typing import Optional, Union
from collections import defaultdict
from tqdm import tqdm

from src.utils.image_utils import LABEL_FORMATS, save_labels, load_labels


def _validate_label_format(format_name: str, arg_name: str) -> None:
    if format_name not in LABEL_FORMATS:
        raise ValueError(f"{arg_name} must be one of {list(LABEL_FORMATS)}; got {format_name!r}")


def _strip_known_extension(filename: str, extension: str) -> str:
    if filename.lower().endswith(extension.lower()):
        return filename[:-len(extension)]
    return filename


def _match_slice_pattern(pattern: str, stem: str, filename: str) -> Optional[re.Match[str]]:
    match = re.fullmatch(pattern, stem)
    if match:
        return match
    return re.fullmatch(pattern, filename)


def combine_2d_to_3d(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    pattern: str = r"(.+?)_z(\d+)(?:_(BF|Cells))?",
    recursive: bool = False,
    z_min: Optional[int] = 1,
    z_max: Optional[int] = None,
    output_format: str = "tif",
    input_format: Optional[str] = None,
):
    """
    Combine saved 2D label slices into 3D volumetric files.

    Args:
        input_dir: Directory containing 2D label slices
        output_dir: Directory to save 3D volumes
        pattern: Regular expression pattern to extract base name, z-index, and suffix
        recursive: Whether to search subdirectories recursively
        z_min: Minimum z-index to include (inclusive). Defaults to 1 to skip z0,
            which is typically a 2D projection rather than an optical section.
        z_max: Maximum z-index to include (inclusive). Defaults to None (no upper limit).
        output_format: Format for the combined 3D volume. One of ``"tif"`` (default),
            ``"zarr"``, or ``"hdf5"``.
        input_format: Format of the input 2D label slices. Defaults to
            ``output_format``. Cross-format conversion is not supported.

    Example:
        Converts files like "sample_z1_BF.tif", "sample_z2_BF.tif", ...
        to a single 3D file "sample_BF_3d.tif" (or .zarr / .h5)
    """
    if input_format is None:
        input_format = output_format

    _validate_label_format(output_format, "output_format")
    _validate_label_format(input_format, "input_format")
    if input_format != output_format:
        raise ValueError(
            "combine_2d_to_3d only supports matching input_format and output_format; "
            f"got input_format={input_format!r}, output_format={output_format!r}"
        )

    input_ext = LABEL_FORMATS[input_format]
    output_ext = LABEL_FORMATS[output_format]

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    assert input_dir.exists(), f"Input directory {input_dir} does not exist."

    # Group files by base name and suffix (_BF or _Cells)
    file_groups = defaultdict(list)
    glob_pattern = f"*{input_ext}"
    file_names = sorted(input_dir.rglob(glob_pattern) if recursive else input_dir.glob(glob_pattern))
    if not file_names:
        print(f"No {input_format} files found in {input_dir}. Please check the directory and file pattern.")
        return

    for file_name in tqdm(file_names, desc="Finding 2D files"):
        fname_only = file_name.name
        stem = _strip_known_extension(fname_only, input_ext)
        match = _match_slice_pattern(pattern, stem, fname_only)
        if match:
            base_name = match.group(1)
            z_index = int(match.group(2))
            suffix = match.group(3) if match.lastindex and match.lastindex >= 3 else None
            if z_min is not None and z_index < z_min:
                continue
            if z_max is not None and z_index > z_max:
                continue
            if suffix is None:
                key = f"{base_name}"
            else:
                key = f"{base_name}_{suffix.strip('_')}"
            file_groups[key].append((z_index, file_name))

    print(f"Found {len(file_groups)} groups of 2D images to combine into 3D volumes.")
    print("Example groups:")
    for key in list(file_groups.keys())[:5]:  # Show first 5 groups
        print(f"  {key}: {len(file_groups[key])} files")

    # Combine 2D labels into 3D volumes
    for key, files in tqdm(file_groups.items(), desc="Combining to 3D"):
        # Sort files by z-index
        files.sort(key=lambda x: x[0])

        images = []
        for _, file_path in files:
            img = load_labels(file_path)
            images.append(img)

        # Skip if no images found
        if not images:
            continue

        volume = np.stack(images, axis=0).astype(images[0].dtype)
        output_path = output_dir / f"{key}_3d{output_ext}"
        save_labels(volume, output_path)

    print(f"Successfully combined {len(file_groups)} 2D image sets into 3D volumes in {output_dir}")


def split_3d_to_2d(
    input_path: Union[str, Path],
    output_dir: Union[str, Path],
    suffix: Optional[str] = None,
    output_format: str = "tif",
):
    """
    Split a 3D label volume into separate 2D slices.
    
    Args:
        input_path: Path to 3D label volume
        output_dir: Directory to save 2D label slices
        suffix: Optional suffix to remove from input_file and/or add to output files (e.g., "BF", "Cells")
        output_format: Format for the output 2D slices. One of ``"tif"`` (default),
            ``"zarr"``, or ``"hdf5"``.
        
    Example:
        Converts "sample_3d.tif" to "sample_z1.tif", "sample_z2.tif", ...
        or with suffix to "sample_z1_BF.tif", "sample_z2_BF.tif", ...
    """
    _validate_label_format(output_format, "output_format")
    output_ext = LABEL_FORMATS[output_format]

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if suffix:
        for ext in (".tiff", *LABEL_FORMATS.values()):
            if suffix.endswith(ext):
                suffix = suffix[:-len(ext)]
                break
        if suffix.endswith('_3d'):
            suffix = suffix[:-3]
        suffix = suffix.strip('_')

    volume = load_labels(input_path)
    
    # Get base filename
    base_name = input_path.stem
    if base_name.endswith('_3d'):
        base_name = base_name[:-3]
    if suffix and base_name.endswith(suffix):
        base_name = base_name[:-(len(suffix))]
    base_name = base_name.rstrip('_')
    
    # Save each z-slice as a separate 2D label file.
    for z in tqdm(range(volume.shape[0]), desc=f"Splitting {base_name}"):
        slice_name = f"{base_name}_z{z+1}"
        if suffix:
            slice_name += f"_{suffix}"
        slice_name += output_ext
        
        output_path = output_dir / slice_name
        if output_format == "tif":
            tiff.imwrite(output_path, volume[z], photometric='minisblack')
        else:
            save_labels(volume[z], output_path)
    
    print(f"Successfully split {input_path} into {volume.shape[0]} 2D slices in {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert between 2D slices and 3D label volumes."
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Combine 2D to 3D command
    combine_parser = subparsers.add_parser("combine", help="Combine 2D label slices into 3D volumes")
    combine_parser.add_argument("--input", required=True, help="Input directory containing 2D label slices")
    combine_parser.add_argument("--output", required=True, help="Output directory for 3D label volumes")
    combine_parser.add_argument(
        "--pattern",
        default=r"(.+?)_z(\d+)(?:_(BF|Cells))?",
        help=(
            "Regex used to extract base name, z-index, and optional suffix. "
            "By default it is matched against the filename stem; legacy patterns "
            "that include the extension are also supported."
        ),
    )
    combine_parser.add_argument("--recursive", action="store_true", 
                             help="Search subdirectories recursively for 2D label slices")
    combine_parser.add_argument("--z-min", type=int, default=1, help="Minimum z-index to include")
    combine_parser.add_argument("--z-max", type=int, default=None, help="Maximum z-index to include")
    combine_parser.add_argument(
        "--input-format",
        choices=sorted(LABEL_FORMATS),
        default=None,
        help="Input slice format. Defaults to --output-format.",
    )
    combine_parser.add_argument(
        "--output-format",
        choices=sorted(LABEL_FORMATS),
        default="tif",
        help="Output volume format. Cross-format combine is not supported.",
    )
    
    # Split 3D to 2D command
    split_parser = subparsers.add_parser("split", help="Split 3D label volumes into 2D slices")
    split_parser.add_argument("--input", required=True, help="Input 3D label volume")
    split_parser.add_argument("--output", required=True, help="Output directory for 2D label slices")
    split_parser.add_argument("--suffix", help="Optional suffix to add to output files (e.g., BF, Cells)")
    split_parser.add_argument(
        "--output-format",
        choices=sorted(LABEL_FORMATS),
        default="tif",
        help="Output slice format.",
    )
    
    args = parser.parse_args()
    
    if args.command == "combine":
        combine_2d_to_3d(
            args.input,
            args.output,
            pattern=args.pattern,
            recursive=args.recursive,
            z_min=args.z_min,
            z_max=args.z_max,
            output_format=args.output_format,
            input_format=args.input_format,
        )
    elif args.command == "split":
        split_3d_to_2d(
            args.input,
            args.output,
            suffix=args.suffix,
            output_format=args.output_format,
        )
    else:
        parser.print_help()
