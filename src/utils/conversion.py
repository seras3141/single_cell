"""
Utilities for converting between 2D and 3D image formats.

This module provides functions to convert between 2D slice-based image datasets
and 3D volumetric image datasets, supporting both raw images and segmentation masks.
"""

import os
import re
import numpy as np
import tifffile as tiff
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm
from glob import glob

def combine_2d_to_3d(input_dir: str, output_dir: str, pattern: str = r"(.+?)_z(\d+)(?:_(BF|Cells))?\.(tif|tiff)"):
    """
    Combine 2D TIFF images into 3D volumetric TIFF files.
    
    Args:
        input_dir: Directory containing 2D TIFF images
        output_dir: Directory to save 3D TIFF volumes
        pattern: Regular expression pattern to extract base name, z-index, and suffix
        
    Example:
        Converts files like "sample_z1_BF.tif", "sample_z2_BF.tif", ...
        to a single 3D file "sample_BF_3d.tif"
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Group files by base name and suffix (_BF or _Cells)
    file_groups = defaultdict(list)
    file_names = []
    for ext in ("*.tif", "*.tiff"):
        file_names.extend([os.path.relpath(f, input_dir) for f in glob(os.path.join(input_dir, "**", ext), recursive=True)])
    if not file_names:
        print(f"No TIFF files found in {input_dir}. Please check the directory and file pattern.")
        return
    
    for file_name in tqdm(file_names, desc="Finding 2D files"):
        if file_name.endswith(".tif") or file_name.endswith(".tiff"):
            fname_only = os.path.basename(file_name)
            match = re.match(pattern, fname_only)
            if match:
                base_name = match.group(1)
                z_index = int(match.group(2))
                suffix = match.group(3)
                if suffix is None:
                    key = f"{base_name}"
                else:
                    key = f"{base_name}_{suffix}"
                file_groups[key].append((z_index, file_name))

    print(f"Found {len(file_groups)} groups of 2D images to combine into 3D volumes.")
    print("Example groups:")
    for key in list(file_groups.keys())[:5]:  # Show first 5 groups
        print(f"  {key}: {len(file_groups[key])} files")    

    # Combine 2D TIFFs into 3D TIFFs
    for key, files in tqdm(file_groups.items(), desc="Combining to 3D"):
        # Sort files by z-index
        files.sort(key=lambda x: x[0])

        images = []
        for _, file_name in files:
            file_path = os.path.join(input_dir, file_name)
            img = tiff.imread(file_path)
            images.append(img)

        # Skip if no images found
        if not images:
            continue

        # Ensure consistent dtype with input images
        output_dtype = images[0].dtype

        # Save as 3D TIFF
        output_path = os.path.join(output_dir, f"{key}_3d.tif")
        tiff.imwrite(output_path, np.stack(images, axis=0).astype(output_dtype), 
                     photometric='minisblack')
        
    print(f"Successfully combined {len(file_groups)} 2D image sets into 3D volumes in {output_dir}")


def split_3d_to_2d(input_path: str, output_dir: str, suffix: Optional[str] = None):
    """
    Split a 3D TIFF file into separate 2D TIFF slices.
    
    Args:
        input_path: Path to 3D TIFF file
        output_dir: Directory to save 2D TIFF slices
        suffix: Optional suffix to add to output files (e.g., "BF", "Cells")
        
    Example:
        Converts "sample_3d.tif" to "sample_z1.tif", "sample_z2.tif", ...
        or with suffix to "sample_z1_BF.tif", "sample_z2_BF.tif", ...
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Load 3D TIFF
    volume = tiff.imread(input_path)
    
    # Get base filename
    base_name = Path(input_path).stem
    if base_name.endswith('_3d'):
        base_name = base_name[:-3]
    
    # Save each z-slice as a separate 2D TIFF
    for z in tqdm(range(volume.shape[0]), desc=f"Splitting {base_name}"):
        slice_name = f"{base_name}_z{z+1}"
        if suffix:
            slice_name += f"_{suffix}"
        slice_name += ".tif"
        
        output_path = os.path.join(output_dir, slice_name)
        tiff.imwrite(output_path, volume[z], photometric='minisblack')
    
    print(f"Successfully split {input_path} into {volume.shape[0]} 2D slices in {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert between 2D and 3D TIFF images")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Combine 2D to 3D command
    combine_parser = subparsers.add_parser("combine", help="Combine 2D TIFFs into 3D volumes")
    combine_parser.add_argument("--input", required=True, help="Input directory containing 2D TIFF images")
    combine_parser.add_argument("--output", required=True, help="Output directory for 3D TIFF volumes")
    combine_parser.add_argument("--pattern", default=r"(.+?)_z(\d+)(?:_(BF|Cells))?\.(tif|tiff)", 
                             help="Regular expression pattern to extract base name, z-index, and suffix")
    
    # Split 3D to 2D command
    split_parser = subparsers.add_parser("split", help="Split 3D TIFF volumes into 2D slices")
    split_parser.add_argument("--input", required=True, help="Input 3D TIFF file")
    split_parser.add_argument("--output", required=True, help="Output directory for 2D TIFF slices")
    split_parser.add_argument("--suffix", help="Optional suffix to add to output files (e.g., BF, Cells)")
    
    args = parser.parse_args()
    
    if args.command == "combine":
        combine_2d_to_3d(args.input, args.output, args.pattern)
    elif args.command == "split":
        split_3d_to_2d(args.input, args.output, args.suffix)
    else:
        parser.print_help()
