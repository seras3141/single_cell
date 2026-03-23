"""
Blur analysis preprocessing for single cell datasets.

This module provides functionality to measure and store blur heatmaps for all images
in a dataset, which can be used for quality assessment and filtering during analysis.
"""

import os
import logging
from pathlib import Path
import traceback
from typing import Union, List, Optional, Dict, Tuple
from glob import glob
from tqdm import tqdm
import numpy as np


from src.utils.blur_measure import get_or_compute_blur_heatmap
from src.utils.file_utils import BlurFileHandler
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def generate_blur_heatmap_batch(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    pattern: str = "**/*.tif",
    patch_size: int = 32,
    stride_size: int = 16,
    normalize: bool = True,
    center_values: bool = True,
    file_handler: Optional[BlurFileHandler] = None,
    overwrite: bool = False
    ) -> Dict[str, str]:
    """
    Generate blur heatmaps for all images in a dataset.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save blur heatmaps
        pattern: Glob pattern to find images (default: "**/*.tif")
        patch_size: Size of patches for blur detection
        stride_size: Stride size between patches
        normalize: Whether to normalize the blur values
        center_values: Whether to center blur values on patches
        file_handler: Optional file handler for standardized naming
        overwrite: Whether to overwrite existing heatmap files
        
    Returns:
        Dictionary mapping input image paths to output heatmap paths
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    image_files = glob(os.path.join(input_dir, pattern), recursive=True)
    # image_files = list(input_dir.glob(pattern))
    if not image_files:
        logger.warning(f"No images found in {input_dir} with pattern {pattern}")
        return {}
    
    logger.info(f"Processing {len(image_files)} images for blur analysis...")
        
    # Process each image
    results = {}
    failed_files = []
    
    for img_path in tqdm(image_files, desc="Generating blur heatmaps"):
        img_path = Path(img_path)
        try:
            # Generate output filename
            if file_handler:
                # Use file handler for standardized naming
                output_name = file_handler.rename_image(str(img_path), suffix="_blur_heatmap")
            else:
                # Use original filename with suffix
                output_name = img_path.stem + "_blur_heatmap.tif"
            
            output_path = output_dir / output_name
            
            # Skip if file exists and overwrite is False
            if output_path.exists() and not overwrite:
                logger.debug(f"Skipping {img_path.name} - heatmap already exists")
                results[str(img_path)] = str(output_path)
                continue
            
            # Generate blur heatmap
            blur_heatmap = get_or_compute_blur_heatmap(
                img_path,
                blur_path=output_path,
                patch_size=patch_size,
                stride_size=stride_size,
                normalize=normalize,
                center_values=center_values,
            )
            
            results[str(img_path)] = str(output_path)
            logger.debug(f"Generated blur heatmap for {img_path.name}")
            
        except Exception as e:
            logger.error(f"Failed to process {img_path}: {e}", exc_info=True)
            print(traceback.format_exc())
            failed_files.append(str(img_path))
    
    # Log summary
    logger.info(f"Successfully processed {len(results)} images")
    if failed_files:
        logger.warning(f"Failed to process {len(failed_files)} images: {failed_files}")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate blur heatmaps for image datasets")
    parser.add_argument("--input", required=True, help="Input directory containing images")
    parser.add_argument("--output", required=True, help="Output directory for blur heatmaps")
    parser.add_argument("--pattern", default="**/*_BF.tif", help="Glob pattern for finding images")
    parser.add_argument("--patch-size", type=int, default=32, help="Patch size for blur detection")
    parser.add_argument("--stride-size", type=int, default=8, help="Stride size for patches")
    parser.add_argument("--no-normalize", action="store_true", help="Disable normalization")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    
    args = parser.parse_args()
    
    # Configure logging
    setup_logging()
    
    results = generate_blur_heatmap_batch(
        input_dir=args.input,
        output_dir=args.output,
        pattern=args.pattern,
        patch_size=args.patch_size,
        stride_size=args.stride_size,
        normalize=not args.no_normalize,
        overwrite=args.overwrite,
        file_handler=BlurFileHandler()
    )
    print(f"Processed {len(results)} images")
