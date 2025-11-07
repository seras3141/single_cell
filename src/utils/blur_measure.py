"""
Utility functions for measuring and detecting blur in microscopy images.

This module provides functions to analyze image sharpness using Laplacian variance,
which helps identify blurry images that should be excluded from training or analysis.
"""

import numpy as np
from scipy.ndimage import laplace
import tifffile as tiff
from pathlib import Path
from tqdm import tqdm
from glob import glob
import logging
from typing import Union, Tuple, List, Dict, Optional, Any
from skimage.transform import resize
from joblib import Parallel, delayed

from src.utils.logging_utils import setup_logging
from src.utils.image_utils import load_image

logger = logging.getLogger(__name__)

def measure_patchwise_blur(
    img: np.ndarray, 
    patch_size: Union[Tuple[int, int], int] = (50, 50), 
    stride_size: Union[Tuple[int, int], int] = (50, 50),
    center_values: bool = True
) -> np.ndarray:
    """
    Measure patchwise blur using Laplacian variance with option to center values.

    Args:
        img: 2D image array
        patch_size: Size of the patches (height, width)
        stride_size: Stride size for sliding the patch
        center_values: If True, the input image will be padded so that blur values
                       are centered on the patches, and the output will match input size

    Returns:
        2D array representing the blur map with variance values for each patch.
        If center_values=True, output size matches input size.
        If center_values=False, output size is reduced based on patch/stride sizes.
    """
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)
    if isinstance(stride_size, int):
        stride_size = (stride_size, stride_size)
    assert len(patch_size) == 2 and len(stride_size) == 2, "patch_size and stride_size must be tuples of (height, width)"
    assert img.ndim == 2, "Input image must be a 2D array"

    patch_height, patch_width = patch_size
    stride_y, stride_x = stride_size
    height, width = img.shape

    if center_values:
        # Calculate padding needed to ensure we can compute patches at stride intervals
        # while covering the entire original image
        pad_y = patch_height // 2
        pad_x = patch_width // 2
        
        # Pad the input image using reflection to avoid edge artifacts
        padded_img = np.pad(img, ((pad_y, pad_y), (pad_x, pad_x)), mode='reflect')
        
        # Calculate output dimensions based on stride
        out_height = (height + stride_y - 1) // stride_y
        out_width = (width + stride_x - 1) // stride_x
        
        blur_map = np.zeros((out_height, out_width))
        
        # Process patches on the padded image at stride intervals
        for i in range(out_height):
            for j in range(out_width):
                # Calculate position in original image coordinates
                orig_y = i * stride_y
                orig_x = j * stride_x
                
                # Skip if we're beyond the original image boundaries
                if orig_y >= height or orig_x >= width:
                    continue
                
                # Extract patch centered on the strided position
                start_y = orig_y  # Position in padded image (padding offset already included)
                start_x = orig_x
                patch = padded_img[start_y:start_y + patch_height, start_x:start_x + patch_width]
                
                # Calculate Laplacian variance
                laplacian = laplace(patch)
                variance = np.var(laplacian)
                blur_map[i, j] = variance
    else:
        # Standard implementation without padding
        # Calculate the output dimensions
        out_height = (height - patch_height) // stride_y + 1
        out_width = (width - patch_width) // stride_x + 1

        blur_map = np.zeros((out_height, out_width))

        # Perform convolution-like operation with patches
        for i in range(out_height):
            for j in range(out_width):
                start_y = i * stride_y
                start_x = j * stride_x
                patch = img[start_y:start_y + patch_height, start_x:start_x + patch_width]
                laplacian = laplace(patch)
                variance = np.var(laplacian)
                blur_map[i, j] = variance

    return blur_map

def measure_image_blur(img_path: str, method: str = 'laplacian') -> float:
    """
    Calculate the blur level of an entire image.
    
    Args:
        img_path: Path to the image file
        method: Method to use for blur detection ('laplacian' or 'sobel')
        
    Returns:
        Blur score (lower values indicate more blur)
    """
    try:
        # Read image
        img = load_image(img_path)
            
        # If 3D stack, use the first image
        if img.ndim > 2:
            img = img[0] if img.shape[0] < img.shape[1] else img
            
        # Apply Laplacian operator
        laplacian = laplace(img.astype(float))
        
        # Calculate variance
        return float(np.var(laplacian))
    except Exception as e:
        logger.error(f"Error analyzing {img_path}: {e}")
        return 0.0
                
def analyze_dataset_blur(
    data_dir: Union[str, Path], 
    pattern: str = '*.tif',
    threshold: float = 100.0
) -> Dict[str, float]:
    """
    Analyze a dataset for blur levels and identify blurry images.
    
    Args:
        data_dir: Directory containing images
        pattern: Glob pattern for finding images
        threshold: Threshold for classifying images as blurry (lower values are more blurry)
        
    Returns:
        Dictionary mapping image paths to their blur scores
    """
    data_dir = Path(data_dir)
    images = glob(str(data_dir / pattern))
    
    if not images:
        logger.warning(f"No images found in {data_dir} with pattern {pattern}")
        return {}
        
    logger.info(f"Analyzing {len(images)} images for blur...")
    
    results = {}
    for img_path in tqdm(images, desc="Analyzing images"):
        blur_score = measure_image_blur(img_path)
        results[img_path] = blur_score
        
    # Classify and report results
    blurry_count = sum(1 for score in results.values() if score < threshold)
    logger.info(f"Found {blurry_count} blurry images out of {len(images)} total images")
    
    return results
    
def filter_blurry_images(
    image_paths: List[str],
    threshold: float = 100.0
) -> List[str]:
    """
    Filter a list of image paths to remove blurry images.
    
    Args:
        image_paths: List of paths to images
        threshold: Threshold for classifying images as blurry
        
    Returns:
        List of paths to non-blurry images
    """
    results = []
    for img_path in tqdm(image_paths, desc="Filtering blurry images"):
        blur_score = measure_image_blur(img_path)
        if blur_score >= threshold:
            results.append(img_path)
            
    logger.info(f"Removed {len(image_paths) - len(results)} blurry images out of {len(image_paths)} total images")
    return results

def resize_image(img: np.ndarray, target_img: np.ndarray) -> np.ndarray:
    """
    Resize an image to the target shape using skimage's resize function.
    
    Args:
        img: Input image as a numpy array
        target_shape: Desired output shape (height, width)
        
    Returns:
        Resized image as a numpy array
    """

    if img.ndim == 3:
        # 3D case: resize each slice
        target_shape = target_img.shape[1:]  # (height, width)
        resized_slices = []
        for slice_ in img:
            if slice_.shape != target_shape:
                resized_slice = resize(slice_, target_shape, order=1, mode='reflect', 
                                     preserve_range=True, anti_aliasing=False)
            else:
                resized_slice = slice_
            resized_slices.append(resized_slice)
        img = np.stack(resized_slices)
    else:
        # 2D case: resize if necessary
        target_shape = target_img.shape  # (height, width)
        if img.shape != target_shape:
            img = resize(img, target_shape, order=1, mode='reflect',
                                preserve_range=True, anti_aliasing=False)
            
    return img

def measure_blur_heatmap(
    input_data: np.ndarray, 
    patch_size: int = 32,
    stride_size: int = 8,
    normalize: bool = True,
    center_values: bool = True,
    **kwargs: Any
) -> np.ndarray:
    """
    Generate a blur heatmap for an image or stack of images.
    
    Args:
        input_data: Image numpy array
        patch_size: Size of patches for blur detection
        stride_size: Stride size between patches
        normalize: Whether to normalize the output to [0, 1] range
        center_values: Whether to center blur values on patches (pad to original size)
        
    Returns:
        Blur heatmap as a numpy array
    """
    img = input_data
        
    # Handle 3D image stacks
    if img.ndim > 2:
        # Process each slice and stack results in parallel using joblib.Parallel
        n_jobs = kwargs.get('n_jobs', -1)  # Use all available cores by default

        def process_slice(z):
            return measure_patchwise_blur(
            img[z], 
            patch_size=patch_size, 
            stride_size=stride_size,
            center_values=center_values
            )

        slices = Parallel(n_jobs=n_jobs)(
            delayed(process_slice)(z) for z in range(img.shape[0])
        )
        # Stack results to create a 3D blur heatmap
        blur_heatmap = np.stack(slices)
    else:
        assert img.ndim == 2 and not normalize, "Only 3D images (z-stacks) can be processed with normalization"
        # Process single 2D image
        blur_heatmap = measure_patchwise_blur(
            img,
            patch_size=patch_size,
            stride_size=stride_size,
            center_values=center_values
        )
    
    # Normalize if requested and for multi-dimensional data
    if normalize and blur_heatmap.size > 0:
        if blur_heatmap.ndim == 3:
            # Normalize each pixel location through the z-stack (axis=0)
            min_val = blur_heatmap.min(axis=0, keepdims=True)
            max_val = blur_heatmap.max(axis=0, keepdims=True)
            # Avoid division by zero
            denom = max_val - min_val
            denom[denom == 0] = 1
            blur_heatmap = (blur_heatmap - min_val) / denom
        else:
            # 2D case: TODO , this must be removed in the future
            # Normalize 2D data globally
            min_val = blur_heatmap.min()
            max_val = blur_heatmap.max()
            if max_val > min_val:
                blur_heatmap = (blur_heatmap - min_val) / (max_val - min_val)

    # Resize to match original input size if needed
    blur_heatmap = resize_image(blur_heatmap, input_data)

    return blur_heatmap

def load_blur_heatmap(blur_path: Union[str, Path]) -> np.ndarray:
    """Load a blur heatmap from disk."""
    blur_path = Path(blur_path)
    if not blur_path.exists():
        raise FileNotFoundError(f"Blur heatmap file not found: {blur_path}")
    try:
        blur_map = tiff.imread(blur_path).astype(np.float32)
    except Exception as e:
        raise Warning(f"Failed to load cached blur map {blur_path}: {e}")

    return blur_map
        
def get_or_compute_blur_heatmap(
    image_path: Union[str, Path],
    blur_path: Optional[Union[str, Path]] = None,
    patch_size: int = 32,
    stride_size: int = 8,
    normalize: bool = True,
    center_values: bool = True,
) -> np.ndarray:
    """    Get or compute a blur heatmap for an image.
    Args:
        image_path: Path to the input image file
        blur_path: Optional path to save or load the blur heatmap
        patch_size: Size of patches for blur detection
        stride_size: Stride size between patches
        normalize: Whether to normalize the output to [0, 1] range
    Returns:
        2D numpy array representing the blur heatmap.
    """
        
    # Check disk cache
    if blur_path is not None and Path(blur_path).exists():
        blur_map = load_blur_heatmap(blur_path)
        logger.debug(f"Loaded cached blur heatmap from {blur_path}")

    else:
        image_path = Path(image_path)
        image = load_image(image_path)

        blur_map = measure_blur_heatmap(
            image,
            patch_size=patch_size,
            stride_size=stride_size,
            normalize=normalize,
            center_values=center_values,
        )

        if blur_path is not None:
            try:
                blur_cache_dir = Path(blur_path).parent
                blur_cache_dir.mkdir(parents=True, exist_ok=True)
                tiff.imwrite(str(blur_path), blur_map.astype(np.float32))

            except Exception as e:
                raise Warning(f"Failed to save blur map to cache: {e}")

        logger.debug(f"Blur heatmap for {image_path.name} computed with shape {blur_map.shape}")

    return blur_map


# Example usage if module is run directly
if __name__ == "__main__":
    import argparse
    
    # Configure argument parser
    parser = argparse.ArgumentParser(description='Analyze images for blur detection')
    parser.add_argument('--input-dir', required=True, help='Directory containing images to analyze')
    parser.add_argument('--pattern', default='*.tif', help='Glob pattern for finding images')
    parser.add_argument('--threshold', type=float, default=100.0, 
                        help='Threshold for blur detection (lower values are more blurry)')
    parser.add_argument('--output', help='Output file to save analysis results (optional)')
    
    args = parser.parse_args()
    
    # Configure logging
    setup_logging()
    
    # Run analysis
    results = analyze_dataset_blur(args.input_dir, args.pattern, args.threshold)
    
    # Sort by blur score (ascending - most blurry first)
    sorted_results = sorted(results.items(), key=lambda x: x[1])
    
    # Print a summary
    print("\nTop 5 most blurry images:")
    for path, score in sorted_results[:5]:
        print(f"{Path(path).name}: {score:.2f}")
        
    print("\nTop 5 sharpest images:")
    for path, score in sorted_results[-5:]:
        print(f"{Path(path).name}: {score:.2f}")
    
    # Save results if requested
    if args.output:
        import csv
        with open(args.output, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Image Path', 'Blur Score', 'Is Blurry'])
            for path, score in sorted_results:
                writer.writerow([path, score, 'Yes' if score < args.threshold else 'No'])
        print(f"\nResults saved to {args.output}")
