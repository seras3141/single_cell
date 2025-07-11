# """
# Cell tracking utilities for 3D cell segmentation data.

# This module provides functions to track cells across z-stacks in 3D segmentation data,
# applying consistent cell IDs and filtering based on blur/sharpness measures.
# """

# import os
# import numpy as np
# import trackpy as tp
# import pandas as pd
# from skimage.measure import regionprops
# from tqdm import tqdm
# import tifffile as tiff
# from typing import List, Tuple, Dict, Optional, Union, Callable
# from pathlib import Path

# from .blur_measure import measure_blur_heatmap


# def extract_cell_features(segmentation_mask: np.ndarray, 
#                         intensity_image: Optional[np.ndarray] = None) -> pd.DataFrame:
#     """
#     Extract features from segmentation mask, including centroids and optional intensity metrics.
    
#     Args:
#         segmentation_mask: 2D segmentation mask with labeled cells
#         intensity_image: Optional intensity image for extracting intensity-based features
    
#     Returns:
#         DataFrame with cell properties including position and label
#     """
#     # Use intensity image if provided
#     if intensity_image is not None:
#         regions = regionprops(segmentation_mask, intensity_image=intensity_image)
#     else:
#         regions = regionprops(segmentation_mask)
    
#     # Extract features from each cell region
#     features = {
#         'x': [region.centroid[0] for region in regions],
#         'y': [region.centroid[1] for region in regions],
#         'label': [region.label for region in regions],
#         'area': [region.area for region in regions]
#     }
    
#     # Add intensity features if intensity image provided
#     if intensity_image is not None:
#         features.update({
#             'mean_intensity': [region.mean_intensity for region in regions],
#             'max_intensity': [region.max_intensity for region in regions],
#             'min_intensity': [region.min_intensity for region in regions]
#         })
    
#     return pd.DataFrame.from_dict(features)


# def filter_cells_by_blur(props: pd.DataFrame, 
#                         blur_image: np.ndarray, 
#                         threshold: float = 0.5,
#                         invert: bool = False) -> pd.DataFrame:
#     """
#     Filter cells based on blur/sharpness values.
    
#     Args:
#         props: DataFrame with cell properties
#         blur_image: Image with blur/sharpness values
#         threshold: Blur threshold for filtering
#         invert: If True, keep cells with values > threshold, else keep cells with values < threshold
    
#     Returns:
#         Filtered DataFrame with cell properties
#     """
#     # Add blur intensity for each cell
#     blur_values = []
#     segmentation_mask = np.zeros_like(blur_image, dtype=int)
    
#     # Recreate a temporary mask for sampling blur values
#     for i, row in props.iterrows():
#         x, y = int(row['x']), int(row['y'])
#         if 0 <= x < blur_image.shape[0] and 0 <= y < blur_image.shape[1]:
#             blur_values.append(blur_image[x, y])
#         else:
#             blur_values.append(0)  # Default for out-of-bounds cells
    
#     props['blur_value'] = blur_values
    
#     # Filter cells based on blur threshold
#     if invert:
#         filtered_props = props[props['blur_value'] > threshold]
#     else:
#         filtered_props = props[props['blur_value'] < threshold]
    
#     return filtered_props


# def track_cells_3d(segmentation_stack: np.ndarray, 
#                   blur_stack: Optional[np.ndarray] = None,
#                   blur_threshold: float = 0.5,
#                   search_range: float = 5.0,
#                   memory: int = 1) -> np.ndarray:
#     """
#     Track cell identities across z-stacks in a 3D segmentation volume.
    
#     Args:
#         segmentation_stack: 3D array with labeled segmentation masks
#         blur_stack: Optional 3D array with blur/sharpness measures
#         blur_threshold: Threshold for blur filtering
#         search_range: Maximum distance between features to consider as same particle
#         memory: How many frames a particle can disappear and reappear
    
#     Returns:
#         3D array with consistent cell labels across z-stacks
#     """
#     all_cells = []
    
#     # Extract cell features from each z-stack
#     for z, mask in enumerate(tqdm(segmentation_stack, desc="Extracting features")):
#         # Get cell properties from the segmentation mask
#         props = extract_cell_features(mask)
        
#         # Filter by blur if blur_stack provided
#         if blur_stack is not None and z < len(blur_stack):
#             props = filter_cells_by_blur(props, blur_stack[z], blur_threshold)
        
#         # Add z-stack information
#         props['frame'] = z
        
#         # Add to collection
#         all_cells.append(props)
    
#     # Combine all cell features
#     if not all_cells:
#         return np.zeros_like(segmentation_stack)
        
#     data = pd.concat(all_cells, ignore_index=True)
    
#     # Perform tracking using trackpy
#     if len(data) > 0:
#         tracked = tp.link_df(data, search_range=search_range, memory=memory)
#     else:
#         # Return original stack if no cells to track
#         return segmentation_stack
    
#     # Create a new 3D array with consistent cell labels
#     tracked_stack = np.zeros_like(segmentation_stack, dtype=np.int32)
    
#     # Replace label values with particle IDs
#     for _, row in tqdm(tracked.iterrows(), desc="Applying tracked IDs", total=len(tracked)):
#         z = int(row['frame'])
#         particle_id = int(row['particle']) + 1  # Add 1 to ensure no zero labels
#         label = int(row['label'])
        
#         # Replace original label with particle ID
#         if z < tracked_stack.shape[0]:
#             mask = (segmentation_stack[z] == label)
#             tracked_stack[z][mask] = particle_id
    
#     return tracked_stack


# def process_dataset(input_dir: str, 
#                    output_dir: str, 
#                    blur_dir: Optional[str] = None,
#                    pattern: str = "*_3d.tif",
#                    blur_threshold: float = 0.5,
#                    measure_blur: bool = False,
#                    patch_size: int = 32,
#                    stride_size: int = 8):
#     """
#     Process a dataset of 3D segmentation files to generate tracked cell volumes.
    
#     Args:
#         input_dir: Directory containing 3D segmentation files
#         output_dir: Directory to save tracked segmentation files
#         blur_dir: Optional directory containing blur/sharpness maps
#         pattern: Pattern to match segmentation files
#         blur_threshold: Threshold for blur filtering
#         measure_blur: Whether to measure blur if blur maps are not available
#         patch_size: Patch size for blur measurement
#         stride_size: Stride size for blur measurement
#     """
#     os.makedirs(output_dir, exist_ok=True)
    
#     if blur_dir:
#         os.makedirs(blur_dir, exist_ok=True)
    
#     # Find all segmentation files
#     input_files = sorted(list(Path(input_dir).glob(pattern)))
    
#     for input_path in tqdm(input_files, desc="Processing files"):
#         try:
#             # Generate output file path
#             filename = input_path.name
#             output_filename = filename.replace(".tif", "_tracked.tif")
#             output_path = Path(output_dir) / output_filename
            
#             # Skip if output already exists
#             if output_path.exists():
#                 print(f"Skipping {filename}, output already exists")
#                 continue
            
#             # Read segmentation stack
#             segmentation_stack = tiff.imread(str(input_path))
            
#             # Find or generate blur map
#             blur_stack = None
#             if blur_dir:
#                 blur_path = Path(blur_dir) / filename.replace(".tif", "_blur.tif")
                
#                 if blur_path.exists():
#                     # Load existing blur map
#                     blur_stack = tiff.imread(str(blur_path))
#                 elif measure_blur:
#                     # Generate new blur map if requested
#                     # Try to find the corresponding brightfield image
#                     bf_path = input_path.with_name(filename.replace("_3d.tif", "_BF_3d.tif"))
#                     if bf_path.exists():
#                         # Measure blur from brightfield image
#                         bf_stack = tiff.imread(str(bf_path))
#                         blur_stack = np.zeros((bf_stack.shape[0], bf_stack.shape[1], bf_stack.shape[2]), 
#                                              dtype=np.float32)
                        
#                         for z in range(bf_stack.shape[0]):
#                             blur_stack[z] = measure_blur_heatmap(
#                                 bf_stack[z], patch_size=patch_size, stride_size=stride_size)
                        
#                         # Save blur map
#                         tiff.imwrite(str(blur_path), blur_stack.astype(np.float32))
            
#             # Track cells across z-stacks
#             tracked_stack = track_cells_3d(segmentation_stack, 
#                                           blur_stack=blur_stack, 
#                                           blur_threshold=blur_threshold)
            
#             # Save tracked segmentation
#             tiff.imwrite(str(output_path), tracked_stack)
            
#             print(f"Processed {filename} -> {output_filename}")
            
#         except Exception as e:
#             print(f"Error processing {input_path}: {e}")


# if __name__ == "__main__":
#     import argparse
    
#     parser = argparse.ArgumentParser(description="Track cells across z-stacks in 3D volumes")
#     parser.add_argument("--input-dir", required=True, help="Directory containing 3D segmentation files")
#     parser.add_argument("--output-dir", required=True, help="Directory to save tracked segmentation files")
#     parser.add_argument("--blur-dir", help="Directory containing blur/sharpness maps")
#     parser.add_argument("--pattern", default="*_3d.tif", help="Pattern to match segmentation files")
#     parser.add_argument("--blur-threshold", type=float, default=0.5, help="Threshold for blur filtering")
#     parser.add_argument("--measure-blur", action="store_true", help="Measure blur if blur maps are not available")
#     parser.add_argument("--patch-size", type=int, default=32, help="Patch size for blur measurement")
#     parser.add_argument("--stride-size", type=int, default=8, help="Stride size for blur measurement")
    
#     args = parser.parse_args()
    
#     process_dataset(
#         args.input_dir,
#         args.output_dir,
#         args.blur_dir,
#         args.pattern,
#         args.blur_threshold,
#         args.measure_blur,
#         args.patch_size,
#         args.stride_size
#     )
