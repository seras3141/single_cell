import os
import glob
from collections import defaultdict
import numpy as np
import tifffile
import re
import logging
from tqdm import tqdm
import napari

def parse_filename(filename, mask):
    """Parse filename to extract sample ID and z position
    
    Args:
        filename: Name of the file to parse
        mask: If True, parse as mask file (Cells/Nuclei format)
              If False, parse as regular image file
    """
    basename = os.path.basename(filename)
    
    # Check if this is a Cells/Nuclei annotation file
    if mask:
        pattern = r'(?:Cells|Nuclei)_R(\d+)-C(\d+)-F\d+-Z(\d+)-T\d+\.tif'
        match = re.search(pattern, basename)
        if match:
            row, col, z = match.groups()
            # Convert row number to letter (1->A, 2->B, etc.)
            row_letter = chr(ord('A') + int(row) - 1)
            # Z position needs to be incremented by 1
            z_pos = int(z) + 1
            # Update column to be 2 digits
            col = f"{int(col):02d}"
            # Construct sample ID from row letter and column
            sample_id = f"{row_letter}{col}"
            return sample_id, z_pos
    
    # Handle regular image files (t1_J03_s1_w1_z1 format)
    else:
        parts = basename.split('_')
        if len(parts) >= 4:
            sample_id = parts[1]  # J03 in t1_J03_s1_w1_z1
            # Extract z position number from the last part
            z_pos = int(re.findall(r'z(\d+)', parts[-1])[0])
            return sample_id, z_pos
    
    return None, None

def combine_tiff_stacks(input_dir, output_dir, file_pattern, annotation_type=None, suffix=None):
    """
    Combine 2D TIFF files into 3D stacks based on sample ID
    
    Args:
        input_dir: Root directory to search for TIFF files
        output_dir: Directory to save combined 3D TIFF files
        file_pattern: File pattern to match (e.g., '*.tif')
        annotation_type: Type of annotation ('Cells', 'Nuclei', or None for raw images)
        suffix: Suffix to add to the output filename (optional)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary of dictionaries to store files by sample ID and z position
    sample_files = defaultdict(lambda: defaultdict(str))
    
    # Recursively find all matching files in subdirectories
    if annotation_type:
        # For Cells and Nuclei, look in all subfolders that start with the annotation type
        search_pattern = os.path.join(input_dir, '**', f'{annotation_type}_{file_pattern}')
    else:
        # For raw images, use the provided file pattern
        search_pattern = os.path.join(input_dir, '**', file_pattern)
    
    # Find all files first
    all_files = list(glob.glob(search_pattern, recursive=True))
    logging.info(f"Found {len(all_files)} files matching pattern {search_pattern}")
    
    # Group files by sample ID and z position with progress bar
    for filepath in tqdm(all_files, desc="Grouping files"):
        sample_id, z_pos = parse_filename(
            os.path.basename(filepath), 
            mask=bool(annotation_type)
        )
        if sample_id and z_pos:
            if sample_files[sample_id][z_pos]:
                logging.warning(f"Multiple files found for sample {sample_id} at z position {z_pos}")
            sample_files[sample_id][z_pos] = filepath
    
    # Process each sample
    for sample_id in tqdm(sample_files.keys(), desc="Processing samples"):
        z_positions = sorted(sample_files[sample_id].keys())
        if not z_positions:
            continue
        else:
            assert z_positions[0] == 1 and z_positions[-1] == len(z_positions), "Z positions are not consecutive"        

        # Read all images for this sample
        images = []
        for z_pos in z_positions:
            filepath = sample_files[sample_id][z_pos]
            img = tifffile.imread(filepath)
            images.append(img)
        
        # Stack images along z-axis
        combined_stack = np.stack(images, axis=0)
        
        # Create output filename with annotation type if applicable
        suffix = suffix if suffix else ""
        output_filename = f'{sample_id}_{suffix}_3D.tif'
        output_path = os.path.join(output_dir, output_filename)
        
        # Save combined stack
        tifffile.imwrite(output_path, combined_stack)
        logging.info(f"Created 3D stack for sample {sample_id}: {output_path}")

def setup_logging(output_dir, verbose=False):
    """Set up logging configuration
    
    Args:
        output_dir: Directory where to store the log file
        verbose: If True, also print logs to console
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up handlers
    handlers = [logging.FileHandler(os.path.join(output_dir, 'tiff_processing.log'))]
    if verbose:
        handlers.append(logging.StreamHandler())
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def view_3d_stack_with_annotations(image_path, cells_path=None, nuclei_path=None):
    """
    Visualize a 3D TIFF stack with its corresponding cell and nuclei annotations using napari
    
    Args:
        image_path: Path to the raw image 3D TIFF
        cells_path: Path to the cell annotation 3D TIFF (optional)
        nuclei_path: Path to the nuclei annotation 3D TIFF (optional)
    """
    # Load the image stacks
    raw_stack = tifffile.imread(image_path)
    cells_stack = tifffile.imread(cells_path) if cells_path else None
    nuclei_stack = tifffile.imread(nuclei_path) if nuclei_path else None
    
    # Create napari viewer
    viewer = napari.Viewer()
    
    # Add raw image layer
    viewer.add_image(
        raw_stack,
        name='Raw Image',
        colormap='gray',
        contrast_limits=[raw_stack.min(), raw_stack.max()]
    )
    
    # Add cell annotations if available
    if cells_stack is not None:
        viewer.add_labels(
            cells_stack,
            name='Cell Annotations',
            opacity=0.5
        )
    
    # Add nuclei annotations if available
    if nuclei_stack is not None:
        viewer.add_labels(
            nuclei_stack,
            name='Nuclei Annotations',
            opacity=0.5
        )
    
    # Start the napari event loop
    napari.run()

def main(args):
    # Base directory containing all images and annotations
    base_dir = "/Users/surensritharan/Projects/single-cell/data/BF+IF Experiments Labeled"
    output_dir = "/Users/surensritharan/Projects/single-cell/data/BF+IF Experiments Labeled_3D"

    stack = True
    view = not stack

    if stack:
        setup_logging(output_dir, args.verbose)
        logging.info("Starting TIFF processing")

        # Get all immediate subdirectories
        subdirs = [d for d in os.listdir(base_dir) 
                  if os.path.isdir(os.path.join(base_dir, d))]
        
        for subdir in subdirs:
            input_subdir = os.path.join(base_dir, subdir)
            output_subdir = os.path.join(output_dir, subdir)
            
            logging.info(f"\nProcessing subfolder: {subdir}")
            
            # Process raw images
            logging.info("Processing raw images...")
            combine_tiff_stacks(
                input_subdir,
                output_subdir,
                "*_w1_z*.tif",
                suffix="BF"
            )
            
            # Process cell annotations
            logging.info("Processing cell annotations...")
            combine_tiff_stacks(
                input_subdir,
                output_subdir,
                "*.tif",
                "Cells",
                suffix="Cells"
            )
            
            # Process nuclei annotations
            logging.info("Processing nuclei annotations...")
            combine_tiff_stacks(
                input_subdir,
                output_subdir,
                "*.tif",
                "Nuclei",
                suffix= "Nuclei"
            )
            
        logging.info("All stacking complete")

    elif view:
        image_path = '/Users/surensritharan/Projects/single-cell/data/BF+IF Experiments Labeled_3D/Plate 2126 Compressed - Timepoint  2hr/J03_3D.tif'
        # Extract sample ID from the image path
        sample_id = os.path.splitext(os.path.basename(image_path))[0].replace('_3D', '')
        cells_path = os.path.join(os.path.dirname(image_path), f"{sample_id}_Cells_3D.tif")
        nuclei_path = os.path.join(os.path.dirname(image_path), f"{sample_id}_Nuclei_3D.tif")
        
        if os.path.exists(image_path):
            logging.info(f"Visualizing 3D stack for sample {sample_id}")
            view_3d_stack_with_annotations(
                image_path,
                cells_path if os.path.exists(cells_path) else None,
                nuclei_path if os.path.exists(nuclei_path) else None
            )
        else:
            logging.error(f"Image file not found: {image_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process and view 3D TIFF stacks.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print processing information to console')
    args = parser.parse_args()
    
    main(args)