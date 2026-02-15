import napari
import tifffile
import numpy as np
from pathlib import Path
import re
from collections import defaultdict
from tqdm import tqdm

def convert_samples_to_4d_tiffs(folder_path, output_dir=None):
    """
    Convert all samples in a folder to individual 4D TIFF files.
    Missing time points or z-stacks are filled with black (zero) images.
    Creates a log file detailing missing frames.
    """
    # Convert folder paths to Path objects
    folder = Path(folder_path)
    output_dir = Path(output_dir) if output_dir else folder / "4d_output"
    output_dir.mkdir(exist_ok=True)
    
    # Create log file
    log_file = output_dir / "missing_frames.log"
    
    # Get all tiff files in the folder
    tiff_files = list(folder.glob("*.tif"))
    
    # Group files by sample
    pattern = r't(\d+)_(.+?)_z(\d+)\.tif'
    sample_files = defaultdict(list)
    
    print("Grouping files by sample...")
    for file in tqdm(tiff_files, desc="Scanning files"):
        match = re.match(pattern, file.name)
        if match:
            t, sample, z = match.groups()
            sample_files[sample].append((file, int(t), int(z)))
    
    # Process each sample
    with open(log_file, 'w') as log:
        log.write("Missing Frames Report\n")
        log.write("===================\n\n")
        
        for sample in tqdm(sample_files.keys(), desc="Processing samples"):
            files = sample_files[sample]
            # Get all time and z values
            t_values = [t for _, t, _ in files]
            z_values = [z for _, _, z in files]
            
            t_min, t_max = min(t_values), max(t_values)
            z_min, z_max = min(z_values), max(z_values)
            
            # Get dimensions from first image
            first_img = tifffile.imread(files[0][0])
            height, width = first_img.shape
            
            # Create 4D array
            data = np.zeros((t_max - t_min + 1, z_max - z_min + 1, height, width), 
                           dtype=first_img.dtype)
            
            # Create a mapping of actual file data
            file_map = {(t, z): file_path for file_path, t, z in files}
            
            # Track missing frames
            missing_frames = []
            
            # Load images into the 4D array
            total_frames = (t_max - t_min + 1) * (z_max - z_min + 1)
            with tqdm(total=total_frames, desc=f"Loading {sample}", leave=False) as pbar:
                for t in range(t_min, t_max + 1):
                    for z in range(z_min, z_max + 1):
                        if (t, z) in file_map:
                            img = tifffile.imread(file_map[(t, z)])
                            data[t - t_min, z - z_min] = img
                        else:
                            missing_frames.append((t, z))
                        pbar.update(1)
            
            # Save as 4D TIFF
            output_file = output_dir / f"{sample}_4D.tif"
            tifffile.imwrite(output_file, data)
            print(f"Saved {sample} to {output_file}")
            print(f"  Time points: {t_max - t_min + 1} (range: {t_min}-{t_max})")
            print(f"  Z-stacks: {z_max - z_min + 1} (range: {z_min}-{z_max})")
            
            # Log missing frames
            if missing_frames:
                print(f"  Note: {len(missing_frames)} missing frames filled with zeros")
                log.write(f"\nSample: {sample}\n")
                log.write(f"Total missing frames: {len(missing_frames)}\n")
                log.write("Missing (t,z) pairs:\n")
                for t, z in sorted(missing_frames):
                    log.write(f"  t={t}, z={z}\n")
                log.write("-" * 40 + "\n")

def view_4d_tiff(file_path):
    """
    View a 4D TIFF file using napari.
    """
    # Load the 4D data
    data = tifffile.imread(file_path)
    
    # Create a napari viewer
    viewer = napari.Viewer()
    
    # Add the 4D data as an image layer
    viewer.add_image(
        data,
        name='4D Image',
        scale=(1, 1, 1, 1),  # Adjust if you have different pixel sizes
        contrast_limits=[data.min(), data.max()],
    )
    
    # Set dimension labels
    viewer.dims.axis_labels = ['T', 'Z', 'Y', 'X']
    
    # Start the napari event loop
    napari.run()

if __name__ == '__main__':
    # Example usage
    input_folder = '/Users/serenasritharan/Projects/single-cell/data/Timelapse Experiment Compressed'
    output_folder = '/Users/serenasritharan/Projects/single-cell/data/Timelapse Experiment Compressed_4D'  # Optional
    
    # Convert all samples to 4D TIFFs
    # convert_samples_to_4d_tiffs(input_folder, output_folder)
    
    # View a specific 4D TIFF file
    view_4d_tiff('/Users/serenasritharan/Projects/single-cell/data/Timelapse Experiment Compressed_4D/F08_s1_w1_4D.tif') 