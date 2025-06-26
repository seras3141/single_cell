import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image

def analyze_tif_file(file_path):
    """
    Analyze a single TIF file and return its statistics.
    """
    try:
        with Image.open(file_path) as img:
            data = np.array(img)
            return {
                'min_value': data.min(),
                'max_value': data.max(),
                'unique_values': set(np.unique(data).tolist()),
                'current_dtype': data.dtype,
                'mode': img.mode,
                'shape': data.shape,
                'file_size': os.path.getsize(file_path)
            }
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def suggest_optimal_dtype(global_min, global_max):
    """
    Suggest the optimal data type based on the global value range.
    """
    if global_min >= 0:
        if global_max <= 255:
            return 'uint8'
        elif global_max <= 65535:
            return 'uint16'
        elif global_max <= 4294967295:
            return 'uint32'
    
    if global_min >= -128 and global_max <= 127:
        return 'int8'
    elif global_min >= -32768 and global_max <= 32767:
        return 'int16'
    elif global_min >= -2147483648 and global_max <= 2147483647:
        return 'int32'
    
    return 'float32'

def main():
    # Directory containing TIF files
    tif_dir = Path('/Users/surensritharan/Projects/single-cell/data/Timelapse Experiment Compressed')
    
    # Find all TIF files
    tif_files = list(tif_dir.glob('**/*.tif'))
    
    if not tif_files:
        print("No TIF files found in the specified directory.")
        return
    
    print(f"Analyzing {len(tif_files)} TIF files...")
    
    # Initialize dataset statistics
    dataset_stats = {
        'global_min': float('inf'),
        'global_max': float('-inf'),
        'total_size': 0,
        'unique_values': set(),
        'dtypes': set(),
        'modes': set(),
        'shapes': set()
    }
    
    # Analyze all files
    for tif_file in tqdm(tif_files):
        stats = analyze_tif_file(tif_file)
        if stats is None:
            continue
        
        # Update global statistics
        dataset_stats['global_min'] = min(dataset_stats['global_min'], stats['min_value'])
        dataset_stats['global_max'] = max(dataset_stats['global_max'], stats['max_value'])
        dataset_stats['total_size'] += stats['file_size']
        dataset_stats['unique_values'].update(stats['unique_values'])
        dataset_stats['dtypes'].add(str(stats['current_dtype']))
        dataset_stats['modes'].add(stats['mode'])
        dataset_stats['shapes'].add(str(stats['shape']))
    
    # Suggest optimal dtype
    suggested_dtype = suggest_optimal_dtype(dataset_stats['global_min'], dataset_stats['global_max'])
    
    # Calculate potential savings
    current_bits = max(np.dtype(dtype).itemsize * 8 for dtype in dataset_stats['dtypes'])
    suggested_bits = np.dtype(suggested_dtype).itemsize * 8
    potential_savings = (1 - suggested_bits/current_bits) * 100 if current_bits > suggested_bits else 0
    
    # Print results
    print("\nDataset Analysis Results:")
    print("-" * 50)
    print(f"Total number of files: {len(tif_files)}")
    print(f"Total dataset size: {dataset_stats['total_size'] / (1024 * 1024):.2f} MB")
    print(f"\nValue Range:")
    print(f"  - Global minimum: {dataset_stats['global_min']}")
    print(f"  - Global maximum: {dataset_stats['global_max']}")
    print(f"  - Total unique values: {len(dataset_stats['unique_values'])}")
    print(f"\nCurrent data types used: {', '.join(dataset_stats['dtypes'])}")
    print(f"Image modes found: {', '.join(dataset_stats['modes'])}")
    print(f"Image shapes found: {', '.join(dataset_stats['shapes'])}")
    
    print("\nOptimization Recommendation:")
    print("-" * 50)
    if potential_savings > 0:
        print(f"Recommended data type: {suggested_dtype}")
        print(f"Potential space savings: {potential_savings:.1f}%")
        print(f"Estimated new dataset size: {dataset_stats['total_size'] * (1 - potential_savings/100) / (1024 * 1024):.2f} MB")
        print("\nConversion is recommended as it will save space while preserving all unique values.")
    else:
        print("No optimization recommended - current data types are optimal for the value range.")
    
    # Print warnings if necessary
    if len(dataset_stats['dtypes']) > 1:
        print("\nWarning: Multiple data types found in dataset. Standardizing to a single type may be beneficial.")
    if len(dataset_stats['modes']) > 1:
        print("\nWarning: Multiple image modes found. Ensure compatibility before conversion.")
    if len(dataset_stats['shapes']) > 1:
        print("\nWarning: Multiple image shapes found in the dataset.")

if __name__ == "__main__":
    main() 