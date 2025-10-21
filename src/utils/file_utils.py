"""
File utilities for cell segmentation datasets.

This module provides utilities for:
- Standardizing file names across different datasets
- Extracting metadata from filenames
- Organizing data into standard formats
- Supporting dataset splitting operations
- Handling different file naming conventions
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import re
import os
import glob
from tqdm import tqdm
import shutil
from collections import defaultdict


@dataclass
class FilePattern:
    """Container for file naming patterns."""
    pattern: str
    groups: List[str]
    output_format: str


@dataclass
class DatasetPaths:
    """Container for dataset file paths."""
    image_path: str
    mask_path: str
    output_dir: str

    def __init__(self, image_path: str, mask_path: str, output_dir: str):
        self.image_path = image_path
        self.mask_path = mask_path
        self.output_dir = output_dir
        self.get_files()

    def get_files(self):
        self.image_files = sorted(glob.glob(self.image_path, recursive=True))
        self.mask_files = sorted(glob.glob(self.mask_path, recursive=True))


class AbstractFileHandler(ABC):
    """Abstract base class for file renaming operations."""

    DEFAULT_PATTERNS = {
    }

    def __init__(self, patterns: Optional[Dict[str, FilePattern]] = None):
        """Initialize with optional custom patterns.

        Args:
            patterns: Dictionary of file type to FilePattern mappings
        """
        self.patterns = patterns or self.DEFAULT_PATTERNS

    @abstractmethod
    def rename_file(self, filename: str, file_type: str) -> str:
        """Rename file according to pattern for given file type.
        
        Args:
            filename: Original filename
            file_type: Type of file (e.g., 'image' or 'mask')
            
        Returns:
            Standardized filename
        """
        pass

    @abstractmethod
    def extract_group_id(self, filename: str) -> str:
        """Extract position grouping from output format.
        
        Args:
            filename: Standardized filename
            
        Returns:
            Group identifier extracted from filename
        """
        pass

    @abstractmethod
    def get_files(self, directory: str, file_type: str) -> List[str]:
        """Get list of files of given type from directory.
        
        Args:
            directory: Directory to search for files
            file_type: Type of file (from patterns)
            
        Returns:
            List of file paths
        """
        pass


class DefaultFileHandler(AbstractFileHandler):
    """Default implementation of file renaming with configurable patterns.
    
    Example input/output:
        Image:
            Input:  t1_A01_s1_w1_z1.tif
            Output: A01_z1_BF.tif
        
        Mask:
            Input:  Cells_R1-C1-F1-Z1-T1.tif
            Output: A01_z2_Cells.tif
    """

    DEFAULT_PATTERNS = {
        'image': FilePattern(
            pattern=r't\d+_([A-Z])(\d+)_s\d+_w\d+_z(\d+)',
            groups=['row', 'col', 'z'],
            output_format="{row}{col}_z{z}_BF.tif"
        ),
        'mask': FilePattern(
            pattern=r'Cells_R(\d+)-C(\d+)-F\d+-Z(\d+)-T\d+\.tif',
            groups=['row_num', 'col', 'z'],
            output_format="{row}{col}_z{z_adjusted}_Cells.tif"
        )
    }

    def __init__(self, patterns: Dict[str, FilePattern] | None = None):
        super().__init__(patterns)
        self.image_pattern = self.patterns['image'].pattern
        self.mask_pattern = self.patterns['mask'].pattern

    def get_files(self, directory: str, file_type: str) -> List[str]:
        if file_type == 'image':
            regex = re.compile(self.image_pattern)
        elif file_type == 'mask':
            regex = re.compile(self.mask_pattern)
        else:
            return []

        files = sorted(glob.glob(f"{directory}/**/*.tif", recursive=True))
        files = [f for f in files if regex.search(Path(f).name)]

        return files

    def rename_file(self, filename: str, file_type: str) -> str:
        if file_type not in self.patterns:
            raise ValueError(f"Unknown file type: {file_type}")

        filename = Path(filename).name
        pattern = self.patterns[file_type]
        match = re.search(pattern.pattern, filename)
        
        if not match:
            return filename

        values = dict(zip(pattern.groups, match.groups()))
        
        if file_type == 'mask':
            values['row'] = chr(ord('A') + int(values['row_num']) - 1)
            values['col'] = f"{int(values['col']):02d}"
            values['z_adjusted'] = str(int(values['z']) + 1)
        
        return pattern.output_format.format(**values)

    def rename_image(self, filename: str) -> str:
        return self.rename_file(filename, 'image')

    def rename_mask(self, filename: str) -> str:
        return self.rename_file(filename, 'mask')
        
    def extract_group_id(self, filename: str) -> str:
        """Extract position grouping from filename."""
        # Try to match standard pattern like "A01_z1_BF.tif"
        match = re.search(r'([A-Z]\d+)', filename)
        if match:
            return match.group(1)
            
        # Fallback to filename without extension
        return Path(filename).stem.split('_')[0]


class BF_IF_FileHandler(DefaultFileHandler):
    """
    File renamer for BF+IF experiments that includes plate number from filepath.
    This class is used to rename files in a specific format for BF+IF experiments.
    Input path contains the plate number, time point, position, well number, and z-stack.
    
    Example input/output:
        Image:
            Input:  Plate 2126/t1_A01_s1_w1_z1.tif
            Output: p2126_A01_z1_BF.tif
        
        Mask:
            Input:  Plate 2126/Cells_R1-C1-F1-Z1-T1.tif
            Output: p2126_A01_z2_Cells.tif
    """

    DEFAULT_PATTERNS = {
        'image': FilePattern(
            pattern=r't\d+_([A-Z])(\d+)_s\d+_w(\d+)_z(\d+)',
            groups=['row', 'col', 'well', 'z'],
            output_format="p{plate}_{row}{col}_z{z}_BF.tif"
        ),
        'mask': FilePattern(
            pattern=r'Cells_R(\d+)-C(\d+)-F\d+-Z(\d+)-T\d+\.tif',
            groups=['row_num', 'col', 'z'],
            output_format="p{plate}_{row}{col}_z{z}_Cells.tif"
        )
    }

    def rename_file(self, filepath: str, file_type: str) -> str:
        if file_type not in self.patterns:
            raise ValueError(f"Unknown file type: {file_type}")
                
        values = self.extract_values(filepath, file_type)

        if not values:
            raise ValueError(f"Could not extract values from filepath: {filepath}")
        
        plate_number = self.extract_plate_number(filepath)
        if not plate_number or plate_number == "unknown":
            raise ValueError(f"Could not extract plate number from filepath: {filepath}")
        else:
            values['plate'] = plate_number

        
        if file_type == 'mask':
            values['row'] = chr(ord('A') + int(values['row_num']) - 1)
            values['col'] = f"{int(values['col']):02d}"
            values['z'] = str(int(values['z']) + 1)
        else:
            # TODO: Support multiple wells (not needed for BF FileHandler)
            pass  # We still use well number in filename, but don't assert it equals '1'

        pattern = self.patterns[file_type]
        
        return pattern.output_format.format(**values)

    def rename_image(self, filename: str) -> str:
        print(self.image_pattern)
        return self.rename_file(filename, 'image')

    def rename_mask(self, filename: str) -> str:
        return self.rename_file(filename, 'mask')
    
    def extract_group_id(self, filename: str) -> str:
        """Extract position grouping from output format."""
        parts = filename.split('_')
        return parts[0] + "_" + parts[1]

    def extract_plate_number(self, filepath: str) -> str:
        """Extract plate number from filepath."""
        plate_match = re.search(r'Plate\s*(\d+)', filepath)
        if not plate_match:
            # Default to empty plate identifier if not found
            return "unknown"
        return plate_match.group(1)

    def extract_values(self, filepath: str, file_type: str) -> Dict[str, str]:
        """Extract values from filename based on pattern."""

        if file_type not in self.patterns:
            raise ValueError(f"Unknown file type: {file_type}")

        filename = Path(filepath).name
        pattern = self.patterns[file_type]
        match = re.search(pattern.pattern, filename)
        
        if not match:
            return {}

        values = dict(zip(pattern.groups, match.groups()))
        return values

class BF_FileHandler(BF_IF_FileHandler):
    """
    File renamer for BF only experiments that includes plate number from filepath and well number == 1.
    Input path contains the plate number, time point, position, well number, and z-stack.
    
    Example input/output:
        Image:
            Input:  Plate 2126/t1_A01_s1_w1_z1.tif
            Output: p2126_A01_z1_BF.tif
        
        Mask:
            Input:  Plate 2126/Cells_R1-C1-F1-Z1-T1.tif
            Output: p2126_A01_z2_Cells.tif
    """

    DEFAULT_PATTERNS = {
        # Brightfield images have a well no. of 1
        'image': FilePattern(
            pattern=r't\d+_([A-Z])(\d+)_s\d+_w(1)_z(\d+)',
            groups=['row', 'col', 'well', 'z'],
            output_format="p{plate}_{row}{col}_z{z}_BF.tif"
        ),
        'mask': FilePattern(
            pattern=r'Cells_R(\d+)-C(\d+)-F\d+-Z(\d+)-T\d+\.tif',
            groups=['row_num', 'col', 'z'],
            output_format="p{plate}_{row}{col}_z{z}_Cells.tif"
        )
    }


class BF_IF_FileHandler_3D(DefaultFileHandler):
    """File renamer for BF+IF 3D experiments that includes plate number from filepath.
    
    Example input/output:
        Image:
            Input:  Plate 2126/A01_BF_3D.tif
            Output: p2126_A01_BF.tif
        
        Mask:
            Input:  Plate 2126/A01_Cells_3D.tif
            Output: p2126_A01_Cells.tif
    """

    DEFAULT_PATTERNS = {
        'image': FilePattern(
            pattern=r'([A-Z])(\d+)_BF_3D',
            groups=['row', 'col'],
            output_format="p{plate}_{row}{col}_BF.tif"
        ),
        'mask': FilePattern(
            pattern=r'([A-Z])(\d+)_Cells_3D',
            groups=['row', 'col'],
            output_format="p{plate}_{row}{col}_Cells.tif"
        )
    }

    def extract_plate_number(self, filepath: str) -> str:
        """Extract plate number from filepath."""
        plate_match = re.search(r'Plate\s*(\d+)', filepath)
        if not plate_match:
            # Default to empty plate identifier if not found
            return "unknown"
        return plate_match.group(1)

    def rename_file(self, filename: str, file_type: str) -> str:
        """Rename file according to pattern for given file type."""
        if file_type not in self.patterns:
            raise ValueError(f"Unknown file type: {file_type}")

        # Extract plate number from filepath using regex
        plate_number = self.extract_plate_number(filename)

        filename = Path(filename).name
        pattern = self.patterns[file_type]
        match = re.search(pattern.pattern, filename)
        
        if not match:
            return filename

        values = dict(zip(pattern.groups, match.groups()))
        values['plate'] = plate_number
                
        return pattern.output_format.format(**values)

    def rename_image(self, filename: str) -> str:
        """Rename image file to standardized format."""
        return self.rename_file(filename, 'image')

    def rename_mask(self, filename: str) -> str:
        """Rename mask file to standardized format."""
        return self.rename_file(filename, 'mask')

    def extract_group_id(self, filename: str) -> str:
        """Extract position grouping from output format."""
        parts = filename.split('_')
        return parts[0] + "_" + parts[1]


# class StandardFileHandler(AbstractFileHandler):
#     """Standard implementation of file renaming for cell segmentation datasets.
    
#     This class converts various file naming conventions to a standardized format:
#     {plate}_{well}_z{z}_{type}.tif
    
#     Where:
#     - plate: plate identifier (e.g., p2126)
#     - well: well identifier (e.g., A01)
#     - z: z-stack number
#     - type: file type (e.g., BF for brightfield, Cells for segmentation masks)
#     """

#     DEFAULT_PATTERNS = {
#             'image': FilePattern(
#                 pattern=r't\d+_([A-Z])(\d+)_s\d+_w(\d+)_z(\d+)',
#                 groups=['row', 'col', 'well', 'z'],
#                 output_format="p{plate}_{row}{col:02d}_z{z}_{image_type}.tif"
#             ),
#             'mask': FilePattern(
#                 pattern=r'Cells_R(\d+)-C(\d+)-F\d+-Z(\d+)-T\d+\.tif',
#                 groups=['row_num', 'col', 'z'],
#                 output_format="p{plate}_{row}{col:02d}_z{z}_{mask_type}.tif"
#             )
#         }    

#     def extract_plate_number(self, filepath: str) -> str:
#         """Extract plate number from filepath."""
#         plate_match = re.search(r'Plate\s*(\d+)', filepath)
#         if not plate_match:
#             # Default to empty plate identifier if not found
#             return "unknown"
#         return plate_match.group(1)
    
#     def rename_file(self, filepath: str, file_type: str) -> str:
#         """Rename file according to pattern for given file type."""
#         if file_type not in self.patterns:
#             raise ValueError(f"Unknown file type: {file_type}")
        
#         # Extract plate number
#         plate_number = self.extract_plate_number(filepath)
        
#         # Extract filename from path
#         filename = Path(filepath).name
#         pattern = self.patterns[file_type]
#         match = re.search(pattern.pattern, filename)
        
#         if not match:
#             return filename
        
#         # Map matched groups to values
#         values = dict(zip(pattern.groups, match.groups()))
#         values['plate'] = plate_number
        
#         # Apply file type specific transformations
#         if file_type == 'mask':
#             # Convert 1-based row number to letter (1 -> A, 2 -> B, etc.)
#             values['row'] = chr(ord('A') + int(values['row_num']) - 1)
#             values['col'] = int(values['col'])
#             values['z'] = str(int(values['z']) + 1)  # Adjust z-index if needed
#             values['mask_type'] = 'Cells'
#         else:
#             values['col'] = int(values['col'])
#             values['image_type'] = 'BF'
        
#         return pattern.output_format.format(**values)
    
#     def rename_image(self, filepath: str) -> str:
#         """Rename image file to standardized format."""
#         return self.rename_file(filepath, 'image')
    
#     def rename_mask(self, filepath: str) -> str:
#         """Rename mask file to standardized format."""
#         return self.rename_file(filepath, 'mask')
        
#     def extract_group_id(self, filepath: str) -> str:
#         """
#         Extract position grouping from filename.
        
#         Args:
#             filepath: Path to the file
            
#         Returns:
#             Group identifier (e.g., 'p2126_A01')
#         """
#         filename = Path(filepath).name
#         # Try to match standard pattern like "p2126_A01_z1_BF.tif"
#         match = re.search(r'(p\d+_[A-Z]\d+)', filename)
#         if match:
#             return match.group(1)
        
#         # Try alternative pattern (plate_row-col)
#         match = re.search(r'([A-Z]\d+)', filename)
#         if match:
#             plate = self.extract_plate_number(filepath)
#             return f"p{plate}_{match.group(1)}"
            
#         # Fallback to filename without extension
#         return Path(filepath).stem.split('_')[0]



class BlurFileHandler(AbstractFileHandler):
    """Implementation of file renaming for blur maps with configurable patterns.
    
    """

    DEFAULT_PATTERNS = {
        'image_BF_3d': FilePattern(
            pattern = r'(.+?)(_BF_3d)?\.tif',
            groups=['key', 'image_suffix'],
            output_format="{key}{image_suffix}{blur_suffix}.tif"
        ),
    }

    def __init__(self, patterns: Dict[str, FilePattern] | None = None):
        super().__init__(patterns)
        self.blur_pattern = self.patterns['image_BF_3d'].pattern

    def rename_file(self, filename: Union[str, Path], file_type: str, suffix: str) -> str:
        if file_type not in self.patterns:
            raise ValueError(f"Unknown file type: {file_type}")

        filename = Path(filename).name
        pattern = self.patterns[file_type]
        match = re.search(pattern.pattern, filename)
        
        if not match:
            raise ValueError(f"Filename '{filename}' does not match pattern '{pattern.pattern}'")

        values = dict(zip(pattern.groups, match.groups()))

        if values['image_suffix'] is None:
            raise ValueError(f"Image suffix not found in filename '{filename}'")
        
        values['blur_suffix'] = suffix
        
        return pattern.output_format.format(**values)

    def rename_image(self, filename: Union[str, Path], suffix: str) -> str:
        return self.rename_file(filename, 'image_BF_3d', suffix)
        
    def extract_group_id(self, filename: str) -> str:
        """Extract position grouping from filename."""
        # Try to match standard pattern like "A01_z1_BF.tif"
        match = re.search(r'([A-Z]\d+)', filename)
        if match:
            return match.group(1)
            
        # Fallback to filename without extension
        return Path(filename).stem.split('_')[0]

    def get_files(self, directory: str, file_type: str) -> List[str]:
        if file_type == 'image' or file_type == 'image_BF_3d':
            return sorted(glob.glob(f"{directory}/**/{self.blur_pattern}.tif", recursive=True))
        return []

def get_groups_from_filenames(file_map: Dict[str, str], file_handler: AbstractFileHandler) -> Dict[str, List[str]]:
    """
    Group files based on a pattern in their filenames.
    
    Args:
        file_map: Dictionary mapping original file names to output names
        file_handler: File handler for extracting group info
        
    Returns:
        Dictionary mapping group identifiers to lists of file paths
    """
    groups = defaultdict(list)
    
    for filepath, out_path in file_map.items():
        group_id = file_handler.extract_group_id(out_path)
        groups[group_id].append(filepath)
            
    return dict(groups)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="File utilities for cell segmentation datasets")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Standardize dataset command
    standardize_parser = subparsers.add_parser("standardize", help="Standardize file names")
    standardize_parser.add_argument("--image-dir", required=True, help="Directory containing image files")
    standardize_parser.add_argument("--mask-dir", required=True, help="Directory containing mask files")
    standardize_parser.add_argument("--output-dir", required=True, help="Directory to save standardized files")
    
    # Example command
    example_parser = subparsers.add_parser("example", help="Show example file renaming")
    example_parser.add_argument("--file", required=True, help="Path to example file")
    example_parser.add_argument("--type", choices=["image", "mask"], required=True, help="File type")
    
    args = parser.parse_args()
    
    if args.command == "standardize":
        raise NotImplementedError("Standardize dataset functionality not implemented yet.")
    elif args.command == "example":
        handler = BF_IF_FileHandler()
        if args.type == "image":
            new_name = handler.rename_image(args.file)
        else:
            new_name = handler.rename_mask(args.file)
        print(f"Original: {args.file}")
        print(f"Renamed:  {new_name}")
