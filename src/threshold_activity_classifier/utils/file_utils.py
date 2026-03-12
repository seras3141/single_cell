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
from typing import List, Dict, Optional, Union
import re
import os
import glob
from collections import defaultdict


@dataclass
class FilePattern:
    """Container for file naming patterns."""
    pattern: str
    groups: List[str]
    output_format: Optional[str]

class AbstractFileHandler(ABC):
    """Abstract base class for file renaming operations."""

    def __init__(self, patterns: Optional[Dict[str, FilePattern]] = None):
        """Initialize with optional custom patterns.

        Args:
            patterns: Dictionary of file type to FilePattern mappings
        """
        self.patterns = patterns or {}

    @abstractmethod
    def rename_file(self, filename: str, file_type: str, target_type: Optional[str] = None) -> str:
        """Rename file according to pattern from file_type to target_type 

        Args:
            filename: Original filename
            file_type: Type of file (e.g., 'image' or 'mask')
            target_type: Target type of file (e.g., 'image' or 'mask'). Optional; when omitted, use file_type.

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

    def __init__(self, patterns: Optional[Dict[str, FilePattern]] = None):
        super().__init__(patterns)

        DEFAULT_PATTERNS = {
            'mask': FilePattern(
                pattern=r'Cells_R(\d+)-C(\d+)-F(\d+)-Z(\d+)-T(\d+)\.tif',
                groups=['row_num', 'col', 'f', 'z', 'time'],
                output_format="{row}{col}_z{z}_Cells.tif"
            ),
            'BF': FilePattern(
                pattern=r't(\d+)_([A-Z])(\d+)_s(\d+)_w(1)_z(\d+)',
                groups=['time', 'row', 'col', 'series', 'wavelength', 'z'],
                output_format="{row}{col}_z{z}_BF.tif"
            ),
            'image': FilePattern(
                pattern=r't(\d+)_([A-Z])(\d+)_s(\d+)_w([2-9])_z(\d+)',
                groups=['time', 'row', 'col', 'series', 'wavelength', 'z'],
                output_format="{row}{col}_z{z}_w{wavelength}.tif"
            ),
        }

        self.patterns = patterns or DEFAULT_PATTERNS


    def get_files(self, directory: str, file_type: str) -> List[str]:
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Error: Directory {directory} does not exist!")
        
        if file_type not in self.patterns:
            raise ValueError(f"Unknown file type: {file_type}")

        pattern = self.patterns[file_type].pattern
        regex = re.compile(pattern)
        
        files = sorted(glob.glob(f"{directory}/**/*.tif", recursive=True))
        files = [f for f in files if regex.search(Path(f).name)]

        return files

    def rename_file(self, filename: str, file_type: str, target_type: Optional[str]=None) -> str:
        if file_type not in self.patterns:
            raise ValueError(f"Unknown file type: {file_type}")
        
        values = self.extract_values(filename, file_type)

        if target_type:
            if target_type not in self.patterns:
                raise ValueError(f"Unknown target file type: {target_type}")
            pattern = self.patterns.get(target_type)
        else:
            pattern = self.patterns[file_type]
        
        if pattern is None or pattern.output_format is None:
            raise ValueError(f"No output format defined for file type: {file_type}")

        return pattern.output_format.format(**values)

    def rename_image(self, filename: str) -> str:
        return self.rename_file(filename, 'BF')   # type: ignore

    def rename_mask(self, filename: str) -> str:
        return self.rename_file(filename, 'mask') # type: ignore
        
    def extract_group_id(self, filename: str) -> str:
        """Extract position grouping from filename."""
        # Try to match standard pattern like "A01_z1_BF.tif"
        match = re.search(r'([A-Z]\d+)', filename)
        if match:
            return match.group(1)
            
        # Fallback to filename without extension
        return Path(filename).stem.split('_')[0]

    def extract_sample_id(self, filename: str) -> str | None:
        """Extract sample ID (e.g., B02) from filename like p2426_B02_z10_w2.tif"""
        match = re.search(r'([A-Z]\d+)', filename)
        if match:
            return match.group(1)
        return None

    def extract_sample_z_info(self, filename: str):
        """Extract sample ID and z-index from image filename"""
        match = re.search(r'([A-Z]\d+)_z(\d+)', filename)
        if match:
            sample_id = match.group(1)
            z_index = int(match.group(2))
            return sample_id, z_index
        return None, None

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

        if file_type == 'mask':
            values['row'] = chr(ord('A') + int(values['row_num']) - 1)
            values['col'] = f"{int(values['col']):02d}"
            values['z'] = str(int(values['z']) + 1)
            values['time'] = str(int(values['time']) + 1)
        else:
            pass  # We still use well number in filename, but don't assert it equals '1'

        return values


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

    def __init__(self, patterns: Dict[str, FilePattern] | None = None):
        super().__init__(patterns)
        self.patterns['BF'].output_format = "p{plate}_{row}{col}_z{z}_BF.tif"
        self.patterns['image'].output_format = "p{plate}_{row}{col}_z{z}_w{wavelength}.tif"
        self.patterns['mask'].output_format = "p{plate}_{row}{col}_z{z}_Cells.tif"

    def rename_file(self, filepath: str, file_type: str, target_type: Optional[str] = None) -> str:
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

        if target_type:
            if target_type not in self.patterns:
                raise ValueError(f"Unknown target file type: {target_type}")
            pattern = self.patterns.get(target_type)
        else:
            pattern = self.patterns[file_type]

        return pattern.output_format.format(**values) # type: ignore

    
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

    def __init__(self, patterns: Dict[str, FilePattern] | None = None):
        super().__init__(patterns)
        self.patterns.pop('image', None)  # Remove 'image' pattern since we only have BF and mask

class ImageTimeFileHandler(BF_IF_FileHandler):
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

    def __init__(self, patterns: Dict[str, FilePattern] | None = None):
        super().__init__(patterns)
        self.patterns['BF'].output_format = "p{plate}_{row}{col}_t{time}_z{z}_BF.tif"
        self.patterns['image'].output_format = "p{plate}_{row}{col}_t{time}_z{z}_w{wavelength}.tif"
        self.patterns['mask'].output_format = "p{plate}_{row}{col}_t{time}_z{z}_Cells.tif"


class StandardFileHandler(AbstractFileHandler):
    """Standardized file naming after renaming.

    This class reads various file naming conventions to a standardized format:
    {plate}_{well}_z{z}_{type}.tif
    
    """

    def __init__(self, patterns: Optional[Dict[str, FilePattern]] = None):
        super().__init__(patterns)

        DEFAULT_PATTERNS = {
            'file': FilePattern(
                pattern=r'p(\d+)_t(\d+)_([A-Z])(\d+)_z(\d+)_(.+)',
                groups=['plate', 'time', 'row', 'col', 'z', 'type'],
                output_format="{plate}_{row}{col}_z{z}_w{type}.tif"
            ),
        }

        self.patterns = patterns or DEFAULT_PATTERNS

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
    
    def rename_file(self, filepath: str, file_type: str, target_type: Optional[str] = None) -> str:
        if file_type not in self.patterns:
            raise ValueError(f"Unknown file type: {file_type}")
                
        values = self.extract_values(filepath, file_type)

        if not values:
            raise ValueError(f"Could not extract values from filepath: {filepath}")
        
        if target_type:
            if target_type not in self.patterns:
                raise ValueError(f"Unknown target file type: {target_type}")
            pattern = self.patterns.get(target_type)
        else:
            pattern = self.patterns[file_type]

        return pattern.output_format.format(**values) # type: ignore


    def extract_group_id(self, filename: str) -> str:
        raise NotImplementedError("StandardFileHandler does not implement extract_group_id.")

    def get_files(self, directory: str, file_type: str) -> List[str]:
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Error: Directory {directory} does not exist!")
        
        if file_type not in self.patterns:
            raise ValueError(f"Unknown file type: {file_type}")

        pattern = self.patterns[file_type].pattern
        regex = re.compile(pattern)
        
        files = sorted(glob.glob(f"{directory}/**/*.tif", recursive=True))
        files = [f for f in files if regex.search(Path(f).name)]

        return files


class BlurFileHandler(AbstractFileHandler):
    """Implementation of file renaming for blur maps with configurable patterns.
    
    """


    def __init__(self, patterns: Optional[Dict[str, FilePattern]] = None):
        super().__init__(patterns)

        DEFAULT_PATTERNS = {
            'file': FilePattern(
                pattern=r'p(\d+)_([A-Z])(\d+)_z(\d+)_(.+)',
                groups=['plate', 'row', 'col', 'z', 'type'],
                output_format="{plate}_{row}{col}_z{z}_w{type}.tif"
            ),
            'file_3D': FilePattern(
                pattern=r'p(\d+)_([A-Z])(\d+)_(.+)',
                groups=['plate', 'row', 'col', 'type'],
                output_format="{plate}_{row}{col}_w{type}.tif"
            ),
            # 'BF_3d': FilePattern(
            #     pattern = r'(.+?)(_BF_3d)?\.tif',
            #     groups=['key', 'image_suffix'],
            #     output_format="{key}{image_suffix}{blur_suffix}.tif"
            # ),
        }

        self.patterns = patterns or DEFAULT_PATTERNS

    def rename_file(self, filename: str, file_type: str, target_type: Optional[str] = None) -> str:

        value = self.extract_values(filename, file_type)

        if not value:
            raise ValueError(f"Could not extract values from filename: {filename}")

        if target_type:
            if target_type not in self.patterns:
                raise ValueError(f"Unknown target file type: {target_type}")
            pattern = self.patterns.get(target_type)
        else:
            pattern = self.patterns[file_type]

        return pattern.output_format.format(**value) # type: ignore


    def rename_image(self, filename: str, suffix: str) -> str:

        values = self.extract_values(filename, 'file_3D')

        if not values:
            raise ValueError(f"Could not extract values from filename: {filename}")
        
        if values['type'] != 'BF_3d':
            raise ValueError(f"Image suffix BF_3d not found in filename '{filename}'")
        
        # Add blur suffix to end of filename
        base = Path(filename).stem
        filename = f"{base}{suffix}.tif"

        return filename

        # values['blur_suffix'] = suffix
        
        # return pattern.output_format.format(**values)
        
    def extract_group_id(self, filename: str) -> str:
        """Extract position grouping from filename."""
        # Try to match standard pattern like "A01_z1_BF.tif"
        match = re.search(r'([A-Z]\d+)', filename)
        if match:
            return match.group(1)
            
        # Fallback to filename without extension
        return Path(filename).stem.split('_')[0]

    def get_files(self, directory: str, file_type: str) -> List[str]:
        if file_type not in self.patterns:
            raise ValueError(f"Unknown file type: {file_type}")
        pattern = self.patterns[file_type]

        return sorted(glob.glob(f"{directory}/**/*{pattern.pattern}.tif", recursive=True))

    def extract_values(self, filepath: str, file_type: str) -> Dict[str, str]:
        """Extract values from filename based on pattern."""

        if file_type not in self.patterns:
            raise ValueError(f"Unknown file type: {file_type}")

        filename = Path(filepath).stem
        pattern = self.patterns[file_type]
        match = re.search(pattern.pattern, filename)

        if not match:
            return {}

        values = dict(zip(pattern.groups, match.groups()))
        return values

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
    example_parser.add_argument("--type", choices=["BF", "mask"], required=True, help="File type")
    
    args = parser.parse_args()
    
    if args.command == "standardize":
        raise NotImplementedError("Standardize dataset functionality not implemented yet.")
    elif args.command == "example":
        handler = BF_IF_FileHandler()
        if args.type == "BF":
            new_name = handler.rename_image(args.file)
        else:
            new_name = handler.rename_mask(args.file)
        print(f"Original: {args.file}")
        print(f"Renamed:  {new_name}")
