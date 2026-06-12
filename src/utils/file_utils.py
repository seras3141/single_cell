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
from typing import List, Dict, Optional, Union, Tuple
import re
import os
import glob
import logging
from collections import defaultdict
import shutil


@dataclass
class FilePattern:
    """Container for file naming patterns."""
    pattern: str
    groups: List[str]
    output_format: Optional[str]


# Ew2-1, Ew2-2: apoptosis marker (FlipGFP) on w1, brightfield on w3
WAVELENGTH_MAPPINGS_EW2: Dict[int, str] = {1: "FlipGFP", 2: "mCherry", 3: "BF"}

# HD1509, HD1883, SA110: brightfield on w1, apoptosis marker (FlipGFP) on w3
WAVELENGTH_MAPPINGS_HD_SA: Dict[int, str] = {1: "BF", 2: "mCherry", 3: "FlipGFP"}

# Default uses the Ew2 convention
DEFAULT_WAVELENGTH_MAPPINGS: Dict[int, str] = WAVELENGTH_MAPPINGS_EW2

# Lookup by experiment name — pass EXPERIMENT_WAVELENGTH_MAPPINGS[experiment_name]
# to ConfigurableFileHandler(wavelength_mappings=...) at the call site.
EXPERIMENT_WAVELENGTH_MAPPINGS: Dict[str, Dict[int, str]] = {
    "Ew2-1": WAVELENGTH_MAPPINGS_EW2,
    "Ew2-2": WAVELENGTH_MAPPINGS_EW2,
    "HD1509": WAVELENGTH_MAPPINGS_HD_SA,
    "HD1883": WAVELENGTH_MAPPINGS_HD_SA,
    "SA110": WAVELENGTH_MAPPINGS_HD_SA,
}

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
    def extract_unique_id(self, filename: str) -> str:
        """
        Extract unique identifier from standardized filename (without z-stack).
        
        Args:
            filename: Standardized filename
            
        Returns:
            Unique identifier extracted from filename
        """
        pass

    @abstractmethod
    def get_files_by_type(self, directory: str, file_type: str) -> List[str]:
        """Get list of files of given type from directory.
        
        Args:
            directory: Directory to search for files
            file_type: Type of file (from patterns)
            
        Returns:
            List of file paths
        """
        pass

    @abstractmethod
    def get_file_type(self, filepath: str) -> str:
        """Determine file type based on filename pattern.
        
        Args:
            filepath: Path to the file
            
        Returns:
            File type as defined in patterns
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
            'image': FilePattern(
                pattern=r't(\d+)_([A-Z])(\d+)_s(\d+)_w([1-9])_z(\d+)',
                groups=['time', 'row', 'col', 'series', 'wavelength', 'z'],
                output_format="{row}{col}_z{z}_w{wavelength}.tif"
            ),
            'processed': FilePattern(
                pattern=r'p(\d+)_t(\d+)_([A-Z])(\d+)_z(\d+)_(.+)',
                groups=['plate', 'time', 'row', 'col', 'z', 'type'],
                output_format="{plate}_{row}{col}_z{z}_w{type}.tif"
            ),
        }

        self.patterns = patterns or DEFAULT_PATTERNS


    def get_files_by_type(self, directory: str, file_type: str) -> List[str]:
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Error: Directory {directory} does not exist!")
        
        if file_type not in self.patterns:
            raise ValueError(f"Unknown file type: {file_type}")

        pattern = self.patterns[file_type].pattern
        regex = re.compile(pattern)
        
        files = sorted(glob.glob(f"{directory}/**/*.tif", recursive=True))
        files = [f for f in files if regex.search(Path(f).name)]

        return files
    
    def get_file_type(self, filepath: str) -> str:
        filename = Path(filepath).name
        for file_type, pattern in self.patterns.items():
            if re.search(pattern.pattern, filename):
                return file_type
        raise ValueError(f"Unknown file type for filepath: {filepath}")
    
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
        return self.rename_file(filename, 'image')   # type: ignore

    def rename_mask(self, filename: str) -> str:
        return self.rename_file(filename, 'mask') # type: ignore
        
    def extract_unique_id(self, filename: str) -> str:
        """Extract position grouping from filename."""
        # Try to match standard pattern like "A01_z1_BF.tif"
        sample_id = self.extract_sample_id(filename)
        if sample_id:
            return sample_id
            
        # Fallback to filename without extension
        return Path(filename).stem.split('_')[0]

    def extract_sample_id(self, filename: str) -> str | None:
        """Extract sample ID (e.g., B02) from filename like p2426_B02_z10_w2.tif"""
        match = re.search(r'_([A-Z]\d+)_', filename)
        if match:
            return match.group(1)
        return None

    def extract_z_index(self, filename: str):
        """Extract z-index from image filename"""
        match = re.search(r'_z(\d+)', filename)
        if match:
            return int(match.group(1))
        return None

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
    Default FileHandler extended to contain plate number, and time point.
    
    Example input/output:
        Image:
            Input:  Plate 2126/t1_A01_s1_w1_z1.tif
            Output: p2126_A01_t1_z1_BF.tif
        
        Mask:
            Input:  Plate 2126/Cells_R1-C1-F1-Z1-T1.tif
            Output: p2126_A01_t1_z2_Cells.tif
    """

    def __init__(self, patterns: Dict[str, FilePattern] | None = None):
        super().__init__(patterns)
        self.patterns['image'].output_format = "p{plate}_{row}{col}_t{time}_z{z}_w{wavelength}.tif"
        self.patterns['mask'].output_format = "p{plate}_{row}{col}_t{time}_z{z}_Cells.tif"

    def rename_file(self, filepath: str, file_type: str, target_type: Optional[str] = None, plate_number: Optional[str] = None) -> str:
        filepath = str(filepath)
        
        if file_type not in self.patterns:
            raise ValueError(f"Unknown file type: {file_type}")
                
        values = self.extract_values(filepath, file_type)

        if not values:
            raise ValueError(f"Could not extract values from filepath: {filepath}")
        
        # Use explicit plate_number if provided, otherwise extract from filepath
        if plate_number is None:
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

    def extract_unique_id(self, filename: str) -> str:
        """Extract unique identifier from standardized filename (without z-stack)."""
        parts = filename.split('_')
        if len(parts) < 3:
            raise ValueError(f"Filename does not contain enough parts to extract unique ID: {filename} : p{{plate}}_{{row}}{{col}}_t{{time}}_z{{z}}")
        return f"{parts[0]}_{parts[1]}_{parts[2]}"  # plate_rowcol_time

    def extract_time_point(self, filename: str) -> str:
        """Extract time point from filename."""
        match = re.search(r'_t(\d+)', filename)
        if match:
            return match.group(1)
        return "unknown"

    def extract_plate_number(self, filepath: str) -> str:
        """Extract plate number from filepath.
        
        Attempts extraction from multiple sources:
        1. Filename itself (e.g., 'p2126_A01_z1.tif' -> '2126')
        2. Parent directory path (e.g., 'Plate 2126/...' -> '2126')
        
        Args:
            filepath: Path to file or filename
            
        Returns:
            Plate number as string, or "unknown" if not found
        """
        # First, try to extract from filename itself (e.g., p2126_A01_z1.tif or pMF5V1_A01_z1.tif)
        filename = Path(filepath).name
        plate_match = re.search(r"p([A-Za-z0-9]+)_", filename)
        if plate_match:
            return plate_match.group(1)
        
        # Fall back to directory path (e.g., Plate 2126/)
        plate_match = re.search(r'Plate\s*(\d+)', str(filepath))
        if plate_match:
            return plate_match.group(1)
        
        # Default to unknown plate identifier if not found
        return "unknown"

class ConfigurableFileHandler(BF_IF_FileHandler):
    """File handler with configurable wavelength-to-channel mappings and flexible plate extraction.
    
    This handler extends BF_IF_FileHandler to support:
    - Dynamic wavelength mappings (e.g. w1->FlipGFP, w2->mCherry, w3->BF for Ew2 experiments)
    - Flexible plate number extraction from filename or directory
    - Runtime parameter overrides for maximum flexibility

    Wavelength Configuration:
    - Defaults to WAVELENGTH_MAPPINGS_EW2 ({1: FlipGFP, 2: mCherry, 3: BF})
    - Override per experiment: ConfigurableFileHandler(wavelength_mappings=EXPERIMENT_WAVELENGTH_MAPPINGS["HD1509"])

    Plate Number Extraction:
    - Attempts extraction from filename first (e.g., 'p2126_A01_z1.tif')
    - Falls back to directory path (e.g., 'Plate 2126/')
    - Can be explicitly provided to rename_file() to override auto-detection
    
    Example usage:
        # Default mappings for Ew2 experiments (w1=FlipGFP, w2=mCherry, w3=BF)
        handler = ConfigurableFileHandler()
        renamed = handler.rename_file('Plate 2126/t1_A01_s1_w2_z1.tif', 'image')
        # Output: p2126_A01_z1_mCherry.tif

        # Custom mappings at runtime
        custom_mappings = {1: "BF", 2: "GFP", 3: "RFP"}
        handler = ConfigurableFileHandler(wavelength_mappings=custom_mappings)
        renamed = handler.rename_file('Plate 2126/t1_A01_s1_w2_z1.tif', 'image')
        # Output: p2126_A01_z1_GFP.tif

        # Explicit plate number override
        renamed = handler.rename_file('Plate 2126/t1_A01_s1_w2_z1.tif', 'image', plate_number='9999')
        # Output: p9999_A01_z1_mCherry.tif
    """

    def __init__(
        self,
        wavelength_mappings: Optional[Dict[int, str]] = None,
        patterns: Optional[Dict[str, FilePattern]] = None,
        plate_number: Optional[str] = None
    ):
        """Initialize ConfigurableFileHandler with optional configuration.

        Args:
            wavelength_mappings: Dictionary mapping wavelength indices to channel names.
                Defaults to WAVELENGTH_MAPPINGS_EW2 ({1: "FlipGFP", 2: "mCherry", 3: "BF"}).
                Use EXPERIMENT_WAVELENGTH_MAPPINGS[experiment_name] for HD/SA experiments.
            patterns: Custom file patterns. If None, uses inherited defaults.
            plate_number: Default plate number to use if not extracted from filepath.
                Can be overridden in rename_file() call.
        """
        super().__init__(patterns)

        self.wavelength_mappings = wavelength_mappings if wavelength_mappings is not None else DEFAULT_WAVELENGTH_MAPPINGS.copy()
        
        # Store default plate number
        self._default_plate_number = plate_number
        
        # Update 'image' output format to use dynamic channel names
        # The actual channel name will be substituted during rename_file()
        self.patterns['image'].output_format = "p{plate}_{row}{col}_t{time}_z{z}_{channel_name}.tif"

    def get_channel_name(self, wavelength_index: int) -> str:
        """Get channel name for a wavelength index.
        
        Args:
            wavelength_index: Wavelength index (e.g., 1, 2, 3)
            
        Returns:
            Channel name (e.g., "BF", "mCherry", "FlipGFP")
            
        Raises:
            ValueError: If wavelength index not found in mappings
        """
        if wavelength_index not in self.wavelength_mappings:
            raise ValueError(
                f"No channel mapping for wavelength {wavelength_index}. "
                f"Available: {list(self.wavelength_mappings.keys())}. "
                f"Current mappings: {self.wavelength_mappings}"
            )
        return self.wavelength_mappings[wavelength_index]

    def rename_file(
        self,
        filepath: str,
        file_type: str,
        target_type: Optional[str] = None,
        plate_number: Optional[str] = None
    ) -> str:
        """Rename file with dynamic wavelength-to-channel mapping.
        
        Overrides parent to substitute channel names for 'image' file type.
        
        Args:
            filepath: Path to input file
            file_type: Type of file ('BF', 'image', 'mask')
            target_type: Target file type for conversion. If None, uses file_type.
            plate_number: Explicit plate number to use. If None, auto-extracts from filepath.
                If auto-extraction fails, uses _default_plate_number if set.
                
        Returns:
            Renamed filename with plate prefix and channel names
            
        Raises:
            ValueError: If file_type unknown or extraction/mapping fails
        """
        if file_type not in self.patterns:
            raise ValueError(f"Unknown file type: {file_type}")
        
        # Extract values from filename
        values = self.extract_values(filepath, file_type)
        
        if not values:
            raise ValueError(f"Could not extract values from filepath: {filepath}")
        
        # Determine plate number: explicit > extracted > default > error
        if plate_number is None:
            plate_number = self.extract_plate_number(filepath)
        
        if (not plate_number or plate_number == "unknown") and self._default_plate_number:
            plate_number = self._default_plate_number
        
        if not plate_number or plate_number == "unknown":
            raise ValueError(f"Could not extract plate number from filepath: {filepath}")
        
        values['plate'] = plate_number
        
        # For 'image' file type, map wavelength to channel name
        if file_type == 'image' and 'wavelength' in values:
            wavelength_index = int(values['wavelength'])
            channel_name = self.get_channel_name(wavelength_index)
            values['channel_name'] = channel_name
        
        # Select target pattern
        if target_type:
            if target_type not in self.patterns:
                raise ValueError(f"Unknown target file type: {target_type}")
            pattern = self.patterns.get(target_type)
        else:
            pattern = self.patterns[file_type]
        
        return pattern.output_format.format(**values)  # type: ignore


class BlurFileHandler(BF_IF_FileHandler):
    """
    File handler for blur heatmap files, extending BF_IF_FileHandler.

    Inherits plate number, time point, sample ID, and z-index extraction from
    BF_IF_FileHandler, and adds support for converting image filenames (either
    with z-stack or 3D) into corresponding blur heatmap filenames.

    Input patterns recognised:
        file (2D/z-stack): p<plate>_<row><col>_t<time>_z<z>_<type>.tif
                           e.g. p2126_A01_t1_z1_BF.tif
        file_3D:           p<plate>_<row><col>_t<time>_<type>.tif
                           e.g. pMF5V1_C09_t11_BF_3d.tif

    Example usage:
        handler = BlurFileHandler()
        handler.rename_image('p2126_A01_t1_z1_BF.tif')
        # -> 'p2126_A01_t1_z1_BF_blur_heatmap.tif'
        handler.rename_image('pMF5V1_C09_t11_BF_3d.tif')
        # -> 'pMF5V1_C09_t11_BF_3d_blur_heatmap.tif'
    """

    BLUR_SUFFIX = "_blur_heatmap"

    def __init__(self, patterns: Optional[Dict[str, FilePattern]] = None):
        super().__init__(None)  # Initialise parent with its defaults

        # Add blur-specific patterns on top of the inherited image/mask patterns.
        # Plate IDs may be alphanumeric (e.g. 'MF5V1') or purely numeric (e.g. '2126').
        self.patterns['file'] = FilePattern(
            pattern=r'p([A-Za-z0-9]+)_([A-Z])(\d+)_t(\d+)_z(\d+)_(.+)',
            groups=['plate', 'row', 'col', 'time', 'z', 'type'],
            output_format="{plate}_{row}{col}_t{time}_z{z}_{type}.tif"
        )
        self.patterns['file_3D'] = FilePattern(
            pattern=r'p([A-Za-z0-9]+)_([A-Z])(\d+)_t(\d+)_(.+)',
            groups=['plate', 'row', 'col', 'time', 'type'],
            output_format="{plate}_{row}{col}_t{time}_{type}.tif"
        )

        # Allow full override via constructor argument
        if patterns:
            self.patterns.update(patterns)

    def rename_file(self, filename: str, file_type: str, target_type: Optional[str] = None, plate_number: Optional[str] = None) -> str:
        """Rename a blur file using the configured patterns.

        For 'file' and 'file_3D' types the plate number is already embedded in
        the filename and is extracted from the pattern groups directly.
        """
        values = self.extract_values(filename, file_type)

        if not values:
            raise ValueError(f"Could not extract values from filename: {filename}")

        if target_type:
            if target_type not in self.patterns:
                raise ValueError(f"Unknown target file type: {target_type}")
            pattern = self.patterns.get(target_type)
        else:
            pattern = self.patterns[file_type]

        return pattern.output_format.format(**values)  # type: ignore

    def rename_image(self, filename: str, suffix: str = BLUR_SUFFIX) -> str:
        """Convert an image filename to the corresponding blur heatmap filename.

        Works for both z-stack (2D) and 3D image filenames by appending
        *suffix* before the .tif extension.

        Args:
            filename: Input image filename, either:
                      - 2D/z-stack: e.g. 'p2126_A01_t1_z1_BF.tif'
                      - 3D:         e.g. 'pMF5V1_C09_t11_BF_3d.tif'
            suffix: String to append to the stem. Defaults to '_blur_heatmap'.

        Returns:
            Blur heatmap filename with *suffix* inserted before the extension.

        Raises:
            ValueError: If *filename* does not match the 'file' or 'file_3D' pattern.
        """
        # Check 'file' first (more specific — requires z component) then 'file_3D'.
        for file_type in ('file', 'file_3D'):
            if self.extract_values(filename, file_type):
                base = Path(filename).stem
                return f"{base}{suffix}.tif"

        raise ValueError(
            f"Could not match '{filename}' to any known blur pattern. "
            f"Expected 'p<plate>_<row><col>_t<time>_z<z>_<type>.tif' "
            f"or 'p<plate>_<row><col>_t<time>_<type>.tif'."
        )

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
        group_id = file_handler.extract_unique_id(out_path)
        groups[group_id].append(filepath)
            
    return dict(groups)

def list_all_files(directory: str, file_handler: AbstractFileHandler) -> Dict[str, List[str]]:
    """List files in a directory using the file handler."""
    file_list = {}
    for k, v in file_handler.patterns.items():
        file_list[k] = file_handler.get_files_by_type(directory, k)
    
    return file_list

def rename_all_files(file_map: Dict[str, List[str]], file_handler: AbstractFileHandler):
    """Rename files according to file_map."""
    renamed_file_map = defaultdict(list)

    for file_type, files in file_map.items():
        for filepath in files:
            new_name = file_handler.rename_file(filepath, file_type)
            renamed_file_map[file_type].append((filepath, new_name))

    return renamed_file_map

def copy_file(
    src_file: Union[str, Path],
    dest_file: Union[str, Path],
    overwrite: bool = False,
) -> None:
    """
    Copy a file from src_file to dest_file with metadata preservation.

    Args:
        src_file: Source file to copy
        dest_file: Destination file path
        overwrite: If False (default), skip files that already exist at dest_file.
    """

    src = Path(src_file).resolve()
    dst = Path(dest_file)

    # Ensure the parent directory exists
    dst.parent.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        raise FileNotFoundError(f"Source file '{src}' does not exist")

    if dst.exists() or dst.is_symlink():
        if not overwrite:
            logging.debug(f"Skipping {dst} — file already exists")
            return
        dst.unlink()  # Remove existing file before overwriting

    try:
        dst.symlink_to(src, target_is_directory=False)
    except (OSError, NotImplementedError):
        # Use shutil.copy2 which preserves metadata instead of symlinks
        # This works on all platforms without admin privileges
        shutil.copy2(src, dst)

def copy_without_split_dict(
        file_tuple : Dict[str, List[Tuple[str, str]]],
        output_dir: Union[str, Path],
        overwrite: bool = False,
    ):
    """
    Copy image and mask files without splitting into train/test sets.

    Args:
        file_tuple: Dictionary containing file tuples {file_type: [(src_path, dest_path)]}
        output_dir: Directory to copy the files to
        overwrite: If False (default), skip files that already exist.
    """
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for k, v in file_tuple.items():
        for file_pair in v:
            src, dst = file_pair
            copy_file(src, output_dir / dst, overwrite=overwrite)
            count += 1
    return count

def copy_with_split_dict(
        train_file_tuple : Dict[str, List[Tuple[str, str]]],
        test_file_tuple : Dict[str, List[Tuple[str, str]]],
        output_dir: Union[str, Path],
        filter_file_keys: Optional[List[str]] = None,
        overwrite: bool = False,
    ):
    """
    Copy image and mask files into train and test subdirectories.

    Args:
        train_file_tuple: Dictionary containing train file tuples {"images": [...], "masks": [...]}
        test_file_tuple: Dictionary containing test file tuples {"images": [...], "masks": [...]}
        output_dir: Directory to copy the files to
        filter_file_keys: Optional list of file type keys to copy (copies all if None)
        overwrite: If False (default), skip files that already exist.
    """
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dir = output_dir / 'train'
    test_dir = output_dir / 'test'
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    if filter_file_keys:
        train_file_tuple = {k: v for k, v in train_file_tuple.items() if k in filter_file_keys}
        test_file_tuple = {k: v for k, v in test_file_tuple.items() if k in filter_file_keys}

    # TODO : Handle masks separately into mask subdirs
    for k, v in test_file_tuple.items():
        for src, dst in v:
            copy_file(src, test_dir / dst, overwrite=overwrite)

    for k, v in train_file_tuple.items():
        for src, dst in v:
            copy_file(src, train_dir / dst, overwrite=overwrite)


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
        elif args.type == "mask":
            new_name = handler.rename_mask(args.file)
        else:
            raise ValueError(f"Unknown file type: {args.type}")
        print(f"Original: {args.file}")
        print(f"Renamed:  {new_name}")
