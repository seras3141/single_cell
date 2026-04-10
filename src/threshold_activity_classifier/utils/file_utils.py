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
from collections import defaultdict
import yaml
import shutil


@dataclass
class FilePattern:
    """Container for file naming patterns."""
    pattern: str
    groups: List[str]
    output_format: Optional[str]


def load_wavelength_config(config_path: Optional[str] = None) -> Dict[int, str]:
    """Load wavelength to channel mappings from YAML config file.
    
    Args:
        config_path: Path to wavelength config YAML file. If None, uses default location.
        
    Returns:
        Dictionary mapping wavelength indices (int) to channel names (str).
        Example: {1: "BF", 2: "mCherry", 3: "AnnexinV"}
        
    Raises:
        FileNotFoundError: If config file not found.
        ValueError: If config format is invalid.
    """
    if config_path is None:
        # Default location relative to this module
        config_file = Path(__file__).parent.parent.parent / "config" / "wavelength_config.yaml"
    else:
        config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Wavelength config file not found: {config_file}")
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML format in {config_file}: {e}")
    
    if not config or 'wavelength_mappings' not in config:
        raise ValueError(f"Config must contain 'wavelength_mappings' key: {config_file}")
    
    mappings = config['wavelength_mappings']
    
    # Convert string keys to integers if needed
    if not isinstance(mappings, dict):
        raise ValueError(f"'wavelength_mappings' must be a dict: {config_file}")
    
    return {int(k): str(v) for k, v in mappings.items()}

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
            'BF': FilePattern(
                pattern=r't(\d+)_([A-Z])(\d+)_s(\d+)_w(1)_z(\d+)',
                groups=['time', 'row', 'col', 'series', 'wavelength', 'z'],
                output_format="{row}{col}_z{z}_BF.tif"
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
        return self.rename_file(filename, 'BF')   # type: ignore

    def rename_mask(self, filename: str) -> str:
        return self.rename_file(filename, 'mask') # type: ignore
        
    def extract_unique_id(self, filename: str) -> str:
        """Extract position grouping from filename."""
        # Try to match standard pattern like "A01_z1_BF.tif"
        match = re.search(r'([A-Z]\d+)', filename)
        if match:
            return match.group(1)
            
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
    Input path contains the plate number, time point, position, well number, and z-stack.
    
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
        self.patterns['BF'].output_format = "p{plate}_{row}{col}_t{time}_z{z}_BF.tif"
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
        return parts[0] + "_" + parts[1] + "_" + parts[2]  # plate_rowcol_time
    
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
    - Dynamic wavelength mappings (w1->BF, w2->mCherry, w3->AnnexinV, etc.)
    - Flexible plate number extraction from filename or directory
    - Runtime parameter overrides for maximum flexibility
    
    Wavelength Configuration:
    - Mappings can be loaded from YAML config file (default or custom path)
    - Mappings can be provided via constructor parameter
    - Constructor parameter overrides YAML config
    
    Plate Number Extraction:
    - Attempts extraction from filename first (e.g., 'p2126_A01_z1.tif')
    - Falls back to directory path (e.g., 'Plate 2126/')
    - Can be explicitly provided to rename_file() to override auto-detection
    
    Example usage:
        # Default: load from config/wavelength_config.yaml
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
        config_path: Optional[str] = None,
        plate_number: Optional[str] = None
    ):
        """Initialize ConfigurableFileHandler with optional configuration.
        
        Args:
            wavelength_mappings: Dictionary mapping wavelength indices to channel names.
                Example: {1: "BF", 2: "mCherry", 3: "AnnexinV"}
                If provided, overrides config_path settings.
            patterns: Custom file patterns. If None, uses inherited defaults.
            config_path: Path to wavelength config YAML file. If None, uses default location.
            plate_number: Default plate number to use if not extracted from filepath.
                Can be overridden in rename_file() call.
        """
        super().__init__(patterns)
        
        # Load wavelength mappings
        if wavelength_mappings is not None:
            # Use provided mappings (highest priority)
            self.wavelength_mappings = wavelength_mappings
        else:
            # Load from YAML config
            try:
                self.wavelength_mappings = load_wavelength_config(config_path)
            except (FileNotFoundError, ValueError):
                # Fall back to default if config not found
                self.wavelength_mappings = {1: "AnnexinV", 2: "mCherry", 3: "BF"}
        
        # Store default plate number
        self._default_plate_number = plate_number
        
        # Update 'image' output format to use dynamic channel names
        # The actual channel name will be substituted during rename_file()
        self.patterns.pop('BF', None)  # Remove BF pattern since it's now handled by 'image' with dynamic mapping
        self.patterns['image'].output_format = "p{plate}_{row}{col}_t{time}_z{z}_{channel_name}.tif"

    def get_channel_name(self, wavelength_index: int) -> str:
        """Get channel name for a wavelength index.
        
        Args:
            wavelength_index: Wavelength index (e.g., 1, 2, 3)
            
        Returns:
            Channel name (e.g., "BF", "mCherry", "AnnexinV")
            
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


    def extract_unique_id(self, filename: str) -> str:
        raise NotImplementedError("StandardFileHandler does not implement extract_group_id.")

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

class BlurFileHandler(AbstractFileHandler):
    """
    Implementation of file renaming for blur maps with configurable patterns.
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
        
    def extract_unique_id(self, filename: str) -> str:
        """Extract position grouping from filename."""
        # Try to match standard pattern like "A01_z1_BF.tif"
        match = re.search(r'([A-Z]\d+)', filename)
        if match:
            return match.group(1)
            
        # Fallback to filename without extension
        return Path(filename).stem.split('_')[0]

    def get_files_by_type(self, directory: str, file_type: str) -> List[str]:
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
        group_id = file_handler.extract_unique_id(out_path)
        groups[group_id].append(filepath)
            
    return dict(groups)

def list_all_files(directory: str, file_handler: AbstractFileHandler) -> Dict[str, List[str]]:
    """List files in a directory using the file handler."""
        # Find all images and masks
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
) -> None:
    """
    Copy a file from src_file to dest_file with metadata preservation.

    Args:
        src_file: Source file to copy
        dest_file: Destination file path
    """

    src = Path(src_file).resolve()
    dst = Path(dest_file)

    # Ensure the parent directory exists
    dst.parent.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        raise FileNotFoundError(f"Source file '{src}' does not exist")

    if dst.exists() or dst.is_symlink():
        dst.unlink()  # Remove existing file

    try:
        dst.symlink_to(src, target_is_directory=False)
    except (OSError, NotImplementedError):
        # Use shutil.copy2 which preserves metadata instead of symlinks
        # This works on all platforms without admin privileges
        shutil.copy2(src, dst)

def copy_without_split_dict(
        file_tuple : Dict[str, List[Tuple[str, str]]],
        output_dir: Union[str, Path],
    ):
    """
    Copy image and mask files without splitting into train/test sets.

    Args:
        file_tuple: Dictionary containing file tuples {file_type: [(src_path, dest_path)]}
        output_dir: Directory to copy the files to
    """
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for k, v in file_tuple.items():
        for file_pair in v:
            src, dst = file_pair
            copy_file(src, output_dir / dst)
            count += 1
    return count

def copy_with_split_dict(
        train_file_tuple : Dict[str, List[Tuple[str, str]]],
        test_file_tuple : Dict[str, List[Tuple[str, str]]],
        output_dir: Union[str, Path],
        filter_file_keys: Optional[List[str]] = None,
    ):
    """
    Copy image and mask files into train and test subdirectories.

    Args:
        train_file_tuple: Dictionary containing train file tuples {"images": [...], "masks": [...]}
        test_file_tuple: Dictionary containing test file tuples {"images": [...], "masks": [...]}
        output_dir: Directory to copy the files to
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
            copy_file(src, test_dir / dst)

    for k, v in train_file_tuple.items():
        for src, dst in v:
            copy_file(src, train_dir / dst)


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
