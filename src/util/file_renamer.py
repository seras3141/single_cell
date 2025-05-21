from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict
import re
from glob import glob


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

        self.image_files = sorted(glob(self.image_path, recursive=True))
        self.mask_files = sorted(glob(self.mask_path, recursive=True))


class AbstractFileRenamer(ABC):
    """Abstract base class for file renaming operations."""

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
    def rename_image(self, filename: str) -> str:
        """Rename image file to standardized format.
        
        Args:
            filename: Original filename
            
        Returns:
            Standardized filename
        """
        pass

    @abstractmethod
    def rename_mask(self, filename: str) -> str:
        """Rename mask file to standardized format.
        
        Args:
            filename: Original filename
            
        Returns:
            Standardized filename
        """
        pass

class DefaultFileRenamer(AbstractFileRenamer):
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

    def __init__(self, patterns: Dict[str, FilePattern] = None):
        """Initialize with optional custom patterns.
        
        Args:
            patterns: Dictionary of file type to FilePattern mappings
        """
        self.patterns = patterns or self.DEFAULT_PATTERNS

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
            values['z'] = str(int(values['z']) + 1)
        
        return pattern.output_format.format(**values)

    def rename_image(self, filename: str) -> str:
        return self.rename_file(filename, 'image')

    def rename_mask(self, filename: str) -> str:
        return self.rename_file(filename, 'mask')



class BF_IF_FileRenamer(DefaultFileRenamer):
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

    def __init__(self, patterns: Dict[str, FilePattern] = None):
        """Initialize with optional custom patterns."""
        self.patterns = patterns or self.DEFAULT_PATTERNS

    def rename_file(self, filename: str, file_type: str) -> str:
        if file_type not in self.patterns:
            raise ValueError(f"Unknown file type: {file_type}")

        # Extract plate number from filepath using regex
        plate_match = re.search(r'Plate\s*(\d+)', filename)
        if not plate_match:
            raise ValueError(f"Could not extract plate number from filepath: {filename}")
        plate_number = plate_match.group(1)

        filename = Path(filename).name
        pattern = self.patterns[file_type]
        match = re.search(pattern.pattern, filename)
        
        if not match:
            return filename

        values = dict(zip(pattern.groups, match.groups()))
        values['plate'] = plate_number

        
        if file_type == 'mask':
            values['row'] = chr(ord('A') + int(values['row_num']) - 1)
            values['col'] = f"{int(values['col']):02d}"
            values['z'] = str(int(values['z']) + 1)
        else:
            assert values['well'] == '1', "Multiple wells not supported"
        
        return pattern.output_format.format(**values)

    def rename_image(self, filename: str) -> str:
        return self.rename_file(filename, 'image')

    def rename_mask(self, filename: str) -> str:
        return self.rename_file(filename, 'mask')
    
    def get_group(self, filename: str) -> str:
        """Extract position grouping from output format."""
        parts = filename.split('_')
        return parts[0] + "_" + parts[1]




class BF_IF_FileRenamer_3D(DefaultFileRenamer):
    """File renamer for BF+IF experiments that includes plate number from filepath.
    
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

    def __init__(self, patterns: Dict[str, FilePattern] = None):
        """Initialize with optional custom patterns."""
        self.patterns = patterns or self.DEFAULT_PATTERNS

    def rename_file(self, filename: str, file_type: str) -> str:
        if file_type not in self.patterns:
            raise ValueError(f"Unknown file type: {file_type}")

        # Extract plate number from filepath using regex
        plate_match = re.search(r'Plate\s*(\d+)', filename)
        if not plate_match:
            raise ValueError(f"Could not extract plate number from filepath: {filename}")
        plate_number = plate_match.group(1)

        filename = Path(filename).name
        pattern = self.patterns[file_type]
        match = re.search(pattern.pattern, filename)
        
        if not match:
            return filename

        values = dict(zip(pattern.groups, match.groups()))
        values['plate'] = plate_number
                
        return pattern.output_format.format(**values)

    def rename_image(self, filename: str) -> str:
        return self.rename_file(filename, 'image')

    def rename_mask(self, filename: str) -> str:
        return self.rename_file(filename, 'mask')


if __name__ == "__main__":
    # Initialize the renamer without plate number
    renamer = BF_IF_FileRenamer()

    # Example image and mask filenames with full paths
    image_filename = "data/BF+IF Experiments Labeled/Plate 2126 Compressed - Timepoint  2hr/t1_J03_s1_w1_z1.tif"
    mask_filename = "Cells_R1-C1-F1-Z1-T1.tif"  # Add appropriate path for mask file

    # Rename the image and mask
    renamed_image = renamer.rename_image(image_filename)
    renamed_mask = renamer.rename_mask(mask_filename)

    print(f"Original image filename: {image_filename}")
    print(f"Renamed image filename: {renamed_image}")
    print(f"Original mask filename: {mask_filename}")
    print(f"Renamed mask filename: {renamed_mask}")