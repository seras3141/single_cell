"""
Concrete implementation of data management for single-cell feature visualization.

This module provides the DataManager class that handles loading, parsing, and
validation of feature data from CSV files.
"""

import re
import logging
from pathlib import Path
from typing import List, Optional, Union
import pandas as pd
import numpy as np

from .visualization_base import BaseDataManager, DataLoadingError


logger = logging.getLogger(__name__)


class FeatureDataManager(BaseDataManager):
    """
    Handles loading and management of single-cell feature data.
    
    This class provides:
    - CSV file loading with validation
    - Metadata parsing from filenames
    - Feature column identification
    - Data integrity checks
    """
    
    # Define feature categories for better organization
    MORPHOLOGY_FEATURES = [
        'area', 'perimeter', 'elongation', 'compactness', 'circularity',
        'feret_diameter', 'radius_of_gyration', 'major_axis', 'minor_axis'
    ]
    
    INTENSITY_FEATURES = [
        'mean_intensity', 'std_intensity', 'cv_intensity', 'total_intensity'
    ]
    
    SPATIAL_FEATURES = [
        'centroid_x', 'centroid_y', 'center_of_mass_x', 'center_of_mass_y', 
        'mass_displacement'
    ]
    
    TEXTURE_FEATURES = [
        'gabor_mean', 'gabor_std', 'skewness', 'kurtosis', 'entropy',
        # 'contrast', 'correlation', 'energy', 'homogeneity', 'dissimilarity',
        # 'asm', 'angular_second_moment'
    ]
    
    # Columns to exclude from feature analysis
    METADATA_COLUMNS = [
        'instance_id', 'filename', 'image_path', 'image_filename', 'mask_filename',
        'sample_id', 'z_stack', 'sample_z_id', 'processing_timestamp',
        'feature_extraction_version', 'dataset_name'
    ]
    
    def __init__(self, filename_pattern: Optional[str] = None):
        """
        Initialize the data manager.
        
        Args:
            filename_pattern: Regex pattern for parsing filenames.
        """
        self.filename_pattern = filename_pattern or r'(p\d+_[A-Z]\d+)_z(\d+)_(BF|masks)\.tif'
        logger.info(f"FeatureDataManager initialized with pattern: {self.filename_pattern}")
    
    def load_data(self, source: Union[str, Path]) -> pd.DataFrame:
        """
        Load data from a CSV file or folder with CSV files.
        
        Args:
            source: Path to the CSV file or directory containing CSV files.

        Returns:
            DataFrame with loaded and parsed data
            
        Raises:
            DataLoadingError: If loading fails
        """
        source_path = Path(source)
        
        if not source_path.exists():
            raise DataLoadingError(f"File not found: {source_path}")
        
        # If source is csv
        if source_path.suffix.lower() == '.csv':
            return self._load_data_from_csv(source_path)

        elif source_path.is_dir():
            return self._load_data_from_directory(source_path)
        
        else:
            raise DataLoadingError(f"Unsupported file type: {source_path.suffix}")

    def _load_data_from_csv(self, csv_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load data specifically from a CSV file.
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            DataFrame with loaded data
        """
        source_path = Path(csv_path)

        try:
            logger.info(f"Loading data from: {source_path}")
            df = pd.read_csv(source_path)
            
            if df.empty:
                raise DataLoadingError(f"CSV file is empty: {source_path}")
            
            logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
            
            # Parse metadata if filename column exists
            df = self._parse_metadata_if_available(df)
            
            return df
            
        except pd.errors.EmptyDataError:
            raise DataLoadingError(f"Invalid or empty CSV file: {source_path}")
        except pd.errors.ParserError as e:
            raise DataLoadingError(f"CSV parsing error in {source_path}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error loading {source_path}: {e}")
            raise DataLoadingError(f"Failed to load CSV: {e}") from e
    
    def _load_data_from_directory(self, dir_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load and concatenate data from all CSV files in a directory.
        
        Args:
            dir_path: Path to the directory
            
        Returns:
            DataFrame with concatenated data from all CSVs
        """
        dir_path = Path(dir_path)
        
        if not dir_path.is_dir():
            raise DataLoadingError(f"Provided path is not a directory: {dir_path}")
        
        csv_files = list(dir_path.glob('*.csv'))
        
        if not csv_files:
            raise DataLoadingError(f"No CSV files found in directory: {dir_path}")
        
        logger.info(f"Found {len(csv_files)} CSV files in {dir_path}")
        
        data_frames = []
        for csv_file in csv_files:
            try:
                df = self._load_data_from_csv(csv_file)

                if df is None or df.empty:
                    logger.warning(f"No valid data found in file: {csv_file}")
                    continue
                else:
                    data_frames.append(df)

            except DataLoadingError as e:
                logger.warning(f"Skipping file due to error: {e}")
        
        if not data_frames:
            raise DataLoadingError(f"No valid CSV files could be loaded from: {dir_path}")
        
        combined_df = pd.concat(data_frames, ignore_index=True)
        logger.info(f"Combined data has {len(combined_df)} rows, {len(combined_df.columns)} columns")
        
        return combined_df
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate data integrity and structure.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        try:
            # Check for minimum required structure
            if df.empty:
                logger.warning("DataFrame is empty")
                return False
            
            # Check for feature columns
            feature_cols = self.get_feature_columns(df)
            if not feature_cols:
                logger.warning("No feature columns found in data")
                return False
            
            # Check for excessive NaN values
            nan_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
            if nan_ratio > 0.5:
                logger.warning(f"High NaN ratio in data: {nan_ratio:.2%}")
                return False
            
            # Check for infinite values in numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            inf_count = np.isinf(df[numeric_cols]).sum().sum()
            if inf_count > 0:
                logger.warning(f"Found {inf_count} infinite values in numeric columns")
                # Still valid, but log warning
            
            logger.info("Data validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Data validation error: {e}")
            return False
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Identify feature columns in the DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            List of feature column names
        """
        all_features = (
            self.MORPHOLOGY_FEATURES + 
            self.INTENSITY_FEATURES + 
            self.SPATIAL_FEATURES + 
            self.TEXTURE_FEATURES
        )
        
        # Get columns that match our feature patterns and are numeric
        feature_cols = []
        for col in df.columns:
            # Skip metadata columns
            if col in self.METADATA_COLUMNS:
                continue
            
            # Include if it's in our known feature list
            if col in all_features:
                feature_cols.append(col)
                continue
            
            # Include if it matches common patterns and is numeric
            if (df[col].dtype in ['int64', 'float64'] and 
                any(pattern in col.lower() for pattern in 
                    ['intensity', 'area', 'texture', 'gabor', 'morph'])):
                feature_cols.append(col)
        
        logger.info(f"Identified {len(feature_cols)} feature columns")
        return feature_cols
    
    def _parse_metadata_if_available(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse metadata from filenames if filename column exists.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with parsed metadata columns
        """
        # Find filename column
        filename_col = None
        for col in ['filename', 'image_filename', 'image_path']:
            if col in df.columns:
                filename_col = col
                break
        
        if not filename_col:
            logger.info("No filename column found, skipping metadata parsing")
            return df
        
        logger.info(f"Parsing metadata from column: {filename_col}")
        return self._parse_sample_metadata(df, filename_col)
    
    def _parse_sample_metadata(self, df: pd.DataFrame, filename_col: str) -> pd.DataFrame:
        """
        Parse sample metadata from filenames using regex.
        Create sample_id and z_stack columns if they do not already exist.
        
        Args:
            df: Input DataFrame
            filename_col: Name of the column containing filenames
            
        Returns:
            DataFrame with additional metadata columns
        """
        df = df.copy()
        
        try:
            # Check if sample_id and z_stack already exist in the data
            has_existing_sample_id = 'sample_id' in df.columns
            has_existing_z_stack = 'z_stack' in df.columns
            
            if has_existing_sample_id and has_existing_z_stack:
                # Use existing metadata, just ensure z_stack is integer
                df['z_stack'] = pd.to_numeric(df['z_stack'], errors='coerce').fillna(1).astype(int)
                logger.info(f"Using existing sample metadata for {df['sample_id'].nunique()} unique samples")
                logger.info(f"Z-stack range: {df['z_stack'].min()} - {df['z_stack'].max()}")
            else:
                if filename_col not in df.columns:
                    raise ValueError(f"Filename column '{filename_col}' not found in DataFrame (csv file)")
                
                # Extract sample information using regex
                pattern_match = df[filename_col].str.extract(self.filename_pattern)

                if pattern_match.shape[1] >= 2:
                    df['sample_id'] = pattern_match[0]
                    df['z_stack'] = pattern_match[1]
                    
                    # Fill missing values with defaults
                    df['sample_id'] = df['sample_id'].fillna('Unknown')
                    df['z_stack'] = df['z_stack'].fillna('1')
                    
                    # Convert z_stack to integer
                    df['z_stack'] = pd.to_numeric(df['z_stack'], errors='coerce').fillna(1).astype(int)
                    
                    logger.info(f"Parsed metadata for {df['sample_id'].nunique()} unique samples")
                    logger.info(f"Z-stack range: {df['z_stack'].min()} - {df['z_stack'].max()}")
                else:
                    logger.warning(f"Filename pattern did not match expected format: {self.filename_pattern}")
                    # If pattern doesn't match and no existing metadata, use defaults
                    if not has_existing_sample_id:
                        df['sample_id'] = 'Unknown'
                    if not has_existing_z_stack:
                        df['z_stack'] = 1
            
            # Create combined identifier
            df['sample_z_id'] = (
                df['sample_id'].astype(str) + '_z' + 
                df['z_stack'].astype(str)
            )
        
        except Exception as e:
            logger.error(f"Error parsing metadata: {e}")
            # Continue without metadata parsing
        
        return df
    
    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """
        Generate a summary of the loaded data.
        
        Args:
            df: DataFrame to summarize
            
        Returns:
            Dictionary with data summary statistics
        """
        feature_cols = self.get_feature_columns(df)
        
        summary = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'feature_columns': len(feature_cols),
            'metadata_columns': len([col for col in df.columns if col in self.METADATA_COLUMNS]),
            'missing_values': int(df.isnull().sum().sum()),
            'missing_percentage': float(df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        }
        
        # Add sample-specific info if available
        if 'sample_id' in df.columns:
            summary.update({
                'unique_samples': df['sample_id'].nunique(),
                'samples': sorted(df['sample_id'].unique())
            })
        
        if 'z_stack' in df.columns:
            summary.update({
                'z_stack_range': f"{df['z_stack'].min()}-{df['z_stack'].max()}",
                'unique_z_stacks': df['z_stack'].nunique()
            })
        
        return summary


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize data manager
    data_manager = FeatureDataManager()
    
    # Example usage (would need actual CSV file)
    # df = data_manager.load_data("sample_features.csv")
    # is_valid = data_manager.validate_data(df)
    # feature_cols = data_manager.get_feature_columns(df)
    # summary = data_manager.get_data_summary(df)
    
    print("FeatureDataManager example completed")
