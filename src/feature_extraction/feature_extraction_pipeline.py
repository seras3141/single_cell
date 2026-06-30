
from typing import List, Tuple, Dict, Any, Optional
import logging
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2

from src.feature_extraction.feature_extractor_incarta import extract_all_instance_features
try:
    from src.feature_extraction.feature_extractor_pyradiomics import get_radiomics_features
except ImportError:
    get_radiomics_features = None
from src.feature_extraction.feature_extractor_regionprops import get_region_properties

class FeatureExtractionPipeline:
    """Pipeline for extracting features from datasets of segmented cells."""

    def __init__(self, config: Dict[str, Any] = {}, 
                 method: str | None = None, 
                 output_dir: str | None = None,
                 log_config: Dict[str, Any] = {},
                 ):
        """Initialize feature extraction pipeline.
        
        Args:
            config: Feature configuration dictionary
            method: Feature extraction method (overrides config if provided)
            output_dir: Output directory (overrides config if provided)
            log_config: Logging configuration dictionary
        """
        
        self.feature_config = config
        
        # Extract configuration sections
        # self.paths_config = feature_config.get('paths', {})
        self.method = method or self.feature_config.get('method', 'incarta')
        self.output_config = self.feature_config.get('output', {})
        self.processing_config = self.feature_config.get('processing', {})

        # Validate method
        if self.method not in ['incarta', 'regionprops', 'pyradiomics']:
            raise ValueError(f"Unsupported feature extraction method: {self.method}")

        # Setup output directory first
        self._setup_output(output_dir, self.output_config)
        
        # Now setup logging (which needs output_dir)
        self.log_config = log_config
        self.setup_logging()

        # Initialize counters and results
        self.skipped_files = 0
        self.error_files = []
        self.all_features = []

    def _setup_output(self, output_dir: str |  None = None, output_config: Dict[str, Any] | None = None):
        """Setup output directory structure."""
        if output_dir:
            self.output_dir = Path(output_dir)
        elif output_config:
            self.output_dir = output_config.get('output_dir', 'output/features')
        else:
            raise NotImplementedError("output_dir or output_config must be set")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'FeatureExtractionPipeline':
        """Create pipeline instance from configuration dictionary."""

        feature_config = config.get('feature_extraction', {})
        log_config = config.get('logging', {})
        output_dir = config.get('paths', {}).get('output_dir')

        return cls(config=feature_config, output_dir=output_dir, log_config=log_config)
    
    def setup_logging(self):
        """Setup logging configuration."""
        from src.utils.logging_utils import setup_logging

        log_level = self.log_config.get('level', 'INFO')
        log_file = self.log_config.get('filename', 'feature_extraction.log')
        log_file = self.output_dir / log_file

        setup_logging(log_level, log_file) # type: ignore
        self.logger = logging.getLogger(__name__)


    def find_image_mask_pairs(
            self, 
            image_dir: Path, 
            mask_dir: Path, 
            image_patterns: List[str] | None = None, 
            mask_patterns: List[str] | None = None
        ) -> List[Tuple[Path, Path]]:
        """Find matching image and mask file pairs.
        
        Args:
            image_dir: Directory containing images
            mask_dir: Directory containing masks
            image_patterns: List of glob patterns for images
            mask_patterns: List of glob patterns for masks

        Returns:
            List of (image_path, mask_path) tuples
        """

        pairs = []
        
        # Get file patterns
        image_patterns = image_patterns or ['*_BF.tif']
        mask_patterns = mask_patterns or ['*_Cells.tif']

        self.logger.debug(f"Searching for image patterns: {image_patterns}")
        self.logger.debug(f"Searching for mask patterns: {mask_patterns}")

        # Find all image and mask files
        image_files = []
        mask_files = []
        
        for pattern in image_patterns:
            image_files.extend(image_dir.rglob(pattern))

        for pattern in mask_patterns:
            mask_files.extend(mask_dir.rglob(pattern))

        self.logger.info(f"Found {len(image_files)} potential image files and {len(mask_files)} mask files")

        # Match files based on configuration
        pairs = self.match_files(image_files, mask_files, mask_patterns=mask_patterns)

        self.logger.info(f"Successfully paired {len(pairs)} image-mask pairs")
        return pairs
    
    def find_image_given_mask(
            self, 
            mask_path: Path, 
            image_files: List[Path],
            mask_patterns: List[str] | None = None
        ) -> Optional[Path]:
        """Find corresponding image file for a given mask.
        
        Args:
            mask_path: Path to the mask file
            image_files: List of available image files
            
        Returns:
            Path to the matching image file, or None if not found
        """

        import re
        # Convert mask path to str
        mask_name = mask_path.name
        mask_patterns = mask_patterns or ['*_Cells.tif']

        for mask_pattern in mask_patterns:
            mask_pattern = mask_pattern.replace('*', '(.*)')
            match = re.match(mask_pattern, mask_name)
            if match:
                mask_prefix = match.group(1)
                self.logger.debug(f"Mask prefix extracted: {mask_prefix}")

                for image in image_files:
                    image_stem = image.stem
                    if image_stem.startswith(mask_prefix):
                        self.logger.debug(f"Matched {image.name} with {mask_path.name} based on prefix {mask_prefix}")
                        return image

        self.logger.warning(f"No matching image found for mask: {mask_path.name}")
        return None

    def match_files(self, image_files: List[Path], mask_files: List[Path], mask_patterns: List[str] | None = None) -> List[Tuple[Path, Path]]:
        """Match image files with corresponding mask files based on patterns.
        
        Args:
            image_files: List of image file paths
            mask_files: List of mask file paths
            mask_patterns: List of glob patterns for masks

        Returns:
            List of matched (image_path, mask_path) tuples
        """
        pairs = []

        for mask in mask_files:
            # Find corresponding image for each mask
            image = self.find_image_given_mask(mask, image_files, mask_patterns=mask_patterns)

            if image:
                pairs.append((image, mask))
        
        return pairs

    # Why use custom function and preprocessing here instead of inside extract_all_instance_features?
    def load_image_and_mask(self, image_path: Path, mask_path: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Load image and mask files.
        
        Args:
            image_path: Path to image file
            mask_path: Path to mask file
            
        Returns:
            Tuple of (image, mask) arrays, or (None, None) if loading fails
        """

        try:
            # Load image
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                self.logger.error(f"Failed to load image: {image_path}")
                return None, None
            
            # Load mask
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if mask is None:
                self.logger.error(f"Failed to load mask: {mask_path}")
                return None, None
            
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            
            # Validate dimensions match
            if image.shape != mask.shape:
                self.logger.error(f"Image and mask dimensions don't match: {image.shape} vs {mask.shape}")
                return None, None
            
            # Apply preprocessing if configured
            preprocessing = self.feature_config.get('preprocessing', {})
            if preprocessing.get('normalize_intensity', False):
                image = image.astype(np.float32) / 255.0
            
            clip_percentiles = preprocessing.get('clip_percentiles')
            if clip_percentiles:
                lower, upper = clip_percentiles
                p_low, p_high = np.percentile(image, [lower, upper]) # type: ignore
                image = np.clip(image, p_low, p_high)
            
            return image, mask
            
        except Exception as e:
            self.logger.error(f"Error loading {image_path} and {mask_path}: {str(e)}")
            return None, None
    
    '''
    def validate_mask(self, mask: np.ndarray, image_path: Path) -> bool:
        """Validate mask meets quality criteria.
        
        Args:
            mask: Mask array
            image_path: Path to corresponding image (for logging)
            
        Returns:
            True if mask is valid, False otherwise
        """
        validation_config = self.processing_config.get('validation', {})
        
        # Count instances
        unique_labels = np.unique(mask)
        n_instances = len(unique_labels) - 1 if 0 in unique_labels else len(unique_labels)
        
        min_instances = validation_config.get('min_instances_per_image', 1)
        max_instances = validation_config.get('max_instances_per_image', 10000)
        
        if n_instances < min_instances:
            logger.warning(f"Skipping {image_path.name}: too few instances ({n_instances})")
            return False
        
        if n_instances > max_instances:
            logger.warning(f"Skipping {image_path.name}: too many instances ({n_instances})")
            return False
        
        return True
    '''
    
    def extract_features_from_path(
            self, 
            image_path: Path | str, 
            mask_path: Path | str, 
        ) -> Optional[pd.DataFrame]:
        """Extract features from a single image-mask pair.
        
        Args:
            image_path: Path to image file
            mask_path: Path to mask file
            
        Returns:
            DataFrame with extracted features, or None if extraction fails
        """

        image_path = Path(image_path)
        mask_path = Path(mask_path)

        if not image_path.exists():
            self.logger.error(f"Image file does not exist: {image_path}")
            self.error_files.append((str(image_path), "File not found"))
            return None
        if not mask_path.exists():
            self.logger.error(f"Mask file does not exist: {mask_path}")
            self.error_files.append((str(mask_path), "File not found"))
            return None

        try:
            # Load image and mask
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if image is None or mask is None:
                self.logger.error(f"Failed to load image: {image_path} or mask: {mask_path}")
                return None
                        
            # Extract features using the main function
            n_jobs = self.feature_config.get('n_jobs', -1)

            if self.method == 'incarta':
                features_df = extract_all_instance_features(mask, image, n_jobs=n_jobs)
            elif self.method == 'regionprops':
                features_df = get_region_properties(mask, intensity_image=image)
            elif self.method == 'pyradiomics':
                if get_radiomics_features is None:
                    raise RuntimeError(
                        "pyradiomics is not installed. Install it with 'pip install pyradiomics' to use this method."
                    )
                features_df = get_radiomics_features(image, mask)
            else:
                raise ValueError(f"Unknown feature extraction method: {self.method}")
            
            if features_df.empty:
                self.logger.warning(f"No features extracted from {mask_path.name}")
                return None
            
            # Add metadata if configured
            if self.output_config.get('include_metadata', True):
                features_df['image_filename'] = image_path.name
                features_df['mask_filename'] = mask_path.name
                # features_df['processing_timestamp'] = datetime.now().isoformat()
                # features_df['feature_extraction_version'] = '1.0'
                features_df['dataset_name'] = image_path.parent.name

            self.logger.debug(f"Extracted {len(features_df)} instances from {image_path.name}")
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"Error extracting features from {image_path}: {str(e)}")
            self.error_files.append((str(image_path), str(e)))
            return None
    
    def save_image_features(self, features_df: pd.DataFrame, image_path: Path):
        """Save features for individual image to CSV file.
        
        Args:
            features_df: Features DataFrame
            image_path: Original image path (for naming output file)
        """
        
        # Create output filename
        output_format = self.output_config.get('individual_format', '{image_name}_features.csv')
        output_name = output_format.format(image_name=image_path.stem)
        
        # Create subdirectory if configured
        output_path = self.output_dir
        if self.output_config.get('create_subdirs', True):
            subdir = image_path.parent.name
            output_path = output_path / subdir
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Save file
        output_file = output_path / output_name
        features_df.to_csv(output_file, index=False)
        self.logger.debug(f"Saved individual features to {output_file}")

    def process_batch(
            self, 
            image_dir: Path | str, 
            mask_dir: Path | str,
            image_patterns: List[str] | None = None, 
            mask_patterns: List[str] | None = None
        ) -> pd.DataFrame:
        """Process single dir containing images and masks, and extract features.
        
        Args:
            image_dir: Directory containing images (if None, uses config)
            mask_dir: Directory containing masks (if None, uses config)
            image_patterns: List of glob patterns for images
            mask_patterns: List of glob patterns for masks
            
        Returns:
            Combined DataFrame with all features
        """

        image_dir = Path(image_dir)
        mask_dir = Path(mask_dir)

        self.logger.info(f"Processing dataset: {mask_dir} with images from {image_dir}")

        # Find image-mask pairs
        pairs = self.find_image_mask_pairs(image_dir, mask_dir, image_patterns=image_patterns, mask_patterns=mask_patterns)
        if not pairs:
            self.logger.error(f"No valid image-mask pairs found in {image_dir}")
            return pd.DataFrame()
        
        # Process pairs
        processed_files = 0
        all_features = []
            
        for image_path, mask_path in tqdm(pairs, desc="Processing files"):
            features_df = self.extract_features_from_path(image_path, mask_path)
            
            if features_df is not None:
                # batch_features.append(features_df)
                all_features.append(features_df)
                processed_files += 1
                
                # Save individual file if configured
                if self.output_config.get('save_individual_files', True):
                    self.save_image_features(features_df, image_path)
            
        # Combine all features
        if all_features:
            combined_df = pd.concat(all_features, ignore_index=True)
            self.logger.info(f"Total features extracted: {len(combined_df)} instances from {processed_files} images")
        else:
            combined_df = pd.DataFrame()
            self.logger.warning("No features extracted from any files")

        return combined_df
    
    def save_combined_features(self, features_df: pd.DataFrame):
        """Save combined features to CSV file.

        Args:
            features_df: Combined features DataFrame
        """
        if not self.output_config.get('save_combined_file', True):
            return
        
        if features_df.empty:
            self.logger.warning("No features to save")
            return
        
        # Save combined file
        combined_filename = self.output_config.get('combined_filename', 'all_features.csv')
        output_file = self.output_dir / combined_filename
        
        features_df.to_csv(output_file, index=False)
        self.logger.info(f"Saved combined features to {output_file}")

        # Save summary statistics
        summary_file = self.output_dir / 'feature_extraction_summary.txt'
        with open(summary_file, 'w') as f:
            f.write(f"Feature Extraction Summary\n")
            f.write(f"========================\n\n")
            f.write(f"Processing completed: {datetime.now()}\n")
            f.write(f"Total files processed: {len(features_df)}\n")
            f.write(f"Files skipped: {self.skipped_files}\n")
            f.write(f"Files with errors: {len(self.error_files)}\n")
            f.write(f"Total instances: {len(features_df)}\n")
            f.write(f"Total features per instance: {len(features_df.columns)}\n\n")
            
            if self.error_files:
                f.write("Error Files:\n")
                for filepath, error in self.error_files:
                    f.write(f"  {filepath}: {error}\n")
            
            f.write(f"\nFeature Columns:\n")
            for col in features_df.columns:
                f.write(f"  {col}\n")

        self.logger.info(f"Saved processing summary to {summary_file}")

    def run(self, image_dirs: List[Path] = [], mask_dirs: List[Path] = []) -> pd.DataFrame:
        """Run the complete feature extraction pipeline.
        
        Args:
            image_dirs: List of image directories (if None, uses config)
            mask_dirs: List of mask directories (if None, uses config)

        Returns:
            Combined features DataFrame
        """
        start_time = time.time()
        self.logger.info("Starting feature extraction pipeline")

        all_datasets_features = []

        if image_dirs and mask_dirs and len(image_dirs) != len(mask_dirs):
            self.logger.error("Number of image directories must match number of mask directories")
            return pd.DataFrame()

        # Process each directory
        for image_dir, mask_dir in zip(image_dirs, mask_dirs):
            self.logger.info(f"Processing directory: {image_dir}")
            features_df = self.process_batch(image_dir, mask_dir)

            if not features_df.empty:
                all_datasets_features.append(features_df)
        
        # Combine all datasets
        if all_datasets_features:
            final_features = pd.concat(all_datasets_features, ignore_index=True)
        else:
            final_features = pd.DataFrame()
        
        # Save results
        self.save_combined_features(final_features)
        
        # Log completion
        elapsed_time = time.time() - start_time
        self.logger.info(f"Feature extraction completed in {elapsed_time:.2f} seconds")
        self.logger.info(f"Final results: {len(final_features)} instances from xxx images")

        return final_features

