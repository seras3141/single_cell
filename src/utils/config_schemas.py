"""
Structured configuration schemas for single cell analysis pipeline.

This module defines all configuration schemas using dataclasses that work
seamlessly with OmegaConf for type-safe configuration management.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path


# =============================================================================
# Core Configuration Schemas
# =============================================================================

@dataclass
class PathsConfig:
    """Data paths configuration."""
    # Paths for individual scripts
    input_dir: str = "data/input"  # Base directory with datasets
    output_dir: str = "data/output"  # Where to store the results
    # Paths for postprocessing
    mask_dir: str = "data/segmentation"  # Directory for segmentation results
    blur_dir: str = "data/blur_heatmaps"  # Directory for blur heatmaps
    image_dir: str = "data/images"  # Directory for input images    
    # Additional paths (For entire pipeline)
    results: str = "results/"
    models: str = "models/"
    temp: str = "temp/"


@dataclass
class PreprocessingConfig:
    """File processing parameters."""
    test_size: float = 0.2
    random_state: int = 42
    split_by_group: bool = True
    split_folder: str = "split_data"
    out_3d_folder: str = "3d_images"  # Directory for 3D images

    raw_data_patterns: Dict[str, str] = field(default_factory=lambda: {
        "brightfield": "t1_*_w1_*.tif",
        "masks": "Cells_*.tif",
        "nuclei": "Nuclei_*.tif"
    })
    
    output_suffixes: Dict[str, str] = field(default_factory=lambda: {
        "brightfield": "*_BF*.tif",
        "masks": "*_Cells*.tif", 
        "nuclei": "*_Nuclei*.tif"
    })


@dataclass
class QualityConfig:
    """Image quality assessment configuration."""
    blur_detection: Dict[str, Any] = field(default_factory=lambda: {
        "patch_size": [32, 32],
        "stride_size": [8, 8],
    })


# =============================================================================
# Segmentation Configuration
# =============================================================================

@dataclass
class CellposeConfig:
    """Cellpose segmentation configuration."""
    model_type: str = "cyto3"
    channels: List[int] = field(default_factory=lambda: [0, 0])
    diameter: Optional[float] = None
    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0
    min_size: int = 30
    normalize: bool = True
    invert: bool = False
    gpu: bool = True

@dataclass
class InferenceConfig:
    """Inference configuration for segmentation."""
    dataset_name: str = "test" # Name of the dataset (subfolder) for inference
    results_folder: str = "results"  # Subfolder to save inference results
    file_pattern: str = "*_BF.tif"  # Glob pattern for selecting image files
    process_z_stacks: bool = False  # Whether to process Z-stacks
    save_overlays: bool = True  # Whether to save overlay images
    save_metadata: bool = True  # Whether to save JSON metadata for predictions

@dataclass
class SegmentationConfig:
    """Segmentation configuration."""
    cellpose: CellposeConfig = field(default_factory=CellposeConfig)
    inference: InferenceConfig = field(default_factory=lambda: InferenceConfig())

# =============================================================================
# Training Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """Training configuration."""
    learning_rate: float = 0.1
    weight_decay: float = 1e-4
    n_epochs: int = 100
    batch_size: int = 8
    min_train_masks: int = 5
    SGD: bool = True
    channels: List[int] = field(default_factory=lambda: [0, 0])
    normalize: bool = True


# =============================================================================
# Postprocessing Configuration (from existing dataclasses)
# =============================================================================

@dataclass
class FilterConfig:
    """Configuration for blur-based filtering."""
    # Blur measurement parameters
    patch_size: int = 32
    stride_size: int = 8
    normalize_blur: bool = True
    
    # Filtering parameters
    blur_threshold: float = 0.5
    invert_threshold: bool = False
    
    # Quality assessment
    min_region_overlap: float = 0.5
    
    # Cache settings
    cache_blur_maps: bool = True
    blur_map_suffix: str = "_blur_heatmap"


@dataclass
class TrackingConfig:
    """Configuration for 3D cell tracking."""
    # Tracking parameters
    search_range: float = 5.0
    memory: int = 1
    min_track_length: int = 3
    
    # Region filtering parameters
    # TODO : Why is size filtering not in FilterConfig?
    min_area: int = 10
    max_area: int = 5000
    
    # Quality assessment
    area_percentiles: Tuple[float, float] = (0.1, 99.9)
    
    # Output options
    save_intermediate: bool = False
    output_dtype: str = "int32"


@dataclass
class PostprocessingConfig:
    """Postprocessing pipeline configuration."""
    tracking_config: TrackingConfig = field(default_factory=TrackingConfig)
    filter_config: FilterConfig = field(default_factory=FilterConfig)
    enable_blur_filtering: bool = True
    filter_before_tracking: bool = True
    save_intermediate_results: bool = False
    mask_pattern: str = "*_masks_3d.tif"
    image_pattern: str = "*_BF_3d.tif"
    blur_heatmap_suffix: str = "_blur_heatmap"
    output_suffix: str = "_tracked"
    overwrite_existing: bool = False
    convert_to_2d: bool = True  # Convert final output to 2D if required


# =============================================================================
# Feature Extraction Configuration
# =============================================================================

@dataclass
class RadiomicsConfig:
    """Radiomics feature extraction configuration."""
    binWidth: int = 25
    interpolator: str = "sitkLinear"
    resampledPixelSpacing: Optional[List[float]] = None
    padDistance: int = 10


@dataclass
class MorphologyConfig:
    """Morphology feature extraction configuration."""
    extract_shape: bool = True
    extract_firstorder: bool = True
    extract_glcm: bool = True
    extract_glrlm: bool = False
    extract_glszm: bool = False


@dataclass
class FeaturesConfig:
    """Feature extraction configuration."""
    radiomics: RadiomicsConfig = field(default_factory=RadiomicsConfig)
    morphology: MorphologyConfig = field(default_factory=MorphologyConfig)

@dataclass
class FeatureExtractionConfig:
    """Feature extraction configuration."""
    n_jobs: int = -1  # Use all available cores
    batch_size: int = 25  # Batch size for parallel processing
    
    features: Dict[str, bool] = field(default_factory=lambda: {
        "morphology": True,
        "intensity": True,
        "spatial": True,
        "texture": True
    })
    
    file_patterns: Dict[str, List[str]] = field(default_factory=lambda: {
        "images": ["*_BF.tif"],
        "masks": ["*_Cells.tif"]
    })
    
    preprocessing: Dict[str, Any] = field(default_factory=lambda: {
        "normalize_intensity": True,
        "clip_percentiles": [1, 99]
    })
    output: Dict[str, Any] = field(default_factory=lambda: {
        "folder_name": "features",  # Subdirectory for features
        "save_individual_files": True,
        "save_combined_file": True,
        "include_metadata": True,
        "individual_format": "{image_name}_features.csv",
        "combined_filename": "all_features.csv",
        "create_subdirs": True,
    })


# =============================================================================
# Visualization Configuration
# =============================================================================

@dataclass
class NapariConfig:
    """Napari viewer configuration."""
    gamma: float = 1.0
    contrast_limits: List[int] = field(default_factory=lambda: [0, 65535])


@dataclass
class VisualizationConfig:
    """Visualization configuration."""
    figure_size: List[int] = field(default_factory=lambda: [12, 8])
    dpi: int = 300
    colormap: str = "viridis"
    napari: NapariConfig = field(default_factory=NapariConfig)


# =============================================================================
# Output Configuration
# =============================================================================

@dataclass
class OutputConfig:
    """Output configuration."""
    save_masks: bool = True
    save_outlines: bool = True
    save_flows: bool = False
    save_features: bool = True


# =============================================================================
# Logging Configuration
# =============================================================================

@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    filename: str = "logs/pipeline.log"


# =============================================================================
# Main Pipeline Configuration
# =============================================================================

@dataclass
class PipelineConfig:
    """Main pipeline configuration schema."""
    paths: PathsConfig = field(default_factory=PathsConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    filtering: FilterConfig = field(default_factory=FilterConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    postprocessing: PostprocessingConfig = field(default_factory=PostprocessingConfig)
    feature_extraction: FeatureExtractionConfig = field(default_factory=FeatureExtractionConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


# =============================================================================
# Validation Functions
# =============================================================================

def validate_pipeline_config(config: PipelineConfig) -> None:
    """Validate pipeline configuration."""
    # Training validation
    if config.training.learning_rate <= 0:
        raise ValueError("Training learning rate must be positive")
    
    if config.training.batch_size <= 0:
        raise ValueError("Training batch size must be positive")
    
    if config.training.n_epochs <= 0:
        raise ValueError("Training epochs must be positive")
    
    # Segmentation validation
    if not 0 <= config.segmentation.cellpose.flow_threshold <= 1:
        raise ValueError("Cellpose flow threshold must be between 0 and 1")
    
    if not 0 <= config.segmentation.cellpose.cellprob_threshold <= 1:
        raise ValueError("Cellpose cellprob threshold must be between 0 and 1")
    
    # Tracking validation
    if config.tracking.search_range <= 0:
        raise ValueError("Tracking search range must be positive")
    
    if config.tracking.min_area >= config.tracking.max_area:
        raise ValueError("Tracking min_area must be less than max_area")
    
    # Filtering validation
    if not 0 <= config.filtering.blur_threshold <= 1:
        raise ValueError("Blur threshold should typically be between 0 and 1")
