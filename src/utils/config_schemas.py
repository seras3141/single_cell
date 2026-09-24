"""
Structured configuration schemas for single cell analysis pipeline.

This module defines all configuration schemas using dataclasses that work
seamlessly with OmegaConf for type-safe configuration management.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple


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
    # File handler options
    wavelength_mappings: Any = None  # Per-experiment override. Use EXPERIMENT_WAVELENGTH_MAPPINGS[experiment_name] from file_utils.py. Ew2: {1:"FlipGFP",2:"mCherry",3:"BF"}; HD/SA: {1:"BF",2:"mCherry",3:"FlipGFP"}.
    plate_number: Optional[str] = None  # Default plate number; overrides auto-detection from filepath
    # Z-range filtering for 2D-to-3D combination
    z_min: Optional[int] = 1  # Skip z-indices below this value (z0 is typically the 2D projection)
    z_max: Optional[int] = None  # Skip z-indices above this value (None = no upper limit)
    # Skip stages
    skip_split: bool = False
    skip_3d: bool = False
    skip_blur: bool = False
    overwrite: bool = False


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
    label_format: str = "zarr"  # Segmentation label format: tif, zarr, or hdf5
    overwrite: bool = False

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
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    filtering: FilterConfig = field(default_factory=FilterConfig)
    enable_blur_filtering: bool = True
    filter_before_tracking: bool = True
    save_intermediate_results: bool = False
    mask_pattern: str = "*_mask_3d.tif"
    image_pattern: str = "*_BF_3d.tif"
    blur_heatmap_suffix: str = "_blur_heatmap"
    output_suffix: str = "_tracked"
    overwrite_existing: bool = False
    convert_to_2d: bool = True  # Convert final output to 2D if required


@dataclass
class RepresentativeSliceConfig:
    """Config for the representative-slice-per-cell branch.

    Output dirs are named inference_<filtered|unfiltered>_<chooser>/<model> by the driver
    (filtered = blur on; chooser = area|sharpness), not fixed here.
    """
    # Paths (explicit dirs; default "" -> validated at stage runtime, not globally,
    # since validate_pipeline_config runs for every pipeline invocation)
    input_masks_dir: str = ""   # inference/<model>/masks_3d (raw 3D masks)
    bf_3d_dir: str = ""         # 3d_data (3D BF stacks, for sharpness)
    output_dir: str = ""        # inference_<filter>_<chooser>/<model> (masks + selection.csv)

    # Selection criterion: blur gate, then chooser (see selection_metric)
    sharpness_gate_fraction: float = 0.7  # keep slices with lap_var >= f * cell-max
    # Chooser applied among gated slices: "sharpness" (sharpest slice — the default;
    # Phase 6c z-edge check showed it avoids the oblique edge-slice area inflation, at
    # zero downstream cost per the Ew2-1 sharpness-vs-area run) or "area" (largest
    # cross-section). The other metric breaks ties, then z nearest the centroid.
    selection_metric: str = "sharpness"  # sharpness | area
    min_area: int = 10          # guard: drop debris (also linker lower bound)
    max_area: int = 100000      # generous (Phase 0: default 5000 drops big cells)

    # Sharpness: compute laplacian_variance on the interior, not a zeroed crop
    # (Phase 0: a background-zeroed crop is dominated by the mask edge)
    sharpness_erosion_px: int = 1  # erode mask before lap_var (0 = unmasked bbox)

    # Optional pre-link absolute blur-map cell filter (mirrors the tracked pipeline's
    # filter_before_tracking): drops blurry detections BEFORE linking, so junk/off-focus
    # cells are removed rather than kept by the per-cell relative blur gate. Default OFF
    # (unfiltered = current behavior). Uses the cached blur heatmaps + the same threshold
    # as postprocessing_config.yaml (blur_threshold=0.5, invert_threshold=False).
    enable_blur_filter: bool = False
    blur_heatmap_dir: str = ""   # dir of cached {prefix}_BF_3d_blur_heatmap.tif ("" -> compute)
    blur_threshold: float = 0.5
    blur_invert_threshold: bool = False

    # trackpy z-linker. min_track_length is FORCED to 1 in code, not exposed here
    # (Phase 0: the TrackingConfig default of 3 drops ~58% of cells).
    search_range: float = 5.0
    memory: int = 1

    # Data convention (Phase 0): 3D-stack index i maps to split_data z(i + offset);
    # split_data z0 is a projection excluded from the stack, so the offset is 1.
    z_index_offset: int = 1

    # Diagnostic mode: emit ALL linked slices (each cell at every z) instead of only the
    # selected representative slice. Used for the per-slice vs one-per-cell comparison on
    # identical masks. Default OFF (normal one-row-per-cell output).
    emit_all_slices: bool = False

    # I/O
    mask_pattern: str = "*_pred_mask_3d.zarr"
    # tif so downstream extraction can read it: incarta uses cv2.imread and
    # mcherry_metrics uses tifffile.imread — neither reads .zarr (matches the tracked
    # branch's final_2d/, which is also .tif).
    output_label_format: str = "tif"

    # Execution
    n_jobs: Optional[int] = None  # None -> SLURM_CPUS_PER_TASK / cpu_count
    overwrite_existing: bool = False


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
class ScportraitConfig:
    """scPortrait deep-learning feature extraction configuration."""
    project_location: str = "tmp/scportrait_projects"
    config_path: str = "src/feature_extraction/scportrait_project/config.yml"
    channel_names: List[str] = field(default_factory=lambda: ["brightfield", "brightfield_ch1"])
    overwrite: bool = True
    debug: bool = False
    save_plots: bool = True


@dataclass
class MorphologyConfig:
    """Morphology feature extraction configuration."""
    extract_shape: bool = True
    extract_firstorder: bool = True
    extract_glcm: bool = True
    extract_glrlm: bool = False
    extract_glszm: bool = False

@dataclass
class IncartaConfig:
    """Incarta feature extraction configuration."""
    features: Dict[str, bool] = field(default_factory=lambda: {
        "morphology": True,
        "intensity": True,
        "spatial": True,
        "texture": True
    })


#: Every feature-extraction method name the pipeline, schema and CLI recognise.
FEATURE_METHODS: Tuple[str, ...] = (
    "incarta",
    "regionprops",
    "pyradiomics",
    "scportrait",
)

#: Recognised but not usable yet, mapped to the reason given to the user. The
#: name stays reserved so configs, the CLI and the schema keep accepting it.
UNAVAILABLE_FEATURE_METHODS: Dict[str, str] = {
    "pyradiomics": (
        "the legacy backend was retired and its replacement has not landed. "
        "Use 'incarta', 'regionprops' or 'scportrait' instead."
    ),
}


def check_feature_method_available(method: str) -> None:
    """Raise unless ``method`` is a recognised, currently usable method.

    Raises:
        ValueError: ``method`` is not in :data:`FEATURE_METHODS`.
        NotImplementedError: ``method`` is reserved but not yet available.
    """
    if method not in FEATURE_METHODS:
        raise ValueError(
            f"Unsupported feature extraction method: {method!r}; "
            f"expected one of {list(FEATURE_METHODS)}"
        )
    if method in UNAVAILABLE_FEATURE_METHODS:
        raise NotImplementedError(
            f"The {method!r} feature-extraction method is not yet available: "
            + UNAVAILABLE_FEATURE_METHODS[method]
        )


@dataclass
class FeatureExtractionConfig:
    """Feature extraction configuration."""
    n_jobs: int = -1  # Use all available cores (or set to 0)

    method: str = "incarta"
    image_pattern: str = "*_BF.tif"
    mask_pattern: str = "*_pred_mask.tif"
            
    preprocessing: Dict[str, Any] = field(default_factory=lambda: {
        "clip_percentiles": [1, 99]
    })
    output: Dict[str, Any] = field(default_factory=lambda: {
        "save_individual_files": True,
        "save_combined_file": True,
        "include_metadata": True,
        "individual_format": "{image_name}_features.csv",
        "combined_filename": "all_features.csv",
        "create_subdirs": True,
    })

    scportrait: ScportraitConfig = field(default_factory=ScportraitConfig)


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
    output_dir: str = "feature_visualizations_temp"
    save_plots: bool = True
    show_plots: bool = False

    # Plot styling
    style: str = "whitegrid"  # Seaborn style
    font_size: int = 12
    figure_size: List[int] = field(default_factory=lambda: [12, 8])
    dpi: int = 300
    colormap: str = "viridis"
    color_palette: str = "tab10"  # for sample colors
    napari: NapariConfig = field(default_factory=NapariConfig)

    # Dimensionality reduction settings
    reduction_methods: Dict[str, Any] = field(default_factory=lambda: {
        "pca": {
            "n_components": 2,
            "random_state": 42
        },
        "tsne": {
            "n_components": 2,
            "random_state": 42,
            "perplexity": 30,
            "n_iter": 1000
        },
        "umap": {
            "n_components": 2,
            "random_state": 42,
            "n_neighbors": 15,
            "min_dist": 0.1
        }
    })

    # Visualization levels
    levels: Dict[str, Any] = field(default_factory=lambda: {
        "z_stack": {
            "enabled": True,
            "max_features_violin": 5,  # Max features for violin plots
            "min_cells_per_stack": 3   # Min cells required for z-stack analysis
        },
        "sample": {
            "enabled": True,
            "max_features_heatmap": 8  # Max features for correlation heatmap
        },
        "dataset": {
            "enabled": True,
            "max_features_boxplot": 6,   # Max features for boxplot
            "max_features_correlation": 10  # Max features for correlation matrix
        }
    })

    # Interactive plotting (plotly, bokeh, etc.)
    interactive: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "hover_data": ["cell_id", "z_index", "z_stack", "sample_id"],
        "plot_width": 800,
        "plot_height": 600
    })

    # TensorBoard logging
    tensorboard: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "log_dir": "tensorboard_logs"
    })

    # Feature selection for visualization
    features: Dict[str, Any] = field(default_factory=lambda: {
        # Features to always include in analysis
        "priority_features": [
            "area", "elongation", "circularity", "mean_intensity", "entropy"
        ],
        # Features to exclude from dimensionality reduction
        "exclude_from_reduction": [
            "cell_id", "scportrait_cell_id", "instance_id", "filename",
            "image_path", "sample_id", "timepoint", "z_index",
            "z_stack", "sample_z_id", "centroid_x", "centroid_y",
            "center_of_mass_x", "center_of_mass_y",
        ]
    })

'''
# =============================================================================
# Dimensionality Reduction Settings
# =============================================================================

@dataclass
class DimensionalityReductionConfig:
    """Dimensionality reduction configuration."""
    methods: Dict[str, Any] = field(default_factory=lambda: {
        "pca": {
            "n_components": 2,
            "random_state": 42
        },
        "tsne": {
            "n_components": 2,
            "random_state": 42,
            "perplexity": 30,
            "n_iter": 1000
        },
        "umap": {
            "n_components": 2,
            "random_state": 42,
            "n_neighbors": 15,
            "min_dist": 0.1
        }
    })

    # Visualization levels
    levels: Dict[str, Any] = field(default_factory=lambda: {
        "z_stack": {
            "enabled": True,
            "max_features_violin": 5,  # Max features for violin plots
            "min_cells_per_stack": 3   # Min cells required for z-stack analysis
        },
        "sample": {
            "enabled": True,
            "max_features_heatmap": 8  # Max features for correlation heatmap
        },
        "dataset": {
            "enabled": True,
            "max_features_boxplot": 6,   # Max features for boxplot
            "max_features_correlation": 10  # Max features for correlation matrix
        }
    })

    # Interactive plotting
    interactive: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "hover_data": ["cell_id", "z_index", "z_stack", "sample_id"],
        "plot_width": 800,
        "plot_height": 600
    })

    # TensorBoard logging
    tensorboard: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "log_dir": "tensorboard_logs"
    })

    # Feature selection for visualization
    features: Dict[str, Any] = field(default_factory=lambda: {
        # Features to always include in analysis
        "priority_features": [
            "area",
            "elongation",
            "circularity",
            "mean_intensity",
            "entropy"
        ],
        # Features to exclude from dimensionality reduction
        "exclude_from_reduction": [
            "cell_id",
            "scportrait_cell_id",
            "instance_id",
            "filename",
            "image_path",
            "sample_id",
            "timepoint",
            "z_index",
            "z_stack",
            "sample_z_id",
            "centroid_x",
            "centroid_y",
            "center_of_mass_x",
            "center_of_mass_y"
        ]
    })
'''

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

# Distribution Configuration
@dataclass
class DistributionConfig:
    """Distributed processing configuration."""
    enabled: bool = False
    backend: str = "nccl"  # or "gloo"
    init_method: Optional[str] = "env://"


# =============================================================================
# Main Pipeline Configuration
# =============================================================================

@dataclass
class PipelineConfig:
    """Main pipeline configuration schema."""
    paths: PathsConfig = field(default_factory=PathsConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    postprocessing: PostprocessingConfig = field(default_factory=PostprocessingConfig)
    representative_slice: RepresentativeSliceConfig = field(
        default_factory=RepresentativeSliceConfig)
    feature_extraction: FeatureExtractionConfig = field(default_factory=FeatureExtractionConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    distributed: DistributionConfig = field(default_factory=DistributionConfig)

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
    if config.postprocessing.tracking.search_range <= 0:
        raise ValueError("Tracking search range must be positive")
    
    if config.postprocessing.tracking.min_area >= config.postprocessing.tracking.max_area:
        raise ValueError("Tracking min_area must be less than max_area")
    
    # Filtering validation
    if not 0 <= config.postprocessing.filtering.blur_threshold <= 1:
        raise ValueError("Blur threshold should typically be between 0 and 1")

    # Representative-slice validation (value ranges only; path checks are deferred to
    # the stage entrypoint, since these defaults apply to every pipeline invocation)
    rs = config.representative_slice
    if not 0 < rs.sharpness_gate_fraction <= 1:
        raise ValueError(
            "representative_slice.sharpness_gate_fraction must be in (0, 1]"
        )
    if rs.selection_metric not in ("area", "sharpness"):
        raise ValueError(
            "representative_slice.selection_metric must be area|sharpness"
        )
    if rs.min_area >= rs.max_area:
        raise ValueError("representative_slice.min_area must be less than max_area")
    if rs.search_range <= 0:
        raise ValueError("representative_slice.search_range must be positive")
    if rs.z_index_offset < 0:
        raise ValueError("representative_slice.z_index_offset must be >= 0")
    if rs.output_label_format not in ("tif", "zarr", "hdf5"):
        raise ValueError(
            "representative_slice.output_label_format must be tif|zarr|hdf5"
        )

    # Feature extraction validation
    # Reserved-but-unavailable names are accepted here; availability is checked
    # where a run starts (pipeline constructor, CLI), see FEATURE_METHODS.
    valid_methods = list(FEATURE_METHODS)
    if config.feature_extraction.method not in valid_methods:
        raise ValueError(f"Feature extraction method must be one of {valid_methods}, got '{config.feature_extraction.method}'")
