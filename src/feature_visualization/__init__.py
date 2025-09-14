"""
Visualization module for single-cell analysis.

This module provides a modular, extensible system for creating visualizations
of single-cell feature data. It includes support for dimensionality reduction,
statistical plots, interactive visualizations, and TensorBoard integration.
"""

from .visualization_base import (
    VisualizationConfig,
    PlotConfig,
    BaseDataManager,
    BasePlotter,
    BaseExporter,
    VisualizationError,
    DataLoadingError,
    PlottingError
)

from .visualization_pipeline import VisualizationPipeline
from .data_manager import FeatureDataManager
from .plotters import (
    ScatterPlotter,
    DistributionPlotter
)

from .extended_config import (
    ScatterPlotConfig,
    DistributionPlotConfig,
    HeatmapConfig,
    AdvancedVisualizationConfig
)

from .visualizer import (
    SampleLevelVisualizer,
    DatasetLevelVisualizer,
    create_sample_visualizer,
    create_dataset_visualizer
)

try:
    from .tensorboard_projector import TensorBoardProjector
except ImportError:
    # TensorBoard may not be available
    TensorBoardProjector = None

try:
    from .exporters import (
        ImageExporter,
        DataExporter,
        ReportExporter,
        TensorBoardExporter,
        create_exporter
    )
except ImportError:
    # Exporters may not be available if dependencies are missing
    ImageExporter = None
    DataExporter = None
    ReportExporter = None
    TensorBoardExporter = None
    create_exporter = None

__all__ = [
    'VisualizationConfig',
    'PlotConfig', 
    'BaseDataManager',
    'BasePlotter',
    'BaseExporter',
    'VisualizationError',
    'DataLoadingError',
    'PlottingError',
    'VisualizationPipeline',
    'FeatureDataManager',
    'ScatterPlotter',
    'DistributionPlotter', 
    'ScatterPlotConfig',
    'DistributionPlotConfig',
    'HeatmapConfig',
    'AdvancedVisualizationConfig',
    'SampleLevelVisualizer',
    'DatasetLevelVisualizer',
    'create_sample_visualizer',
    'create_dataset_visualizer',
    'TensorBoardProjector',
    'ImageExporter',
    'DataExporter',
    'ReportExporter',
    'TensorBoardExporter',
    'create_exporter'
]
