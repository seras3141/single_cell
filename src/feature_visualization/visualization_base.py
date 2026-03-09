"""
Base classes and interfaces for the refactored visualization system.

This module provides the foundation for a modular, testable visualization architecture.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


@dataclass
class VisualizationConfig:
    """Global visualization configuration that applies to all plots."""
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 300
    font_size: int = 12
    style: str = "seaborn-v0_8-whitegrid"
    color_palette: str = "husl"
    figure_format: str = "png"
    # save_formats: List[str] = field(default_factory=lambda: ["png", "svg"]). # Update later if multiple fomats are needed
    output_dir: Path = field(default_factory=lambda: Path("visualizations"))
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'VisualizationConfig':
        """Create config from dictionary."""
        return cls(**config_dict)


@dataclass
class PlotConfig:
    """Configuration for individual plots - contains plot-specific settings only."""
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    alpha: float = 0.7
    marker_size: int = 50
    show_legend: bool = True
    interactive: bool = False
    figure_dpi: Optional[int] = None  # Override global DPI for specific plots


class VisualizationError(Exception):
    """Base exception for visualization errors."""
    pass


class DataLoadingError(VisualizationError):
    """Raised when data loading fails."""
    pass


class PlottingError(VisualizationError):
    """Raised when plotting operations fail."""
    pass


class BaseDataManager(ABC):
    """Abstract base class for data management operations."""
    
    @abstractmethod
    def load_data(self, source: Union[str, Path]) -> pd.DataFrame:
        """Load data from source."""
        pass
    
    @abstractmethod
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate data integrity."""
        pass
    
    @abstractmethod
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature columns."""
        pass

    @abstractmethod
    def get_data_summary(self, df: pd.DataFrame) -> str:
        """Return a summary of the loaded data."""
        pass


class BasePlotter(ABC):
    """Abstract base class for plotting strategies."""
    
    @abstractmethod
    def create_plot(self, data: pd.DataFrame, plot_config: PlotConfig, ax: Axes):
        """Create a plot from data."""
        pass

    def _setup_axes(self, ax: Axes, plot_config: PlotConfig) -> Axes:
        """Set up axes with consistent styling."""
        if plot_config.title:
            ax.set_title(plot_config.title) #, fontsize=self.config.font_size)
        if plot_config.xlabel:
            ax.set_xlabel(plot_config.xlabel) #, fontsize=self.config.font_size)
        if plot_config.ylabel:
            ax.set_ylabel(plot_config.ylabel) #, fontsize=self.config.font_size)

        return ax

    

class BaseExporter(ABC):
    """
    Abstract base class for result exporters.
    
    Future implementations planned:
    - ImageExporter: Save plots in multiple formats with metadata
    - TensorBoardExporter: Export embeddings and projector data
    - ReportExporter: Generate comprehensive analysis reports
    - DataExporter: Export processed data and statistics
    """
    
    @abstractmethod
    def export(self, results: Dict[str, Any], output_path: Path) -> None:
        """Export results to specified path."""
        pass

"""
This class was designed as an abstract base for visualization pipelines using
composition pattern (data_manager + plotters + exporters). Currently, the 
VisualizationPipeline class provides the concrete implementation using a 
different architectural approach.

The BaseVisualizer may be useful in the future for:
- Alternative pipeline implementations
- Plugin-based visualization systems
- More complex workflow orchestration
"""

class BaseVisualizer:
    """
    Base class for visualization pipelines.
    
    This class provides the framework for a modular visualization system
    where data management, plotting, and exporting are handled by separate
    specialized components.
    """
    
    def __init__(self, 
                 config: VisualizationConfig,
                 data_manager: BaseDataManager,
                 plotters: Dict[str, BasePlotter],
                 exporters: Dict[str, BaseExporter]):
        """
        Initialize the base visualizer.
        
        Args:
            config: Visualization configuration
            data_manager: Data loading and management component
            plotters: Dictionary of plotting strategies {name: plotter}
            exporters: List of result exporters
        """
        self.config = config
        self.data_manager = data_manager
        self.plotters = plotters
        self.exporters = exporters
        self._setup_output_directory()
    
    def _setup_output_directory(self) -> None:
        """Create output directory if it doesn't exist."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_data(self, source: Union[str, Path]) -> pd.DataFrame:
        """Load and validate data."""

        try:
            df = self.data_manager.load_data(source)
            if not self.data_manager.validate_data(df):
                raise DataLoadingError("Data validation failed")
            return df
        except Exception as e:
            raise DataLoadingError(f"Failed to load data from {source}: {e}")
    
    def create_visualization(self, 
                           data: pd.DataFrame, 
                           plot_types: List[str],
                           plot_configs: List[PlotConfig]) -> Figure:

        """Create (mxn) visualization with subplots."""
        fig = self._setup_figure()
        axes = self._setup_subplots(fig, len(plot_types))

        for plot_type, plot_config, ax in zip(plot_types, plot_configs, axes):
            self.add_plots(data, plot_type, plot_config, ax)

        return fig
        
    def add_plots(self, data: pd.DataFrame, plot_type: str, plot_config: PlotConfig, ax: Axes) -> None:
        """Add plots to an existing Axes object."""
        if plot_type not in self.plotters:
            available = list(self.plotters.keys())
            raise PlottingError(f"Unknown plot type '{plot_type}'. Available: {available}")

        try:
            plotter = self.plotters[plot_type]
            plotter.create_plot(data, plot_config, ax)
        except Exception as e:
            raise PlottingError(f"Failed to add {plot_type} plot: {e}")

    def export_results(self, results: Dict[str, Any]) -> None:
        """Export results using all configured exporters."""
        for name, exporter in self.exporters.items():
            try:
                output_path = self.config.output_dir / f"results_{exporter.__class__.__name__.lower()}"
                exporter.export(results, output_path)
            except Exception as e:
                # Log warning but don't fail the whole process
                print(f"Warning: Export failed with {exporter.__class__.__name__}: {e}")

    def _setup_figure(self, **kwargs) -> Figure:
        """Set up figure with consistent styling."""

        fig_size = kwargs.get('figsize', self.config.figure_size)
        dpi = kwargs.get('dpi', self.config.dpi)

        plt.style.use(self.config.style)
        fig = plt.figure(
            figsize=fig_size,
            dpi=dpi
        )

        return fig

    def _setup_subplots(self, fig: Figure, n_plots: Union[tuple, int]) -> List[Axes]:
        """Create subplots depending on number of plots."""
        import math

        if isinstance(n_plots, tuple) and len(n_plots) == 2:
            rows, cols = n_plots
            n_plots = rows * cols
        elif isinstance(n_plots, int):
            cols = math.ceil(math.sqrt(n_plots))
            rows = math.ceil(n_plots / cols)
        else:
            raise ValueError("n_plots must be an int or a tuple of (rows, cols)")

        axes = [fig.add_subplot(rows, cols, i + 1) for i in range(n_plots)] # type: ignore

        return axes

    
    def _save_figure(self, fig: Figure, filename: str, output_dir: Path) -> List[Path]:
        """Save figure in configured formats."""
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = []

        for fmt in [self.config.figure_format]:
            filepath = output_dir / f"{filename}.{fmt}"
            fig.savefig(
                filepath, 
                format=fmt, 
                dpi=self.config.dpi,
                bbox_inches='tight',
                transparent=True if fmt == 'svg' else False
            )
            saved_paths.append(filepath)
        
        return saved_paths




# Example usage and concrete implementations would go in separate files
if __name__ == "__main__":
    config = VisualizationConfig(
        figure_size=(10, 6),
        style="seaborn-v0_8-whitegrid",
        output_dir=Path("test_output")
    )
    
    print("Base visualization framework initialized")
    print(f"Config: {config}")
