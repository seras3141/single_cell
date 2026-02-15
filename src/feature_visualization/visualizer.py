"""
Specialized visualizer classes for different levels of analysis.

This module provides concrete implementations of visualization pipelines
specialized for sample-level and dataset-level analysis workflows.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from src.utils.dimensionality_reduction import DimensionalityReducer

from src.feature_visualization.visualization_base import (
    BaseVisualizer, VisualizationConfig, PlotConfig, 
    BaseDataManager, BasePlotter, BaseExporter
)
from src.feature_visualization.extended_config import ScatterPlotConfig, DistributionPlotConfig
from src.feature_visualization.data_manager import FeatureDataManager
from src.feature_visualization.plotters import ScatterPlotter, DistributionPlotter, create_correlation_plot, create_covariance_plot, create_feature_distribution_plot
from src.feature_visualization.exporters import ImageExporter, DataExporter, ReportExporter


logger = logging.getLogger(__name__)


class SampleLevelVisualizer(BaseVisualizer):
    """
    Visualizer specialized for single-sample analysis.
    
    This class focuses on analyzing individual samples, creating visualizations
    that explore within-sample variation and cellular heterogeneity.
    """
    
    def __init__(self, 
                 config: Optional[VisualizationConfig] = None,
                 data_manager: Optional[BaseDataManager] = None,
                 plotters: Optional[Dict[str, BasePlotter]] = None,
                 exporters: Optional[Dict[str, BaseExporter]] = None):
        """
        Initialize sample-level visualizer with defaults.
        
        Args:
            config: Visualization configuration
            data_manager: Data loading component
            plotters: Dictionary of plotting strategies
            exporters: Dictionary of result exporters
        """
        # Set up defaults if not provided
        if config is None:
            config = VisualizationConfig(
                output_dir=Path("sample_analysis_output"),
                figure_size=(10, 8)
            )
        
        if data_manager is None:
            data_manager = FeatureDataManager()
        
        if plotters is None:
            plotters = {
                'scatter': ScatterPlotter(),
                'distribution': DistributionPlotter()
            }
        
        if exporters is None:
            exporters = {
                'figure': ImageExporter(formats=['png']),
                'data': DataExporter(),
                'report': ReportExporter()
            }

        super().__init__(config, data_manager, plotters, exporters)
        self.sample_id = None
        self.current_sample_data = None

        # Initialize dimensionality reducer
        self.dim_reducer = DimensionalityReducer()

    
    def set_sample(self, sample_data: pd.DataFrame, sample_id: str) -> None:
        """
        Set the current sample for analysis.
        
        Args:
            sample_data: DataFrame containing single sample data
            sample_id: Identifier for this sample
        """
        if 'sample_id' not in sample_data.columns:
            raise ValueError("Sample data must contain 'sample_id' column")
        
        self.current_sample_data = sample_data
        self.sample_id = sample_id
        logger.info(f"Set sample {sample_id} with {len(sample_data)} cells")

    
    def create_sample_overview(self, 
                             reduction_methods: List[str] = ['pca', 'tsne'],
                             n_components: int = 2):
        """
        Create comprehensive overview visualizations for the current sample.
        
        Args:
            reduction_methods: List of dimensionality reduction methods to apply
            n_components: Number of components for reduction (2 or 3)
            
        Returns:
            Dictionary of figures created
        """
        if self.current_sample_data is None:
            raise ValueError("No sample data set. Call set_sample() first.")
        
        fig = self._setup_figure()
        axes = self._setup_subplots(fig, len(reduction_methods))

        for method, ax in zip(reduction_methods, axes):

            try:
                # Sample level dimensionality reduction
                X_reduced = self.dim_reducer.reduce_dimensionality(
                    self.current_sample_data, method=method, n_components=n_components
                )

                # Convert X_reduced to a pandas DataFrame for easier handling
                X_reduced = pd.DataFrame(
                    X_reduced,
                    columns=[f"{method.upper()}_{i+1}" for i in range(X_reduced.shape[1])],
                    index=self.current_sample_data.index
                )

                # Create scatter plot config for this method
                z_stack_available = 'z_stack' in self.current_sample_data.columns
                config = ScatterPlotConfig(
                    title=f"Sample {self.sample_id} - {method.upper()} Analysis",
                    xlabel=f"{method.upper()} Component 1",
                    ylabel=f"{method.upper()} Component 2",
                    color_by='z_stack' if z_stack_available else 'sample_id',
                    n_components=n_components
                )

                self.add_plots(X_reduced, 'scatter', config, ax)

                
            except Exception as e:
                logger.error(f"Failed to create {method} visualization: {e}")

        self.exporters['figure'].export({'figure':fig}, self.config.output_dir / f"sample_{self.sample_id}_overview")

        logger.info(f"Created overview plots for sample {self.sample_id}")

    def create_feature_distributions(self, 
                                   feature_columns: Optional[List[str]] = None,
                                   plot_types: List[str] = ['histogram', 'boxplot']) -> Dict[str, Figure]:
        """
        Create individual feature distribution plots for the current sample.
        
        Args:
            feature_columns: Specific features to plot (if None, auto-detect)
            plot_types: Types of distribution plots to create
            
        Returns:
            Dictionary of figures created
        """
        if self.current_sample_data is None:
            raise ValueError("No sample data set. Call set_sample() first.")
        
        figures = {}
        
        # Auto-detect feature columns if not provided
        if feature_columns is None:
            feature_columns = self.data_manager.get_feature_columns(self.current_sample_data)
        
        # Limit to first 5 features for readability
        feature_columns = feature_columns[:5]
        
        for plot_type in plot_types:
            for feature in feature_columns:
                try:
                    config = DistributionPlotConfig(
                        title=f"Sample {self.sample_id} - {feature} Distribution",
                        feature_column=feature,
                        distribution_type=plot_type,
                        xlabel="Value" if plot_type == 'histogram' else "Feature",
                        ylabel="Frequency" if plot_type == 'histogram' else feature
                    )
                    
                    figure = self.create_visualization(
                        self.current_sample_data,
                        ['distribution'],
                        [config]
                    )
                    
                    figures[f'{feature}_{plot_type}'] = figure
                    
                except Exception as e:
                    logger.error(f"Failed to create {plot_type} for {feature}: {e}")
        
        return figures

    def create_sample_summary_plots(self, 
                                   feature_columns: Optional[List[str]] = None,
                                   correlation_method: str = 'pearson') -> Figure:
        """
        Create side-by-side visualizations of feature statistics for the current sample.
        
        Creates three plots:
        1. Feature distribution (violin plots)
        2. Feature correlation matrix (heatmap)
        3. Feature covariance matrix (heatmap)
        
        Args:
            feature_columns: Specific features to analyze (if None, auto-detect)
            correlation_method: Method for correlation calculation ('pearson', 'spearman', 'kendall')
            
        Returns:
            Figure containing the three plots
        """
        if self.current_sample_data is None:
            raise ValueError("No sample data set. Call set_sample() first.")
        
        # Auto-detect feature columns if not provided
        if feature_columns is None:
            feature_columns = self.data_manager.get_feature_columns(self.current_sample_data)
        
        # Limit to manageable number of features for visualization
        if len(feature_columns) > 20:
            logger.warning(f"Too many features ({len(feature_columns)}), limiting to first 20 for visualization")
            feature_columns = feature_columns[:20]
        
        # Get numeric feature data
        feature_data = self.current_sample_data[feature_columns].select_dtypes(include=[np.number])
        
        if feature_data.empty or len(feature_data.columns) < 2:
            raise ValueError("Need at least 2 numeric features for summary plots")
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(f'Sample {self.sample_id} - Feature Summary Analysis', fontsize=16, fontweight='bold')
        
        try:
            # Plot 1: Feature Distribution
            create_feature_distribution_plot(feature_data, axes[0])

            # Plot 2: Correlation Matrix
            create_correlation_plot(feature_data, axes[1], correlation_method)

            # Plot 3: Covariance Matrix
            create_covariance_plot(feature_data, axes[2])
            
        except Exception as e:
            logger.error(f"Error creating summary plots: {e}")
            raise
        
        plt.tight_layout()
        
        # Export the figure
        output_path = self.config.output_dir / f"sample_{self.sample_id}_summary"
        self.exporters['figure'].export({'figure': fig}, output_path)
        
        logger.info(f"Created feature summary plots for sample {self.sample_id}")
        return fig

    def export_sample_results(self) -> None:
        """
        Export all sample analysis results.
        
        Args:
            figures: Dictionary of figures to export
        """
        if self.sample_id is None:
            raise ValueError("No sample ID set")
        
        # Prepare results for export
        results = {
            'sample_id': self.sample_id,
            'cell_count': len(self.current_sample_data) if self.current_sample_data is not None else 0,
            'data': self.current_sample_data,
            'metadata': {
                'analysis_type': 'sample_level',
                'sample_id': self.sample_id,
                'timestamp': pd.Timestamp.now().isoformat()
            }
        }
        
        # Create sample-specific output directory
        sample_output_dir = self.config.output_dir / f"sample_{self.sample_id}"
        sample_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export with each exporter
        for exporter_name, exporter in self.exporters.items():
            try:
                output_path = sample_output_dir / f"sample_{self.sample_id}_{exporter_name}"
                exporter.export(results, output_path)
                logger.info(f"Exported sample results with {exporter_name}")
            except Exception as e:
                logger.warning(f"Export failed with {exporter_name}: {e}")

    def run_visualizer(self):
        """
        Run the full visualization pipeline for the current sample.
        
        This includes creating overview plots, feature distributions,
        and exporting all results.
        """
        if self.current_sample_data is None or self.sample_id is None:
            raise ValueError("No sample data set. Call set_sample() first.")
        
        
        # Create overview visualizations
        overview_figures = self.create_sample_overview()
        
        # Create feature distribution plots
        # distribution_figures = self.create_feature_distributions()
        # all_figures.update(distribution_figures)
        
        # Create summary plots
        summary_figure = self.create_sample_summary_plots()
        
        # Export all results
        self.export_sample_results()
        
        logger.info(f"Completed visualization pipeline for sample {self.sample_id}")
                            
class DatasetLevelVisualizer(BaseVisualizer):
    """
    Visualizer specialized for multi-sample dataset analysis.
    
    This class focuses on comparing samples and finding patterns across
    the entire dataset, including batch effects and sample-to-sample variation.
    """
    
    def __init__(self, 
                 config: Optional[VisualizationConfig] = None,
                 data_manager: Optional[BaseDataManager] = None,
                 plotters: Optional[Dict[str, BasePlotter]] = None,
                 exporters: Optional[Dict[str, BaseExporter]] = None):
        """
        Initialize dataset-level visualizer with defaults.
        
        Args:
            config: Visualization configuration
            data_manager: Data loading component
            plotters: Dictionary of plotting strategies
            exporters: Dictionary of result exporters
        """
        # Set up defaults if not provided
        if config is None:
            config = VisualizationConfig(
                output_dir=Path("dataset_analysis_output"),
                figure_size=(12, 10)
            )
        
        if data_manager is None:
            data_manager = FeatureDataManager()
        
        if plotters is None:
            plotters = {
                'scatter': ScatterPlotter(),
                'distribution': DistributionPlotter()
            }
        
        if exporters is None:
            exporters = {
                'image': ImageExporter(formats=['png', 'svg']),
                'data': DataExporter(),
                'report': ReportExporter()
            }
        
        super().__init__(config, data_manager, plotters, exporters)
        self.dataset = None
        self.sample_ids = []

        self.dim_reducer = DimensionalityReducer()
    
    def set_dataset(self, dataset: pd.DataFrame) -> None:
        """
        Set the dataset for analysis.
        
        Args:
            dataset: DataFrame containing multi-sample data
        """
        if 'sample_id' not in dataset.columns:
            raise ValueError("Dataset must contain 'sample_id' column")
        
        self.dataset = dataset
        self.sample_ids = list(dataset['sample_id'].unique())
        logger.info(f"Set dataset with {len(dataset)} cells from {len(self.sample_ids)} samples")
    
    def create_dataset_overview(self, 
                              reduction_methods: List[str] = ['pca', 'umap'],
                              n_components: int = 2,
                              color_by: str = 'sample_id'):
        """
        Create comprehensive overview visualizations for the entire dataset.
        
        Args:
            reduction_methods: List of dimensionality reduction methods to apply
            n_components: Number of components for reduction (2 or 3)
            color_by: Column to use for coloring points
            
        Returns:
            Dictionary of figures created
        """
        if self.dataset is None:
            raise ValueError("No dataset set. Call set_dataset() first.")

        fig = self._setup_figure()
        axes = self._setup_subplots(fig, len(reduction_methods))

        for method, ax in zip(reduction_methods, axes):

            try:
                # Sample level dimensionality reduction
                X_reduced = self.dim_reducer.reduce_dimensionality(
                    self.dataset, method=method, n_components=n_components
                )

                # Convert X_reduced to a pandas DataFrame for easier handling
                X_reduced = pd.DataFrame(
                    X_reduced,
                    columns=[f"{method.upper()}_{i+1}" for i in range(X_reduced.shape[1])],
                    index=self.dataset.index
                )

                # Create scatter plot config for this method
                config = ScatterPlotConfig(
                    title=f"Dataset Overview - {method.upper()} Analysis ({len(self.sample_ids)} samples)",
                    xlabel=f"{method.upper()} Component 1",
                    ylabel=f"{method.upper()} Component 2",
                    color_by=color_by,
                    n_components=n_components,
                    show_legend=True
                )

                self.add_plots(X_reduced, 'scatter', config, ax)
                
                
            except Exception as e:
                logger.error(f"Failed to create dataset {method} visualization: {e}")

            logger.info(f"Created dataset {method} overview with {len(self.dataset)} cells")

        fig.tight_layout()
        fig.savefig(self.config.output_dir / f"dataset_overview.png", dpi=self.config.dpi)

        # self.exporters['figure'].export({'figure':fig}, self.config.output_dir / f"dataset_overview")        
        
    
    def create_sample_comparison(self, 
                               feature_columns: Optional[List[str]] = None,
                               plot_types: List[str] = ['boxplot', 'violinplot']) -> Dict[str, Figure]:
        """
        Create plots comparing features across samples.
        
        Args:
            feature_columns: Specific features to compare (if None, auto-detect)
            plot_types: Types of comparison plots to create
            
        Returns:
            Dictionary of figures created
        """
        if self.dataset is None:
            raise ValueError("No dataset set. Call set_dataset() first.")
        
        figures = {}
        
        # Auto-detect feature columns if not provided
        if feature_columns is None:
            feature_columns = self.data_manager.get_feature_columns(self.dataset)
        
        # Limit to first 3 features for readability in comparison plots
        feature_columns = feature_columns[:3]
        
        for plot_type in plot_types:
            for feature in feature_columns:
                try:
                    config = DistributionPlotConfig(
                        title=f"Sample Comparison - {feature} ({plot_type})",
                        feature_column=feature,
                        distribution_type=plot_type,
                        group_by='sample_id',
                        xlabel="Sample",
                        ylabel=feature
                    )
                    
                    figure = self.create_visualization(
                        self.dataset,
                        ['distribution'],
                        [config]
                    )
                    
                    figures[f'comparison_{feature}_{plot_type}'] = figure
                    
                except Exception as e:
                    logger.error(f"Failed to create {plot_type} comparison for {feature}: {e}")

        for name, fig in figures.items():
            fig.savefig(self.config.output_dir / f"{name}.png", dpi=self.config.dpi)

        return figures
    
    def create_per_sample_analysis(self, 
                                 reduction_method: str = 'pca',
                                 max_samples: int = 6):
        """
        Create individual analysis for each sample in the dataset.
        
        Args:
            reduction_method: Dimensionality reduction method to use
            max_samples: Maximum number of samples to analyze individually
            
        Returns:
            Dictionary of figures created
        """
        if self.dataset is None:
            raise ValueError("No dataset set. Call set_dataset() first.")
        
        sample_visualizer = SampleLevelVisualizer(
            config=VisualizationConfig(
                output_dir=self.config.output_dir / "individual_samples",
                figure_size=self.config.figure_size
            )
        )
        
        # Limit number of samples to avoid too many plots
        samples_to_analyze = self.sample_ids[:max_samples]
        
        for sample_id in samples_to_analyze:
            try:
                sample_data = self.dataset[self.dataset['sample_id'] == sample_id]
                
                if len(sample_data) < 5:  # Skip samples with too few cells
                    logger.warning(f"Skipping sample {sample_id} - too few cells ({len(sample_data)})")
                    continue

                sample_visualizer.set_sample(sample_data, sample_id)
                sample_visualizer.run_visualizer()
                
                # sample_figures = sample_visualizer.create_sample_overview(
                #     reduction_methods=[reduction_method]
                # )
                
                # # Add sample figures to main collection
                # if sample_figures:
                #     for fig_name, figure in sample_figures.items():
                #         figures[f'sample_{sample_id}_{fig_name}'] = figure
                
            except Exception as e:
                logger.error(f"Failed to analyze sample {sample_id}: {e}")
        
    def export_dataset_results(self) -> None:
        """
        Export all dataset analysis results.
        """

        if self.dataset is None:
            raise ValueError("No dataset set")
        
        # Prepare results for export
        results = {
            'sample_count': len(self.sample_ids),
            'cell_count': len(self.dataset),
            'data': self.dataset,
            'metadata': {
                'analysis_type': 'dataset_level',
                'sample_ids': self.sample_ids,
                'timestamp': pd.Timestamp.now().isoformat()
            }
        }
        
        # Export with each exporter
        for exporter_name, exporter in self.exporters.items():
            try:
                output_path = self.config.output_dir / f"dataset_{exporter_name}"
                exporter.export(results, output_path)
                logger.info(f"Exported dataset results with {exporter_name}")
            except Exception as e:
                logger.warning(f"Export failed with {exporter_name}: {e}")

    def create_dataset_summary_plots(self, 
                                   feature_columns: Optional[List[str]] = None,
                                   correlation_method: str = 'pearson') -> Figure:
        """
        Create side-by-side visualizations of feature statistics for the whole dataset.
        
        Creates three plots:
        1. Feature distribution (violin plots)
        2. Feature correlation matrix (heatmap)
        3. Feature covariance matrix (heatmap)
        
        Args:
            feature_columns: Specific features to analyze (if None, auto-detect)
            correlation_method: Method for correlation calculation ('pearson', 'spearman', 'kendall')
            
        Returns:
            Figure containing the three plots
        """
        if self.dataset is None:
            raise ValueError("Dataset is not initialized.")
        
        # Auto-detect feature columns if not provided
        if feature_columns is None:
            feature_columns = self.data_manager.get_feature_columns(self.dataset)

        # Limit to manageable number of features for visualization
        if len(feature_columns) > 20:
            logger.warning(f"Too many features ({len(feature_columns)}), limiting to first 20 for visualization")
            feature_columns = feature_columns[:20]
        
        # Get numeric feature data
        feature_data = self.dataset[feature_columns].select_dtypes(include=[np.number])

        if feature_data.empty or len(feature_data.columns) < 2:
            raise ValueError("Need at least 2 numeric features for summary plots")
        
        # Create figure with subplots
        fig = self._setup_figure(figsize=(20, 6))
        axes = self._setup_subplots(fig, (1, 3))
        fig.suptitle(f'Dataset - Feature Summary Analysis', fontsize=16, fontweight='bold')
        
        try:
            # Plot 1: Feature Distribution
            create_feature_distribution_plot(feature_data, axes[0])

            # Plot 2: Correlation Matrix
            create_correlation_plot(feature_data, axes[1], correlation_method)

            # Plot 3: Covariance Matrix
            create_covariance_plot(feature_data, axes[2])
            
        except Exception as e:
            logger.error(f"Error creating summary plots: {e}")
            raise
        
        plt.tight_layout()
        
        # Export the figure
        output_path = self.config.output_dir / f"dataset_summary"
        fig.savefig(output_path, dpi=self.config.dpi)

        logger.info(f"Created feature summary plots for dataset")
        return fig


    def run_visualizer(self):
        """
        Run the full visualization pipeline for the entire dataset.
        
        This includes creating overview plots, sample comparisons,
        per-sample analyses, and exporting all results.
        """
        if self.dataset is None:
            raise ValueError("No dataset set. Call set_dataset() first.")
        
        
        # Create dataset overview visualizations
        overview_figures = self.create_dataset_overview()
        # all_figures.update(overview_figures)
        
        # Create sample comparison plots
        comparison_figures = self.create_sample_comparison()
        # all_figures.update(comparison_figures)

        # Create dataset summary plots
        self.create_dataset_summary_plots()

        # Create per-sample analyses
        per_sample_figures = self.create_per_sample_analysis()
        
        # Export all results
        self.export_dataset_results()
        
        logger.info(f"Completed visualization pipeline for dataset with {len(self.sample_ids)} samples")
        
        # return all_figures

# Factory functions for easy creation
def create_sample_visualizer(output_dir: Optional[Union[str, Path]] = None) -> SampleLevelVisualizer:
    """Create a sample-level visualizer with default configuration."""
    config = None
    if output_dir:
        config = VisualizationConfig(output_dir=Path(output_dir))
    
    return SampleLevelVisualizer(config=config)


def create_dataset_visualizer(output_dir: Optional[Union[str, Path]] = None) -> DatasetLevelVisualizer:
    """Create a dataset-level visualizer with default configuration."""
    config = None
    if output_dir:
        config = VisualizationConfig(output_dir=Path(output_dir))
    
    return DatasetLevelVisualizer(config=config)


# Example usage
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("Sample and Dataset Visualizers initialized")
    
    # Example: Create visualizers
    sample_viz = create_sample_visualizer("sample_output")
    dataset_viz = create_dataset_visualizer("dataset_output_p2126_J04")

    df = dataset_viz.data_manager.load_data("/Users/serenasritharan/Projects/single-cell/data/sample_data_out/features_output/split_data/test/test")
    
    print(dataset_viz.data_manager.get_data_summary(df))

    dataset_viz.set_dataset(df)
    dataset_viz.run_visualizer()

    print(f"Sample visualizer: {sample_viz}")
    print(f"Dataset visualizer: {dataset_viz}")
