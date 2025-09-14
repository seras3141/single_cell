"""
Extended configuration classes for the visualization system.

This module provides enhanced configuration classes that support all the
specialized parameters needed for different types of plots.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Union, Dict, Any
from pathlib import Path

from .visualization_base import PlotConfig, VisualizationConfig


@dataclass
class ScatterPlotConfig(PlotConfig):
    """Configuration specific to scatter plots."""
    x_column: Optional[str] = None  # For direct scatter plots
    y_column: Optional[str] = None  # For direct scatter plots
    z_column: Optional[str] = None  # For 3D scatter plots
    color_by: str = "sample_id"  # Column to use for coloring
    size_by: Optional[str] = None  # Column to use for marker size
    n_components: int = 2  # For displaying dimensionality (2 or 3)
    # NOTE: reduction_method removed - this should be handled before plotting


@dataclass
class DistributionPlotConfig(PlotConfig):
    """Configuration specific to distribution plots."""
    distribution_type: str = "histogram"  # 'histogram', 'boxplot', 'violinplot'
    feature_column: Optional[str] = None  # Specific feature to plot
    bins: int = 30  # For histograms
    group_by: str = "sample_id"  # Column to group by
    show_kde: bool = False  # Show kernel density estimate
    

@dataclass
class HeatmapConfig(PlotConfig):
    """Configuration for heatmap visualizations."""
    correlation_method: str = "pearson"  # 'pearson', 'spearman', 'kendall'
    cluster_rows: bool = True
    cluster_cols: bool = True
    show_values: bool = False
    colormap: str = "viridis"
    center: Optional[float] = None  # Center colormap at this value


@dataclass 
class AdvancedVisualizationConfig(VisualizationConfig):
    """Extended visualization configuration with advanced options."""
    
    # Feature selection
    max_features_for_correlation: int = 50
    feature_selection_method: str = "variance"  # 'variance', 'correlation', 'manual'
    exclude_low_variance: bool = True
    variance_threshold: float = 0.01
    
    # Color and styling
    use_colorblind_palette: bool = True
    custom_palette: Optional[List[str]] = None
    
    # Performance settings
    max_cells_for_plotting: int = 10000  # Subsample if more cells
    enable_caching: bool = True
    
    # Export settings
    export_data: bool = True  # Export processed data alongside plots
    export_config: bool = True  # Export configuration used
    
    # Interactive features
    enable_plotly: bool = False
    enable_bokeh: bool = False
    
    @classmethod
    def from_yaml(cls, yaml_path: Path) -> 'AdvancedVisualizationConfig':
        """Load configuration from YAML file."""
        import yaml
        
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(**config_dict.get('visualization', {}))
    
    def to_yaml(self, yaml_path: Path) -> None:
        """Save configuration to YAML file."""
        import yaml
        from dataclasses import asdict
        
        config_dict = {'visualization': asdict(self)}
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)


# Factory functions for common configurations
def create_pca_scatter_config(title: str = "PCA Scatter Plot", 
                             n_components: int = 2) -> ScatterPlotConfig:
    """Create a standard PCA scatter plot configuration."""
    return ScatterPlotConfig(
        title=title,
        n_components=n_components,
        xlabel=f"PC1",
        ylabel=f"PC2",
        alpha=0.7,
        marker_size=50,
        show_legend=True
    )


def create_tsne_scatter_config(title: str = "t-SNE Scatter Plot") -> ScatterPlotConfig:
    """Create a standard t-SNE scatter plot configuration."""
    return ScatterPlotConfig(
        title=title,
        n_components=2,
        xlabel="t-SNE 1",
        ylabel="t-SNE 2",
        alpha=0.7,
        marker_size=30,
        show_legend=True
    )


def create_umap_scatter_config(title: str = "UMAP Scatter Plot") -> ScatterPlotConfig:
    """Create a standard UMAP scatter plot configuration."""
    return ScatterPlotConfig(
        title=title,
        n_components=2,
        xlabel="UMAP 1",
        ylabel="UMAP 2",
        alpha=0.7,
        marker_size=30,
        show_legend=True
    )


def create_feature_distribution_config(feature: str, 
                                     plot_type: str = "violinplot") -> DistributionPlotConfig:
    """Create configuration for feature distribution plotting."""
    return DistributionPlotConfig(
        title=f"{feature} Distribution",
        distribution_type=plot_type,
        feature_column=feature,
        xlabel="Sample",
        ylabel=feature,
        show_legend=False
    )


def create_correlation_heatmap_config(title: str = "Feature Correlations") -> HeatmapConfig:
    """Create configuration for correlation heatmap."""
    return HeatmapConfig(
        title=title,
        correlation_method="pearson",
        cluster_rows=True,
        cluster_cols=True,
        colormap="RdBu_r",
        center=0,
        show_legend=True
    )


# Example usage
if __name__ == "__main__":
    # Create various plot configurations
    pca_config = create_pca_scatter_config("PCA Analysis", n_components=3)
    tsne_config = create_tsne_scatter_config("t-SNE Clustering")
    
    # Create advanced visualization config
    advanced_config = AdvancedVisualizationConfig(
        figure_size=(10, 8),
        max_features_for_correlation=30,
        use_colorblind_palette=True,
        enable_plotly=True
    )
    
    print("Extended configuration classes created successfully")
    print(f"PCA config: {pca_config.title}, components: {pca_config.n_components}")
    print(f"Advanced config: max features: {advanced_config.max_features_for_correlation}")
