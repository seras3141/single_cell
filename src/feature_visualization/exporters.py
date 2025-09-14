"""
Concrete implementations of BaseExporter for different output formats.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import logging

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    pd = None
    plt = None
    Figure = None

from .visualization_base import BaseExporter


logger = logging.getLogger(__name__)


class ImageExporter(BaseExporter):
    """
    Exports plots as image files with metadata.
    
    Supports multiple formats (PNG, SVG, PDF) and includes
    metadata about plot creation parameters.
    """
    
    def __init__(self, 
                 formats: Optional[List[str]] = None,
                 dpi: int = 300,
                 include_metadata: bool = True):
        """
        Initialize ImageExporter.
        
        Args:
            formats: List of image formats to export ['png', 'svg', 'pdf']
            dpi: Resolution for raster formats
            include_metadata: Whether to save metadata alongside images
        """
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError("matplotlib and pandas are required for ImageExporter")
        
        self.formats = formats or ['png', 'svg']
        self.dpi = dpi
        self.include_metadata = include_metadata
    
    def export(self, results: Dict[str, Any], output_path: Path) -> None:
        """
        Export matplotlib figures as image files.
        
        Args:
            results: Dictionary containing 'figure' and optionally 'metadata'
            output_path: Base path for output files (without extension)
        """
        try:
            if isinstance(results, dict):
                figure = results.get('figure')
            elif isinstance(results, Figure):
                figure = results
            else:
                raise ValueError("Results must be a dictionary or a matplotlib Figure")
            
            if figure is None:
                raise ValueError("No figure found in results to export")

            if Figure and not isinstance(figure, Figure):
                raise ValueError("Results must contain a 'figure' key with matplotlib Figure")

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            saved_files = []
            
            # Save in each requested format
            for fmt in self.formats:
                file_path = output_path.with_suffix(f'.{fmt}')
                
                # Configure format-specific options
                save_kwargs = {
                    'dpi': self.dpi if fmt in ['png', 'jpg', 'jpeg'] else None,
                    'bbox_inches': 'tight',
                    'transparent': fmt == 'svg',
                    'format': fmt
                }
                
                # Remove None values
                save_kwargs = {k: v for k, v in save_kwargs.items() if v is not None}
                
                figure.savefig(str(file_path), **save_kwargs)
                saved_files.append(file_path)
                logger.info(f"Saved {fmt.upper()} to {file_path}")
            
            # Save metadata if requested
            if self.include_metadata and 'metadata' in results:
                metadata_path = output_path.with_suffix('.json')
                metadata = {
                    'timestamp': datetime.now().isoformat(),
                    'formats': self.formats,
                    'dpi': self.dpi,
                    'files': [str(f) for f in saved_files],
                    **results['metadata']
                }
                
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
                logger.info(f"Saved metadata to {metadata_path}")
            
            # Close figure to free memory
            if plt:
                plt.close(figure)
            
        except Exception as e:
            logger.error(f"Failed to export image: {e}")
            raise


class DataExporter(BaseExporter):
    """
    Exports processed data and statistics to various formats.
    
    Supports CSV, JSON, and HDF5 formats for data export.
    """
    
    def __init__(self, 
                 data_format: str = 'csv',
                 include_statistics: bool = True):
        """
        Initialize DataExporter.
        
        Args:
            data_format: Format for data export ('csv', 'json', 'hdf5')
            include_statistics: Whether to include summary statistics
        """
        self.data_format = data_format
        self.include_statistics = include_statistics
    
    def export(self, results: Dict[str, Any], output_path: Path) -> None:
        """
        Export data and statistics.
        
        Args:
            results: Dictionary containing 'data' and optionally 'statistics'
            output_path: Base path for output files
        """
        try:
            data = results.get('data')
            if data is None:
                logger.warning("No data found in results, skipping data export")
                return
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Export main data
            if pd and hasattr(data, 'to_csv'):  # Check if it's DataFrame-like
                if self.data_format == 'csv':
                    data_path = output_path.with_suffix('.csv')
                    data.to_csv(data_path, index=False)
                elif self.data_format == 'json':
                    data_path = output_path.with_suffix('.json')
                    data.to_json(data_path, orient='records', indent=2)
                elif self.data_format == 'hdf5':
                    data_path = output_path.with_suffix('.h5')
                    data.to_hdf(data_path, key='data', mode='w')
                
                logger.info(f"Saved data to {data_path}")
            
            # Export statistics if available and requested
            if self.include_statistics and 'statistics' in results:
                stats_path = output_path.with_suffix('.stats.json')
                statistics = results['statistics']
                
                # Ensure statistics are JSON serializable
                if isinstance(statistics, dict):
                    with open(stats_path, 'w') as f:
                        json.dump(statistics, f, indent=2, default=str)
                    logger.info(f"Saved statistics to {stats_path}")
            
        except Exception as e:
            logger.error(f"Failed to export data: {e}")
            raise


class ReportExporter(BaseExporter):
    """
    Generates comprehensive analysis reports in various formats.
    
    Creates HTML or markdown reports with embedded visualizations
    and analysis summaries.
    """
    
    def __init__(self, 
                 report_format: str = 'html',
                 include_code: bool = False):
        """
        Initialize ReportExporter.
        
        Args:
            report_format: Format for report ('html', 'markdown')
            include_code: Whether to include code snippets in report
        """
        self.report_format = report_format
        self.include_code = include_code
    
    def export(self, results: Dict[str, Any], output_path: Path) -> None:
        """
        Generate and export analysis report.
        
        Args:
            results: Dictionary containing analysis results and metadata
            output_path: Path for output report file
        """
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if self.report_format == 'html':
                self._export_html_report(results, output_path.with_suffix('.html'))
            elif self.report_format == 'markdown':
                self._export_markdown_report(results, output_path.with_suffix('.md'))
            
        except Exception as e:
            logger.error(f"Failed to export report: {e}")
            raise
    
    def _export_html_report(self, results: Dict[str, Any], output_path: Path) -> None:
        """Export HTML report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Single-Cell Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; }}
                .section {{ margin: 20px 0; }}
                .metadata {{ background-color: #f9f9f9; padding: 10px; }}
                .statistics {{ display: flex; flex-wrap: wrap; gap: 20px; }}
                .stat-box {{ border: 1px solid #ddd; padding: 15px; min-width: 200px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Single-Cell Feature Analysis Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Analysis Summary</h2>
                <div class="metadata">
                    {self._format_metadata_html(results.get('metadata', {}))}
                </div>
            </div>
            
            <div class="section">
                <h2>Data Statistics</h2>
                <div class="statistics">
                    {self._format_statistics_html(results.get('statistics', {}))}
                </div>
            </div>
            
            <div class="section">
                <h2>Visualizations</h2>
                <p>Visualizations have been saved as separate image files.</p>
            </div>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        logger.info(f"Saved HTML report to {output_path}")
    
    def _export_markdown_report(self, results: Dict[str, Any], output_path: Path) -> None:
        """Export Markdown report."""
        md_content = f"""# Single-Cell Feature Analysis Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Analysis Summary

{self._format_metadata_markdown(results.get('metadata', {}))}

## Data Statistics

{self._format_statistics_markdown(results.get('statistics', {}))}

## Visualizations

Visualizations have been saved as separate image files in the output directory.
"""
        
        with open(output_path, 'w') as f:
            f.write(md_content)
        logger.info(f"Saved Markdown report to {output_path}")
    
    def _format_metadata_html(self, metadata: Dict[str, Any]) -> str:
        """Format metadata for HTML output."""
        if not metadata:
            return "<p>No metadata available.</p>"
        
        html_items = []
        for key, value in metadata.items():
            html_items.append(f"<p><strong>{key}:</strong> {value}</p>")
        return "".join(html_items)
    
    def _format_statistics_html(self, statistics: Dict[str, Any]) -> str:
        """Format statistics for HTML output."""
        if not statistics:
            return "<p>No statistics available.</p>"
        
        html_items = []
        for key, value in statistics.items():
            html_items.append(f"""
                <div class="stat-box">
                    <h3>{key.replace('_', ' ').title()}</h3>
                    <p>{value}</p>
                </div>
            """)
        return "".join(html_items)
    
    def _format_metadata_markdown(self, metadata: Dict[str, Any]) -> str:
        """Format metadata for Markdown output."""
        if not metadata:
            return "No metadata available."
        
        md_items = []
        for key, value in metadata.items():
            md_items.append(f"- **{key}**: {value}")
        return "\n".join(md_items)
    
    def _format_statistics_markdown(self, statistics: Dict[str, Any]) -> str:
        """Format statistics for Markdown output."""
        if not statistics:
            return "No statistics available."
        
        md_items = []
        for key, value in statistics.items():
            md_items.append(f"- **{key.replace('_', ' ').title()}**: {value}")
        return "\n".join(md_items)


class TensorBoardExporter(BaseExporter):
    """
    Exports embeddings and metadata for TensorBoard Projector visualization.
    
    Creates the necessary files for interactive embedding visualization
    in TensorBoard's projector tool.
    """
    
    def __init__(self, max_points: Optional[int] = None):
        """
        Initialize TensorBoardExporter.
        
        Args:
            max_points: Maximum number of points to export (for performance)
        """
        self.max_points = max_points
    
    def export(self, results: Dict[str, Any], output_path: Path) -> None:
        """
        Export embeddings and metadata for TensorBoard.
        
        Args:
            results: Dictionary containing 'embeddings' and 'metadata'
            output_path: Base path for TensorBoard files
        """
        try:
            embeddings = results.get('embeddings')
            metadata = results.get('metadata', {})
            
            if embeddings is None:
                logger.warning("No embeddings found, skipping TensorBoard export")
                return
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert embeddings to appropriate format
            if pd and hasattr(embeddings, 'select_dtypes'):  # DataFrame-like
                embedding_data = embeddings.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).values
                metadata_df = embeddings.select_dtypes(exclude=['float64', 'float32', 'int64', 'int32'])
            else:
                embedding_data = embeddings
                metadata_df = None
            
            # Limit points if specified
            if self.max_points and len(embedding_data) > self.max_points:
                if pd:
                    indices = pd.Series(range(len(embedding_data))).sample(self.max_points).sort_values()
                    embedding_data = embedding_data[indices]
                    if metadata_df is not None:
                        metadata_df = metadata_df.iloc[indices]
            
            # Save embeddings
            embeddings_path = output_path.with_suffix('.tsv')
            if pd:
                pd.DataFrame(embedding_data).to_csv(embeddings_path, sep='\\t', header=False, index=False)
            else:
                # Fallback without pandas
                import csv
                with open(embeddings_path, 'w', newline='') as f:
                    writer = csv.writer(f, delimiter='\\t')
                    for row in embedding_data:
                        writer.writerow(row)
            logger.info(f"Saved embeddings to {embeddings_path}")
            
            # Save metadata if available
            if metadata_df is not None and not metadata_df.empty:
                metadata_path = output_path.with_suffix('.metadata.tsv')
                metadata_df.to_csv(metadata_path, sep='\\t', index=False)
                logger.info(f"Saved metadata to {metadata_path}")
            else:
                metadata_path = None
            
            # Create projector config
            config_path = output_path.parent / 'projector_config.pbtxt'
            self._create_projector_config(config_path, embeddings_path, metadata_path)
            
        except Exception as e:
            logger.error(f"Failed to export TensorBoard data: {e}")
            raise
    
    def _create_projector_config(self, config_path: Path, 
                               embeddings_path: Path, 
                               metadata_path: Optional[Path]) -> None:
        """Create TensorBoard projector configuration file."""
        config_content = f'''embeddings {{
  tensor_path: "{embeddings_path.name}"
  '''
        
        if metadata_path:
            config_content += f'metadata_path: "{metadata_path.name}"\\n'
        
        config_content += '}\\n'
        
        with open(config_path, 'w') as f:
            f.write(config_content)
        logger.info(f"Saved projector config to {config_path}")


# Factory function for creating exporters
def create_exporter(exporter_type: str, **kwargs) -> BaseExporter:
    """
    Factory function to create exporters.
    
    Args:
        exporter_type: Type of exporter ('image', 'data', 'report', 'tensorboard')
        **kwargs: Additional arguments for the specific exporter
    
    Returns:
        BaseExporter instance
    """
    exporters = {
        'image': ImageExporter,
        'data': DataExporter,
        'report': ReportExporter,
        'tensorboard': TensorBoardExporter
    }
    
    if exporter_type not in exporters:
        raise ValueError(f"Unknown exporter type: {exporter_type}. Available: {list(exporters.keys())}")
    
    return exporters[exporter_type](**kwargs)
