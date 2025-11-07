"""
Plotting utilities for evaluation metrics visualization.

This module provides functions to visualize evaluation metrics across different
IoU thresholds and create comprehensive performance plots.
"""

from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from pathlib import Path
import logging
import pandas as pd

try:
    import seaborn as sns
except ImportError:
    sns = None

logger = logging.getLogger(__name__)


class EvaluationPlotter:
    """
    Visualization utilities for evaluation metrics.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the evaluation plotter.
        
        Args:
            style: Matplotlib style to use
            figsize: Default figure size
        """
        self.figsize = figsize
        try:
            plt.style.use(style)
        except:
            # Fallback to a basic style if seaborn is not available
            plt.style.use('default')
            logger.warning(f"Style '{style}' not available, using default")
    
    def plot_metrics_vs_iou(
        self,
        metrics_data: Dict[str, Any],
        output_path: Optional[Union[str, Path]] = None,
        show: bool = True
    ) -> Figure:
        """
        Plot precision, recall, and F1 score against IoU thresholds.
        
        Args:
            metrics_data: Results from EvaluationPipeline.compute_final_metrics()
            output_path: Optional path to save the plot
            show: Whether to display the plot
            
        Returns:
            matplotlib Figure object
        """
        # Extract IoU thresholds and corresponding metrics
        iou_thresholds = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        # Parse the metrics data to extract threshold-specific values
        for key, value in metrics_data.items():
            if key.startswith('precision_@'):
                threshold = float(key.split('@')[1])
                iou_thresholds.append(threshold)
                precision_scores.append(value)
            elif key.startswith('recall_@'):
                threshold = float(key.split('@')[1])
                recall_key_index = len([k for k in metrics_data.keys() if k.startswith('recall_@') and float(k.split('@')[1]) <= threshold]) - 1
                if recall_key_index < len(recall_scores):
                    continue
                recall_scores.append(value)
            elif key.startswith('f1_@'):
                threshold = float(key.split('@')[1])
                f1_key_index = len([k for k in metrics_data.keys() if k.startswith('f1_@') and float(k.split('@')[1]) <= threshold]) - 1
                if f1_key_index < len(f1_scores):
                    continue
                f1_scores.append(value)
        
        # Sort by IoU threshold
        if iou_thresholds:
            sorted_data = sorted(zip(iou_thresholds, precision_scores[:len(iou_thresholds)], 
                                   recall_scores[:len(iou_thresholds)], f1_scores[:len(iou_thresholds)]))
            iou_thresholds, precision_scores, recall_scores, f1_scores = zip(*sorted_data)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if iou_thresholds:
            ax.plot(iou_thresholds, precision_scores, 'o-', label='Precision', linewidth=2, markersize=6)
            ax.plot(iou_thresholds, recall_scores, 's-', label='Recall', linewidth=2, markersize=6)
            ax.plot(iou_thresholds, f1_scores, '^-', label='F1 Score', linewidth=2, markersize=6)
        
        ax.set_xlabel('IoU Threshold', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Instance Segmentation Metrics vs IoU Threshold', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        
        # Add some styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        # Save if requested
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved IoU threshold plot to {output_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_average_precision_curve(
        self,
        detailed_results: List[Dict[str, Any]],
        output_path: Optional[Union[str, Path]] = None,
        show: bool = True
    ) -> Figure:
        """
        Plot Average Precision curve across IoU thresholds.
        
        Args:
            detailed_results: Detailed results from evaluation pipeline
            output_path: Optional path to save the plot
            show: Whether to display the plot
            
        Returns:
            matplotlib Figure object
        """
        # Extract IoU thresholds and AP values
        if not detailed_results:
            logger.warning("No detailed results provided for AP curve plotting")
            return plt.figure()
        
        # Get IoU thresholds from the first result
        first_result = detailed_results[0]
        instance_metrics = first_result.get('instance_metrics', {})
        
        # Extract thresholds and AP values
        iou_thresholds = []
        ap_values = []
        
        for key, value in instance_metrics.items():
            if key.startswith('ap_') and '@' not in key:
                # Extract threshold from key (e.g., 'ap_0.50' -> 0.50)
                try:
                    threshold_str = key.split('_')[1]
                    threshold = float(threshold_str)
                    iou_thresholds.append(threshold)
                    ap_values.append(value)
                except (IndexError, ValueError):
                    continue
        
        # Sort by threshold
        if iou_thresholds:
            sorted_data = sorted(zip(iou_thresholds, ap_values))
            iou_thresholds, ap_values = zip(*sorted_data)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if iou_thresholds:
            ax.plot(iou_thresholds, ap_values, 'o-', linewidth=3, markersize=8, 
                   color='darkblue', label='Average Precision')
            ax.fill_between(iou_thresholds, ap_values, alpha=0.3, color='lightblue')
        
        ax.set_xlabel('IoU Threshold', fontsize=12)
        ax.set_ylabel('Average Precision', fontsize=12)
        ax.set_title('Average Precision vs IoU Threshold', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        
        # Only show legend if there are labeled artists
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=11)
        
        # Add mAP annotation
        if ap_values:
            mean_ap = np.mean(ap_values)
            ax.axhline(y=float(mean_ap), color='red', linestyle='--', alpha=0.7)
            ax.text(0.02, 0.98, f'mAP = {mean_ap:.3f}', transform=ax.transAxes, 
                   fontsize=11, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        # Save if requested
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved AP curve to {output_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_comprehensive_metrics_dashboard(
        self,
        summary_metrics: Dict[str, Any],
        detailed_results: Optional[List[Dict[str, Any]]] = None,
        output_path: Optional[Union[str, Path]] = None,
        show: bool = True
    ) -> Figure:
        """
        Create a comprehensive dashboard with multiple metric visualizations.
        
        Args:
            summary_metrics: Summary metrics from evaluation pipeline
            detailed_results: Optional detailed results for additional plots
            output_path: Optional path to save the plot
            show: Whether to display the plot
            
        Returns:
            matplotlib Figure object
        """
        fig = plt.figure(figsize=(16, 12))
        
        # Create subplots
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. IoU Threshold vs Metrics (top left, spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_iou_metrics_subplot(ax1, summary_metrics)
        
        # 2. Overall Performance Summary (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_summary_metrics_subplot(ax2, summary_metrics)
        
        # 3. Pixel vs Instance Metrics Comparison (middle left)
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_pixel_vs_instance_subplot(ax3, summary_metrics)
        
        # 4. Distribution of Scores (middle center)
        ax4 = fig.add_subplot(gs[1, 1])
        if detailed_results:
            self._plot_score_distribution_subplot(ax4, detailed_results)
        else:
            ax4.text(0.5, 0.5, 'No detailed results\navailable', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Score Distribution')
        
        # 5. Metric Correlation Heatmap (middle right)
        ax5 = fig.add_subplot(gs[1, 2])
        if detailed_results:
            self._plot_correlation_heatmap_subplot(ax5, detailed_results)
        else:
            ax5.text(0.5, 0.5, 'No detailed results\navailable', 
                    ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Metric Correlations')
        
        # 6. Per-image Performance (bottom, spans all columns)
        ax6 = fig.add_subplot(gs[2, :])
        if detailed_results:
            self._plot_per_image_performance_subplot(ax6, detailed_results)
        else:
            ax6.text(0.5, 0.5, 'No detailed results available', 
                    ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Per-Image Performance')
        
        plt.suptitle('Single-Cell Segmentation Evaluation Dashboard', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        # Save if requested
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved comprehensive dashboard to {output_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def _plot_iou_metrics_subplot(self, ax: Axes, metrics_data: Dict[str, Any]):
        """Helper method to plot IoU threshold metrics in a subplot."""
        # Extract threshold-based metrics
        thresholds = []
        precision_vals = []
        recall_vals = []
        f1_vals = []
        
        for key, value in metrics_data.get('instance_metrics', {}).items():
            if key.startswith('precision_@'):
                threshold = float(key.split('@')[1])
                thresholds.append(threshold)
                precision_vals.append(value)
            elif key.startswith('recall_@'):
                threshold = float(key.split('@')[1])
                if threshold in thresholds:
                    idx = thresholds.index(threshold)
                    if idx < len(recall_vals):
                        continue
                    recall_vals.append(value)
            elif key.startswith('f1_@'):
                threshold = float(key.split('@')[1])
                if threshold in thresholds:
                    idx = thresholds.index(threshold)
                    if idx < len(f1_vals):
                        continue
                    f1_vals.append(value)
        
        if thresholds:
            # Sort by threshold
            sorted_data = sorted(zip(thresholds, precision_vals[:len(thresholds)], 
                                   recall_vals[:len(thresholds)], f1_vals[:len(thresholds)]))
            thresholds, precision_vals, recall_vals, f1_vals = zip(*sorted_data)
            
            ax.plot(thresholds, precision_vals, 'o-', label='Precision', linewidth=2)
            ax.plot(thresholds, recall_vals, 's-', label='Recall', linewidth=2)
            ax.plot(thresholds, f1_vals, '^-', label='F1 Score', linewidth=2)
        
        ax.set_xlabel('IoU Threshold')
        ax.set_ylabel('Score')
        ax.set_title('Metrics vs IoU Threshold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
    
    def _plot_summary_metrics_subplot(self, ax: Axes, metrics_data: Dict[str, Any]):
        """Helper method to plot summary metrics in a subplot."""
        instance_metrics = metrics_data.get('instance_metrics', {})
        pixel_metrics = metrics_data.get('pixel_metrics', {})
        
        # Key metrics to display
        key_metrics = {
            'mAP': instance_metrics.get('mAP', 0),
            'Mean F1': instance_metrics.get('mean_F1', 0),
            'Mean AJI': instance_metrics.get('mean_AJI', 0),
            'Mean Dice': pixel_metrics.get('mean_dice', 0),
            'Mean IoU': pixel_metrics.get('mean_iou', 0)
        }
        
        metrics = list(key_metrics.keys())
        values = list(key_metrics.values())
        import matplotlib.cm as cm
        colors = cm.get_cmap('tab10')(np.linspace(0, 1, len(metrics)))
        
        bars = ax.barh(metrics, values, color=colors)
        ax.set_xlim(0, 1)
        ax.set_title('Key Performance Metrics')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{value:.3f}', va='center', fontsize=9)
    
    def _plot_pixel_vs_instance_subplot(self, ax: Axes, metrics_data: Dict[str, Any]):
        """Helper method to compare pixel vs instance metrics."""
        instance_metrics = metrics_data.get('instance_metrics', {})
        pixel_metrics = metrics_data.get('pixel_metrics', {})
        
        # Compare similar metrics
        comparisons = [
            ('Precision', pixel_metrics.get('mean_precision', 0), 
             instance_metrics.get('precision_@0.50', 0)),
            ('Recall', pixel_metrics.get('mean_recall', 0), 
             instance_metrics.get('recall_@0.50', 0)),
            ('F1/Dice', pixel_metrics.get('mean_dice', 0), 
             instance_metrics.get('f1_@0.50', 0))
        ]
        
        metrics = [comp[0] for comp in comparisons]
        pixel_vals = [comp[1] for comp in comparisons]
        instance_vals = [comp[2] for comp in comparisons]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax.bar(x - width/2, pixel_vals, width, label='Pixel-wise', alpha=0.8)
        ax.bar(x + width/2, instance_vals, width, label='Instance-wise', alpha=0.8)
        
        ax.set_xlabel('Metric Type')
        ax.set_ylabel('Score')
        ax.set_title('Pixel vs Instance Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.set_ylim(0, 1.05)
    
    def _plot_score_distribution_subplot(self, ax: Axes, detailed_results: List[Dict[str, Any]]):
        """Helper method to plot score distributions."""
        dice_scores = []
        f1_scores = []
        
        for result in detailed_results:
            pixel_metrics = result.get('pixel_metrics', {})
            instance_metrics = result.get('instance_metrics', {})
            
            if 'dice' in pixel_metrics:
                dice_scores.append(pixel_metrics['dice'])
            if 'f1_0.50' in instance_metrics:
                f1_scores.append(instance_metrics['f1_0.50'])
        
        if dice_scores:
            ax.hist(dice_scores, alpha=0.7, label='Dice Coefficient', bins=10)
        if f1_scores:
            ax.hist(f1_scores, alpha=0.7, label='F1@0.5', bins=10)
        
        ax.set_xlabel('Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Score Distribution')
        ax.legend()
    
    def _plot_correlation_heatmap_subplot(self, ax: Axes, detailed_results: List[Dict[str, Any]]):
        """Helper method to plot metric correlation heatmap."""
        # Collect metrics from all images
        metrics_df = []
        
        for result in detailed_results:
            row = {}
            pixel_metrics = result.get('pixel_metrics', {})
            instance_metrics = result.get('instance_metrics', {})
            
            # Add key metrics
            if 'dice' in pixel_metrics:
                row['Dice'] = pixel_metrics['dice']
            if 'iou' in pixel_metrics:
                row['IoU'] = pixel_metrics['iou']
            if 'precision' in pixel_metrics:
                row['Pixel Precision'] = pixel_metrics['precision']
            if 'f1_0.50' in instance_metrics:
                row['Instance F1'] = instance_metrics['f1_0.50']
            if 'aji' in instance_metrics:
                row['AJI'] = instance_metrics['aji']
            
            if row:  # Only add if we have some metrics
                metrics_df.append(row)
        
        if metrics_df:
            df = pd.DataFrame(metrics_df)
            correlation_matrix = df.corr()
            
            im = ax.imshow(correlation_matrix.values, cmap='coolwarm', vmin=-1, vmax=1)
            ax.set_xticks(range(len(correlation_matrix.columns)))
            ax.set_yticks(range(len(correlation_matrix.columns)))
            ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha='right')
            ax.set_yticklabels(correlation_matrix.columns)
            
            # Add correlation values as text
            for i in range(len(correlation_matrix.columns)):
                for j in range(len(correlation_matrix.columns)):
                    text = ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title('Metric Correlations')
    
    def _plot_per_image_performance_subplot(self, ax: Axes, detailed_results: List[Dict[str, Any]]):
        """Helper method to plot per-image performance."""
        image_ids = []
        dice_scores = []
        f1_scores = []
        
        for result in detailed_results:
            image_ids.append(result.get('image_id', 'Unknown'))
            
            pixel_metrics = result.get('pixel_metrics', {})
            instance_metrics = result.get('instance_metrics', {})
            
            dice_scores.append(pixel_metrics.get('dice', 0))
            f1_scores.append(instance_metrics.get('f1_0.50', 0))
        
        x = range(len(image_ids))
        ax.plot(x, dice_scores, 'o-', label='Dice Coefficient', alpha=0.8)
        ax.plot(x, f1_scores, 's-', label='F1@0.5', alpha=0.8)
        
        ax.set_xlabel('Image Index')
        ax.set_ylabel('Score')
        ax.set_title('Per-Image Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        
        # Show only every nth label to avoid crowding
        step = max(1, len(image_ids) // 10)
        ax.set_xticks(x[::step])
        ax.set_xticklabels([image_ids[i] for i in x[::step]], rotation=45, ha='right')


def plot_evaluation_results(
    results: Dict[str, Any],
    output_dir: Optional[Union[str, Path]] = None,
    show: bool = True
) -> Dict[str, Figure]:
    """
    Convenience function to plot all evaluation results.
    
    Args:
        results: Results from EvaluationPipeline
        output_dir: Optional directory to save plots
        show: Whether to display plots
        
    Returns:
        Dictionary of figure names to matplotlib Figure objects
    """
    plotter = EvaluationPlotter()
    figures = {}
    
    # Extract relevant data
    summary_metrics = results.get('summary', {})
    detailed_results = results.get('detailed_results', [])
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. IoU threshold plot
    instance_metrics = summary_metrics.get('instance_metrics', {})
    if instance_metrics:
        fig1 = plotter.plot_metrics_vs_iou(
            instance_metrics,
            output_path=output_dir / 'metrics_vs_iou.png' if output_dir else None,
            show=show
        )
        figures['metrics_vs_iou'] = fig1
    
    # 2. AP curve
    if detailed_results:
        fig2 = plotter.plot_average_precision_curve(
            detailed_results,
            output_path=output_dir / 'ap_curve.png' if output_dir else None,
            show=show
        )
        figures['ap_curve'] = fig2
    
    # 3. Comprehensive dashboard
    fig3 = plotter.plot_comprehensive_metrics_dashboard(
        summary_metrics,
        detailed_results,
        output_path=output_dir / 'evaluation_dashboard.png' if output_dir else None,
        show=show
    )
    figures['dashboard'] = fig3
    
    if output_dir:
        logger.info(f"Saved {len(figures)} plots to {output_dir}")
    
    return figures
