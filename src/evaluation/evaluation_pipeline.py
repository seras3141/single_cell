"""
Comprehensive evaluation pipeline for single-cell segmentation.

This module provides a unified interface for evaluating segmentation results
using both instance-level and pixel-level metrics.
"""

import os
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import json
import logging
from datetime import datetime
from tqdm import tqdm

from .instance_metrics import InstanceSegmentationMetrics
from .pixel_metrics import PixelWiseMetrics
from .file_matching import load_single_file, load_image_file, check_prediction_label_matching

logger = logging.getLogger(__name__)


class EvaluationPipeline:
    """
    Comprehensive evaluation pipeline for single-cell segmentation tasks.
    
    This class orchestrates both instance-level and pixel-level evaluation
    metrics and provides unified reporting capabilities.
    """
    
    def __init__(
        self,
        iou_thresholds: Optional[List[float]] = None,
        pixel_spacing: Optional[Union[float, Tuple[float, ...]]] = None,
        calculate_instance_metrics: bool = True,
        calculate_pixel_metrics: bool = True,
        compute_distances: bool = True
    ):
        """
        Initialize the evaluation pipeline.
        
        Args:
            iou_thresholds: List of IoU thresholds for instance evaluation
            pixel_spacing: Physical spacing between pixels for distance metrics
            compute_distances: Whether to compute distance-based metrics
        """
        self.instance_metrics = InstanceSegmentationMetrics(iou_thresholds) if calculate_instance_metrics else None
        self.pixel_metrics = PixelWiseMetrics(pixel_spacing) if calculate_pixel_metrics else None
        self.compute_distances = compute_distances
        
        self.results = {
            'image_results': [],
            'metadata': {
                'iou_thresholds': self.instance_metrics.iou_thresholds,
                'pixel_spacing': pixel_spacing,
                'compute_distances': compute_distances,
                'evaluation_date': datetime.now().isoformat()
            }
        }
    
    def evaluate_single_image(
        self,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
        image_id: Optional[str] = None,
        # metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single prediction-ground truth pair.
        
        Args:
            pred_mask: Predicted segmentation mask
            gt_mask: Ground truth segmentation mask
            image_id: Optional identifier for the image
            metadata: Optional metadata about the image
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        logger.info(f"Evaluating image: {image_id or 'unknown'}")
        
        # Validate input shapes
        if pred_mask.shape != gt_mask.shape:
            raise ValueError(f"Shape mismatch: pred {pred_mask.shape} vs gt {gt_mask.shape}")
        
        # Initialize result dictionary
        image_result = {
            'image_id': image_id,
            # 'metadata': metadata or {},
            'shape': pred_mask.shape
        }
        
        # Evaluate instance-level metrics
        if self.instance_metrics is not None:
            try:
                instance_results = self.instance_metrics.evaluate_image_pair(pred_mask, gt_mask)
                image_result['instance_metrics'] = instance_results
                logger.debug(f"Instance metrics computed for {image_id}")
            except Exception as e:
                logger.error(f"Failed to compute instance metrics for {image_id}: {e}")
                image_result['instance_metrics'] = {}
        
        # Evaluate pixel-level metrics
        if self.pixel_metrics is not None:
            try:
                # For pixel metrics, we need binary masks
                pred_binary = (pred_mask > 0).astype(np.uint8)
                gt_binary = (gt_mask > 0).astype(np.uint8)
                
                pixel_results = self.pixel_metrics.evaluate_image_pair(
                    pred_binary, gt_binary, self.compute_distances
                )
                image_result['pixel_metrics'] = pixel_results
                logger.debug(f"Pixel metrics computed for {image_id}")
            except Exception as e:
                logger.error(f"Failed to compute pixel metrics for {image_id}: {e}")
                image_result['pixel_metrics'] = {}
        
        # Store result : TODO : Move this to calling function (to avoid parallel issues)
        # self.results['image_results'].append(image_result)
        
        return image_result

    # TODO : Remove this function later    
    def evaluate_batch(
        self,
        pred_masks: List[np.ndarray],
        gt_masks: List[np.ndarray],
        image_ids: Optional[List[str]] = None,
        # metadata_list: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluate multiple prediction-ground truth pairs.
        
        Args:
            pred_masks: List of predicted segmentation masks
            gt_masks: List of ground truth segmentation masks
            image_ids: Optional list of image identifiers
            metadata_list: Optional list of metadata dictionaries
            
        Returns:
            List of evaluation results for each image
        """
        if len(pred_masks) != len(gt_masks):
            raise ValueError("Number of prediction and ground truth masks must match")
        
        # Prepare optional arguments
        if image_ids is None:
            image_ids = [f"image_{i:04d}" for i in range(len(pred_masks))]
        elif len(image_ids) != len(pred_masks):
            raise ValueError("Number of image IDs must match number of masks")
        
        # if metadata_list is None:
        #     metadata_list = [{}] * len(pred_masks)
        # elif len(metadata_list) != len(pred_masks):
        #     raise ValueError("Number of metadata entries must match number of masks")
        
        logger.info(f"Evaluating batch of {len(pred_masks)} images")
        
        batch_results = []
        for i, (pred_mask, gt_mask, img_id) in enumerate(
            tqdm(zip(pred_masks, gt_masks, image_ids), total=len(pred_masks), desc="Evaluating images")
        ):
            try:
                result = self.evaluate_single_image(pred_mask, gt_mask, img_id)
                batch_results.append(result)
            except Exception as e:
                logger.error(f"Failed to evaluate image {img_id}: {e}")
                # Store failed result with error information
                batch_results.append({
                    'image_id': img_id,
                    # 'metadata': metadata,
                    'error': str(e),
                    'instance_metrics': {},
                    'pixel_metrics': {}
                })

        self.results['image_results'].extend(batch_results)
        
        return batch_results

    
    def evaluate_batch_path(
        self,
        pred_masks: List[str],
        gt_masks: List[str],
        image_ids: Optional[List[str]] = None,
        n_jobs: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluate multiple prediction-ground truth pairs.
        
        Args:
            pred_masks: List of predicted segmentation paths
            gt_masks: List of ground truth segmentation paths
            image_ids: Optional list of image identifiers
            
        Returns:
            List of evaluation results for each image
        """
        if len(pred_masks) != len(gt_masks):
            raise ValueError("Number of prediction and ground truth masks must match")
        
        # Prepare optional arguments
        if image_ids is None:
            image_ids = [f"image_{i:04d}" for i in range(len(pred_masks))]
        elif len(image_ids) != len(pred_masks):
            raise ValueError("Number of image IDs must match number of masks")

        # Respect SLURM environment variable if available
        if n_jobs is None:
            n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
        logger.info(f"Evaluating batch of {len(pred_masks)} images with {n_jobs} Jobs")
        
        batch_results = []

        def _process_single(pred_path, gt_path, img_id):
            try:
                pred = load_single_file(pred_path)
                gt = load_single_file(gt_path)
                results = self.evaluate_single_image(pred, gt, img_id)
            except Exception as e:
                logger.error(f"Failed to evaluate image {img_id}: {e}")
                results = {
                    'image_id': img_id,
                    # 'metadata': metadata,
                    'error': str(e),
                    'instance_metrics': {},
                    'pixel_metrics': {}
                }
            return results

        batch_results = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(_process_single)(p, g, i)
            for p, g, i in tqdm(zip(pred_masks, gt_masks, image_ids), total=len(pred_masks))
        )

        # Aggregate all results into self.results
        self.results['image_results'].extend(batch_results)

        return batch_results #type: ignore

    
    def compute_summary_metrics(self) -> Dict[str, Any]:
        """
        Compute summary metrics across all evaluated images.
        
        Returns:
            Dictionary containing aggregated metrics
        """
        if not self.results['image_results']:
            logger.warning("No evaluation results available")
            return {}
        
        logger.info("Computing summary metrics")
        
        # Get final metrics from individual metric classes
        instance_summary = self.instance_metrics.compute_final_metrics() if self.instance_metrics else {}
        pixel_summary = self.pixel_metrics.compute_final_metrics() if self.pixel_metrics else {}

        # Combine summaries
        summary = {
            'num_images_evaluated': len(self.results['image_results']),
            'instance_metrics': instance_summary,
            'pixel_metrics': pixel_summary
        }
        
        # Add some additional aggregate statistics
        summary['aggregate_stats'] = self._compute_aggregate_stats()
        
        return summary
    
    def _compute_aggregate_stats(self) -> Dict[str, Any]:
        """Compute additional aggregate statistics."""
        stats = {}
        
        # Collect metrics from all images
        dice_scores = []
        iou_scores = []
        f1_scores = []
        
        for result in self.results['image_results']:
            if 'pixel_metrics' in result and 'dice' in result['pixel_metrics']:
                dice_scores.append(result['pixel_metrics']['dice'])
            if 'pixel_metrics' in result and 'iou' in result['pixel_metrics']:
                iou_scores.append(result['pixel_metrics']['iou'])
            if 'instance_metrics' in result:
                # Get F1 at 0.5 threshold
                f1_key = 'f1_0.50'
                if f1_key in result['instance_metrics']:
                    f1_scores.append(result['instance_metrics'][f1_key])
        
        # Calculate percentiles
        if dice_scores:
            stats['dice_percentiles'] = {
                '25th': float(np.percentile(dice_scores, 25)),
                '50th': float(np.percentile(dice_scores, 50)),
                '75th': float(np.percentile(dice_scores, 75)),
                '90th': float(np.percentile(dice_scores, 90))
            }
        
        if iou_scores:
            stats['iou_percentiles'] = {
                '25th': float(np.percentile(iou_scores, 25)),
                '50th': float(np.percentile(iou_scores, 50)),
                '75th': float(np.percentile(iou_scores, 75)),
                '90th': float(np.percentile(iou_scores, 90))
            }
        
        if f1_scores:
            stats['f1_percentiles'] = {
                '25th': float(np.percentile(f1_scores, 25)),
                '50th': float(np.percentile(f1_scores, 50)),
                '75th': float(np.percentile(f1_scores, 75)),
                '90th': float(np.percentile(f1_scores, 90))
            }
        
        return stats
    
    def export_results(
        self,
        output_path: Union[str, Path],
        format: str = 'json',
        include_detailed: bool = True
    ) -> None:
        """
        Export evaluation results to file.
        
        Args:
            output_path: Path to save the results
            format: Output format ('json', 'csv', 'xlsx')
            include_detailed: Whether to include detailed per-image results
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for export
        export_data = {
            'metadata': self.results['metadata'],
            'summary_metrics': self.compute_summary_metrics()
        }
        
        if include_detailed:
            export_data['detailed_results'] = self.results['image_results']
        
        if format.lower() == 'json':
            self._export_json(export_data, output_path)
        elif format.lower() == 'csv':
            self._export_csv(export_data, output_path)
        elif format.lower() == 'xlsx':
            self._export_xlsx(export_data, output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Results exported to {output_path}")
    
    def _export_json(self, data: Dict[str, Any], output_path: Path) -> None:
        """Export results as JSON."""
        with open(output_path.with_suffix('.json'), 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def _export_csv(self, data: Dict[str, Any], output_path: Path) -> None:
        """Export results as CSV (summary only)."""
        if 'detailed_results' not in data:
            logger.warning("No detailed results available for CSV export")
            return
        
        # Flatten detailed results for CSV
        rows = []
        for result in data['detailed_results']:
            row = {
                'image_id': result.get('image_id', ''),
                'shape_height': result.get('shape', [0, 0])[0],
                'shape_width': result.get('shape', [0, 0])[1] if len(result.get('shape', [])) > 1 else 0,
            }
            
            # Add pixel metrics
            pixel_metrics = result.get('pixel_metrics', {})
            for key, value in pixel_metrics.items():
                row[f'pixel_{key}'] = value
            
            # Add instance metrics
            instance_metrics = result.get('instance_metrics', {})
            for key, value in instance_metrics.items():
                row[f'instance_{key}'] = value
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path.with_suffix('.csv'), index=False)
    
    def _export_xlsx(self, data: Dict[str, Any], output_path: Path) -> None:
        """Export results as Excel file with multiple sheets."""
        with pd.ExcelWriter(output_path.with_suffix('.xlsx'), engine='openpyxl') as writer:
            # Summary sheet
            summary_data = []
            for category, metrics in data['summary_metrics'].items():
                if isinstance(metrics, dict):
                    for metric, value in metrics.items():
                        summary_data.append({
                            'Category': category,
                            'Metric': metric,
                            'Value': value
                        })
                else:
                    summary_data.append({
                        'Category': 'General',
                        'Metric': category,
                        'Value': metrics
                    })
            
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            # Detailed results sheet (if available)
            if 'detailed_results' in data:
                # Flatten for Excel
                rows = []
                for result in data['detailed_results']:
                    row = {'image_id': result.get('image_id', '')}
                    
                    # Add all metrics
                    pixel_metrics = result.get('pixel_metrics', {})
                    instance_metrics = result.get('instance_metrics', {})
                    
                    row.update({f'pixel_{k}': v for k, v in pixel_metrics.items()})
                    row.update({f'instance_{k}': v for k, v in instance_metrics.items()})
                    
                    rows.append(row)
                
                pd.DataFrame(rows).to_excel(writer, sheet_name='Detailed Results', index=False)
    
    def reset(self) -> None:
        """Reset all evaluation results."""
        if self.instance_metrics is not None:
            self.instance_metrics.reset()
        if self.pixel_metrics is not None:
            self.pixel_metrics.reset()
        self.results['image_results'].clear()
        logger.info("Evaluation pipeline reset")
    
    def get_results(self) -> Dict[str, Any]:
        """
        Get all evaluation results.
        
        Returns:
            Dictionary containing all results
        """
        return self.results.copy()
    
    def plot_results(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        summary_metrics: Optional[Dict[str, Any]] = None,
        show: bool = True,
        plot_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate plots for evaluation results.
        
        Args:
            output_dir: Directory to save plots
            show: Whether to display plots
            plot_types: List of plot types to generate. Options: 
                       ['iou_metrics', 'ap_curve', 'dashboard', 'all']
            
        Returns:
            Dictionary of generated figures
        """
        try:
            from .plotting import EvaluationPlotter
        except ImportError as e:
            logger.error(f"Failed to import plotting module: {e}")
            logger.error("Install matplotlib to use plotting functionality: pip install matplotlib")
            return {}
        
        if not self.results['image_results']:
            logger.warning("No evaluation results available for plotting")
            return {}
        
        if plot_types is None:
            plot_types = ['all']
        
        plotter = EvaluationPlotter()
        figures = {}
        
        # Compute summary metrics if not already done
        if summary_metrics is None:
            summary_metrics = self.compute_summary_metrics()
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate requested plots
        if 'iou_metrics' in plot_types or 'all' in plot_types:
            instance_metrics = summary_metrics.get('instance_metrics', {})
            if instance_metrics:
                fig = plotter.plot_metrics_vs_iou(
                    instance_metrics,
                    output_path=output_dir / 'metrics_vs_iou.png' if output_dir else None,
                    show=show
                )
                figures['iou_metrics'] = fig
        
        if 'ap_curve' in plot_types or 'all' in plot_types:
            if self.results['image_results']:
                fig = plotter.plot_average_precision_curve(
                    self.results['image_results'],
                    output_path=output_dir / 'ap_curve.png' if output_dir else None,
                    show=show
                )
                figures['ap_curve'] = fig
        
        if 'dashboard' in plot_types or 'all' in plot_types:
            fig = plotter.plot_comprehensive_metrics_dashboard(
                summary_metrics,
                self.results['image_results'] if self.results['image_results'] else None,
                output_path=output_dir / 'evaluation_dashboard.png' if output_dir else None,
                show=show
            )
            figures['dashboard'] = fig
        
        if output_dir and figures:
            logger.info(f"Saved {len(figures)} plots to {output_dir}")
        
        return figures

def evaluate_segmentation_performance(
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
    output_dir: Optional[Union[str, Path]] = None,
    output_format: List[str] = ['csv'],
    image_ids: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Convenience function for comprehensive segmentation evaluation.
    
    Args:
        pred_masks: List of predicted segmentation masks
        gt_masks: List of ground truth segmentation masks
        output_dir: Optional directory to save results
        image_ids: Optional list of image identifiers
        
    Returns:
        Dictionary containing evaluation results
    """
    # Initialize pipeline
    pipeline = EvaluationPipeline()
    
    # Evaluate batch
    pipeline.evaluate_batch(pred_masks, gt_masks, image_ids)
    
    # Compute summary
    summary = pipeline.compute_summary_metrics()
    
    # Export results if output directory is provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export in multiple formats
        if 'json' in output_format:
            pipeline.export_results(output_dir / 'evaluation_results.json', 'json')
        if 'csv' in output_format:
            pipeline.export_results(output_dir / 'evaluation_results.csv', 'csv')
        if 'xlsx' in output_format:
            pipeline.export_results(output_dir / 'evaluation_results.xlsx', 'xlsx')

        logger.info(f"Evaluation results saved to {output_dir}")
    
    return {
        'summary': summary,
        'detailed_results': pipeline.get_results()['image_results']
    }
