"""
Pixel-wise metrics for segmentation evaluation.

This module implements standard pixel-level segmentation metrics including:
- Dice Coefficient (F1-score for segmentation)
- Intersection over Union (IoU) / Jaccard Index
- Hausdorff Distance
- Average Surface Distance
- Sensitivity (Recall) and Specificity
- Precision and Accuracy
"""

from typing import Dict, List, Optional, Union, Tuple
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import distance_transform_edt
import logging

logger = logging.getLogger(__name__)


class PixelWiseMetrics:
    """
    Comprehensive pixel-wise segmentation evaluation metrics.
    """
    
    def __init__(self, spacing: Optional[Union[float, Tuple[float, ...]]] = None):
        """
        Initialize the pixel-wise metrics calculator.
        
        Args:
            spacing: Physical spacing between pixels (for distance-based metrics).
                    Can be a single value or tuple for different dimensions.
        """
        self.spacing = spacing
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics."""
        self.results = {
            'dice_scores': [],
            'iou_scores': [],
            'precision_scores': [],
            'recall_scores': [],
            'specificity_scores': [],
            'accuracy_scores': [],
            'hausdorff_distances': [],
            'avg_surface_distances': []
        }
    
    def evaluate_image_pair(
        self,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
        compute_distances: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate a single prediction-ground truth pair.
        
        Args:
            pred_mask: Predicted binary segmentation mask
            gt_mask: Ground truth binary segmentation mask
            compute_distances: Whether to compute distance-based metrics
            
        Returns:
            Dictionary containing metrics for this image pair
        """
        import time
        timings = {}
        # Ensure masks are binary
        t0 = time.time()
        pred_mask = (pred_mask > 0).astype(np.uint8)
        gt_mask = (gt_mask > 0).astype(np.uint8)
        timings['binarize_masks'] = time.time() - t0

        # Calculate basic metrics
        t0 = time.time()
        dice = self._calculate_dice(pred_mask, gt_mask)
        timings['calculate_dice'] = time.time() - t0

        t0 = time.time()
        iou = self._calculate_iou(pred_mask, gt_mask)
        timings['calculate_iou'] = time.time() - t0

        t0 = time.time()
        precision = self._calculate_precision(pred_mask, gt_mask)
        timings['calculate_precision'] = time.time() - t0

        t0 = time.time()
        recall = self._calculate_recall(pred_mask, gt_mask)
        timings['calculate_recall'] = time.time() - t0

        t0 = time.time()
        specificity = self._calculate_specificity(pred_mask, gt_mask)
        timings['calculate_specificity'] = time.time() - t0

        t0 = time.time()
        accuracy = self._calculate_accuracy(pred_mask, gt_mask)
        timings['calculate_accuracy'] = time.time() - t0

        image_results = {
            'dice': dice,
            'iou': iou,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'accuracy': accuracy
        }

        # Calculate distance-based metrics if requested
        if compute_distances:
            t0 = time.time()
            hausdorff_dist = self._calculate_hausdorff_distance(pred_mask, gt_mask)
            timings['calculate_hausdorff_distance'] = time.time() - t0

            t0 = time.time()
            avg_surface_dist = self._calculate_average_surface_distance(pred_mask, gt_mask)
            timings['calculate_avg_surface_distance'] = time.time() - t0

            image_results['hausdorff_distance'] = hausdorff_dist
            image_results['avg_surface_distance'] = avg_surface_dist

        # Accumulate results (do not include timings in accumulation)
        self._accumulate_results(image_results, compute_distances)

        # Add timings only to the returned dictionary, not to accumulated results
        image_results_with_timings = dict(image_results)
        image_results_with_timings['timings'] = timings
        return image_results_with_timings
    
    def _calculate_dice(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """
        Calculate Dice coefficient (F1-score for segmentation).
        
        Args:
            pred_mask: Predicted binary mask
            gt_mask: Ground truth binary mask
            
        Returns:
            Dice coefficient
        """
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        total = pred_mask.sum() + gt_mask.sum()
        
        if total == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return 2.0 * intersection / total
    
    def _calculate_iou(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """
        Calculate Intersection over Union (Jaccard Index).
        
        Args:
            pred_mask: Predicted binary mask
            gt_mask: Ground truth binary mask
            
        Returns:
            IoU score
        """
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return intersection / union
    
    def _calculate_precision(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """
        Calculate precision (positive predictive value).
        
        Args:
            pred_mask: Predicted binary mask
            gt_mask: Ground truth binary mask
            
        Returns:
            Precision score
        """
        true_positives = np.logical_and(pred_mask, gt_mask).sum()
        predicted_positives = pred_mask.sum()
        
        if predicted_positives == 0:
            return 1.0 if true_positives == 0 else 0.0
        
        return true_positives / predicted_positives
    
    def _calculate_recall(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """
        Calculate recall (sensitivity, true positive rate).
        
        Args:
            pred_mask: Predicted binary mask
            gt_mask: Ground truth binary mask
            
        Returns:
            Recall score
        """
        true_positives = np.logical_and(pred_mask, gt_mask).sum()
        actual_positives = gt_mask.sum()
        
        if actual_positives == 0:
            return 1.0 if true_positives == 0 else 0.0
        
        return true_positives / actual_positives
    
    def _calculate_specificity(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """
        Calculate specificity (true negative rate).
        
        Args:
            pred_mask: Predicted binary mask
            gt_mask: Ground truth binary mask
            
        Returns:
            Specificity score
        """
        true_negatives = np.logical_and(~pred_mask, ~gt_mask).sum()
        actual_negatives = (~gt_mask).sum()
        
        if actual_negatives == 0:
            return 1.0 if true_negatives == 0 else 0.0
        
        return true_negatives / actual_negatives
    
    def _calculate_accuracy(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """
        Calculate overall accuracy.
        
        Args:
            pred_mask: Predicted binary mask
            gt_mask: Ground truth binary mask
            
        Returns:
            Accuracy score
        """
        correct_predictions = (pred_mask == gt_mask).sum()
        total_pixels = pred_mask.size
        
        return correct_predictions / total_pixels
    
    def _calculate_hausdorff_distance(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """
        Calculate Hausdorff distance between segmentation boundaries.
        
        Args:
            pred_mask: Predicted binary mask
            gt_mask: Ground truth binary mask
            
        Returns:
            Hausdorff distance
        """
        # Get boundary points
        pred_boundary = self._get_boundary_points(pred_mask)
        gt_boundary = self._get_boundary_points(gt_mask)
        
        if len(pred_boundary) == 0 and len(gt_boundary) == 0:
            return 0.0
        
        if len(pred_boundary) == 0 or len(gt_boundary) == 0:
            # Return maximum possible distance in the image
            return np.sqrt(pred_mask.shape[0]**2 + pred_mask.shape[1]**2)
        
        # Calculate directed Hausdorff distances
        hausdorff_1 = directed_hausdorff(pred_boundary, gt_boundary)[0]
        hausdorff_2 = directed_hausdorff(gt_boundary, pred_boundary)[0]
        
        # Return maximum of the two directed distances
        hausdorff_dist = max(hausdorff_1, hausdorff_2)
        
        # Apply spacing if provided
        if self.spacing is not None:
            if isinstance(self.spacing, (tuple, list)):
                avg_spacing = np.mean(self.spacing)
            else:
                avg_spacing = self.spacing
            hausdorff_dist *= avg_spacing
        
        return float(hausdorff_dist)
    
    def _calculate_average_surface_distance(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """
        Calculate average symmetric surface distance.
        
        Args:
            pred_mask: Predicted binary mask
            gt_mask: Ground truth binary mask
            
        Returns:
            Average surface distance
        """
        # Get boundary points
        pred_boundary = self._get_boundary_points(pred_mask)
        gt_boundary = self._get_boundary_points(gt_mask)
        
        if len(pred_boundary) == 0 and len(gt_boundary) == 0:
            return 0.0
        
        if len(pred_boundary) == 0 or len(gt_boundary) == 0:
            # Return maximum possible distance in the image
            return np.sqrt(pred_mask.shape[0]**2 + pred_mask.shape[1]**2)
        
        # Calculate distances from pred boundary to gt boundary
        distances_1 = []
        for point in pred_boundary:
            min_dist = np.min(np.sqrt(np.sum((gt_boundary - point)**2, axis=1)))
            distances_1.append(min_dist)
        
        # Calculate distances from gt boundary to pred boundary
        distances_2 = []
        for point in gt_boundary:
            min_dist = np.min(np.sqrt(np.sum((pred_boundary - point)**2, axis=1)))
            distances_2.append(min_dist)
        
        # Calculate average
        all_distances = distances_1 + distances_2
        avg_dist = np.mean(all_distances)
        
        # Apply spacing if provided
        if self.spacing is not None:
            if isinstance(self.spacing, (tuple, list)):
                avg_spacing = np.mean(self.spacing)
            else:
                avg_spacing = self.spacing
            avg_dist *= avg_spacing
        
        return float(avg_dist)
    
    def _get_boundary_points(self, mask: np.ndarray) -> np.ndarray:
        """
        Extract boundary points from a binary mask.
        
        Args:
            mask: Binary mask
            
        Returns:
            Array of boundary point coordinates
        """
        from scipy.ndimage import binary_erosion
        
        if mask.sum() == 0:
            return np.array([]).reshape(0, 2)
        
        # Get boundary by subtracting erosion from original
        eroded = binary_erosion(mask)
        boundary = mask.astype(bool) & ~eroded
        
        # Get coordinates of boundary points
        boundary_points = np.array(np.where(boundary)).T
        
        return boundary_points
    
    def _accumulate_results(self, image_results: Dict[str, float], compute_distances: bool):
        """Accumulate results from a single image."""
        self.results['dice_scores'].append(image_results['dice'])
        self.results['iou_scores'].append(image_results['iou'])
        self.results['precision_scores'].append(image_results['precision'])
        self.results['recall_scores'].append(image_results['recall'])
        self.results['specificity_scores'].append(image_results['specificity'])
        self.results['accuracy_scores'].append(image_results['accuracy'])
        
        if compute_distances:
            self.results['hausdorff_distances'].append(image_results['hausdorff_distance'])
            self.results['avg_surface_distances'].append(image_results['avg_surface_distance'])
    
    def compute_final_metrics(self) -> Dict[str, float]:
        """
        Compute final aggregated metrics across all evaluated images.
        
        Returns:
            Dictionary containing final metrics
        """
        final_metrics = {}
        
        # Calculate means and standard deviations
        for metric_name, scores in self.results.items():
            if scores:  # Check if list is not empty
                metric_base = metric_name.replace('_scores', '').replace('_distances', '')
                final_metrics[f"mean_{metric_base}"] = np.mean(scores)
                final_metrics[f"std_{metric_base}"] = np.std(scores)
                final_metrics[f"median_{metric_base}"] = np.median(scores)
                final_metrics[f"min_{metric_base}"] = np.min(scores)
                final_metrics[f"max_{metric_base}"] = np.max(scores)
        
        return final_metrics
    
    def get_detailed_results(self) -> Dict[str, List[float]]:
        """
        Get detailed results including per-image metrics.
        
        Returns:
            Dictionary containing all accumulated results
        """
        return self.results.copy()


def calculate_dice_coefficient(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Calculate Dice coefficient for a single mask pair.
    
    Args:
        pred_mask: Predicted binary segmentation mask
        gt_mask: Ground truth binary segmentation mask
        
    Returns:
        Dice coefficient
    """
    metrics = PixelWiseMetrics()
    result = metrics.evaluate_image_pair(pred_mask, gt_mask, compute_distances=False)
    return result['dice']


def calculate_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Calculate IoU (Jaccard Index) for a single mask pair.
    
    Args:
        pred_mask: Predicted binary segmentation mask
        gt_mask: Ground truth binary segmentation mask
        
    Returns:
        IoU score
    """
    metrics = PixelWiseMetrics()
    result = metrics.evaluate_image_pair(pred_mask, gt_mask, compute_distances=False)
    return result['iou']


def calculate_batch_dice(pred_masks: List[np.ndarray], gt_masks: List[np.ndarray]) -> float:
    """
    Calculate mean Dice coefficient across multiple mask pairs.
    
    Args:
        pred_masks: List of predicted binary segmentation masks
        gt_masks: List of ground truth binary segmentation masks
        
    Returns:
        Mean Dice coefficient
    """
    if len(pred_masks) != len(gt_masks):
        raise ValueError("Number of prediction and ground truth masks must match")
    
    metrics = PixelWiseMetrics()
    
    for pred_mask, gt_mask in zip(pred_masks, gt_masks):
        metrics.evaluate_image_pair(pred_mask, gt_mask, compute_distances=False)
    
    final_metrics = metrics.compute_final_metrics()
    return final_metrics['mean_dice']


def calculate_batch_iou(pred_masks: List[np.ndarray], gt_masks: List[np.ndarray]) -> float:
    """
    Calculate mean IoU across multiple mask pairs.
    
    Args:
        pred_masks: List of predicted binary segmentation masks
        gt_masks: List of ground truth binary segmentation masks
        
    Returns:
        Mean IoU score
    """
    if len(pred_masks) != len(gt_masks):
        raise ValueError("Number of prediction and ground truth masks must match")
    
    metrics = PixelWiseMetrics()
    
    for pred_mask, gt_mask in zip(pred_masks, gt_masks):
        metrics.evaluate_image_pair(pred_mask, gt_mask, compute_distances=False)
    
    final_metrics = metrics.compute_final_metrics()
    return final_metrics['mean_iou']
