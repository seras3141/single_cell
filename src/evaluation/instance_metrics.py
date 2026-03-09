"""
Instance segmentation metrics for single-cell analysis.

This module implements standard instance segmentation evaluation metrics including:
- Average Precision (AP) and mean Average Precision (mAP)
- F1 Score
- Precision and Recall
- Aggregated Jaccard Index (AJI)
- Panoptic Quality (PQ)
"""

from joblib import Parallel, delayed

from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import logging
import time

logger = logging.getLogger(__name__)


class InstanceSegmentationMetrics:
    """
    Comprehensive instance segmentation evaluation metrics.
    """
    
    def __init__(self, iou_thresholds: Optional[List[float]] = None):
        """
        Initialize the instance segmentation metrics calculator.
        
        Args:
            iou_thresholds: List of IoU thresholds for evaluation.
                          Default is [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        """
        if iou_thresholds is None:
            self.iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        else:
            self.iou_thresholds = sorted(iou_thresholds)
        
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics."""
        self.results = {
            'ap_per_threshold': {},
            'precision_per_threshold': {},
            'recall_per_threshold': {},
            'f1_per_threshold': {},
            'tp_per_threshold': {},
            'fp_per_threshold': {},
            'fn_per_threshold': {},
            'aji_scores': [],
            'pq_scores': []
        }

    def compute_metrics_for_threshold(
        self,
        iou_matrix: np.ndarray,
        threshold: float
    ) -> Dict[str, float]:
        """
        Compute metrics for a single IoU threshold.
        
        Args:
            pred_mask: Predicted instance segmentation mask (H, W) with instance IDs
            gt_mask: Ground truth instance segmentation mask (H, W) with instance IDs
            threshold: IoU threshold for evaluation

        Returns:
            Dictionary containing metrics for this threshold
        """
        timings = {}
        image_results = {}

        t0 = time.time()
        tp, fp, fn, matches = self._match_instances_optimized(iou_matrix, threshold)
        timings[f'match_instances_{threshold:.2f}'] = time.time() - t0

        t0 = time.time()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        timings[f'precision_recall_f1_{threshold:.2f}'] = time.time() - t0

        threshold_key = f"{threshold:.2f}"
        image_results[f"precision_{threshold_key}"] = precision
        image_results[f"recall_{threshold_key}"] = recall
        image_results[f"f1_{threshold_key}"] = f1
        image_results[f"tp_{threshold_key}"] = tp
        image_results[f"fp_{threshold_key}"] = fp
        image_results[f"fn_{threshold_key}"] = fn

        return image_results


    def evaluate_image_pair_old(
        self,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate a single prediction-ground truth pair.
        
        Args:
            pred_mask: Predicted instance segmentation mask (H, W) with instance IDs
            gt_mask: Ground truth instance segmentation mask (H, W) with instance IDs
            
        Returns:
            Dictionary containing metrics for this image pair
        """
        import time
        timings = {}
        # Extract instances
        t0 = time.time()
        pred_instances = self._extract_instances(pred_mask)
        gt_instances = self._extract_instances(gt_mask)

        timings['extract_instances'] = time.time() - t0

        # Calculate IoU matrix
        t0 = time.time()
        # iou_matrix = self.calculate_iou_matrix_fast(pred_pixels, gt_pixels)
        iou_matrix = self._calculate_iou_matrix(pred_instances, gt_instances)
        timings['calculate_iou_matrix'] = time.time() - t0

        # Calculate metrics at each threshold
        image_results = {}

        for threshold in self.iou_thresholds:
            t0 = time.time()
            metrics_at_threshold = self.compute_metrics_for_threshold(iou_matrix, threshold)
            timings[f'compute_metrics_{threshold:.2f}'] = time.time() - t0
            image_results.update(metrics_at_threshold)

        # Calculate AJI (Aggregated Jaccard Index)
        t0 = time.time()
        # aji = self._calculate_aji_optimized(pred_pixels, gt_pixels, iou_matrix)
        aji = self._calculate_aji(pred_instances, gt_instances, iou_matrix)
        timings['calculate_aji'] = time.time() - t0
        image_results['aji'] = aji

        # Calculate Panoptic Quality
        t0 = time.time()
        # pq = self._calculate_panoptic_quality_optimized(pred_pixels, gt_pixels, iou_matrix)
        pq = self._calculate_panoptic_quality(pred_instances, gt_instances, iou_matrix)
        timings['calculate_panoptic_quality'] = time.time() - t0
        image_results['pq'] = pq

        # Accumulate results
        self._accumulate_results(image_results)

        image_results['timings'] = timings
        return image_results
    

    def evaluate_image_pair(
        self,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate a single prediction-ground truth pair.
        
        Args:
            pred_mask: Predicted instance segmentation mask (H, W) with instance IDs
            gt_mask: Ground truth instance segmentation mask (H, W) with instance IDs
            
        Returns:
            Dictionary containing metrics for this image pair
        """
        import time
        timings = {}
        # Extract instances
        t0 = time.time()
        pred_ids, pred_pixels = self.extract_instances_fast(pred_mask)
        gt_ids, gt_pixels = self.extract_instances_fast(gt_mask)

        timings['extract_instances'] = time.time() - t0

        # Calculate IoU matrix
        t0 = time.time()
        iou_matrix = self.calculate_iou_matrix_fast(pred_pixels, gt_pixels)
        # iou_matrix = self._calculate_iou_matrix(pred_instances, gt_instances)
        timings['calculate_iou_matrix'] = time.time() - t0

        # Calculate metrics at each threshold
        image_results = {}

        for threshold in self.iou_thresholds:
            t0 = time.time()
            metrics_at_threshold = self.compute_metrics_for_threshold(iou_matrix, threshold)
            timings[f'compute_metrics_{threshold:.2f}'] = time.time() - t0
            image_results.update(metrics_at_threshold)

        # Calculate AJI (Aggregated Jaccard Index)
        t0 = time.time()
        aji = self._calculate_aji_optimized(pred_pixels, gt_pixels, iou_matrix)
        timings['calculate_aji'] = time.time() - t0
        image_results['aji'] = aji

        # Calculate Panoptic Quality
        t0 = time.time()
        pq = self._calculate_panoptic_quality_optimized(pred_pixels, gt_pixels, iou_matrix)
        timings['calculate_panoptic_quality'] = time.time() - t0
        image_results['pq'] = pq

        # Accumulate results
        self._accumulate_results(image_results)

        image_results['timings'] = timings
        return image_results
    
    def _extract_instances(self, mask: np.ndarray) -> List[np.ndarray]:
        """
        Extract individual instance masks from a labeled mask.
        
        Args:
            mask: Labeled instance mask (H, W)
            
        Returns:
            List of binary masks for each instance
        """
        instances = []
        unique_ids = np.unique(mask)
        
        for instance_id in unique_ids:
            if instance_id == 0:  # Skip background
                continue
            instance_mask = (mask == instance_id).astype(np.uint8)
            instances.append(instance_mask)
        
        return instances
    
    def _calculate_iou_matrix(
        self, 
        pred_instances: List[np.ndarray], 
        gt_instances: List[np.ndarray]
    ) -> np.ndarray:
        """
        Calculate IoU matrix between predicted and ground truth instances.
        
        Args:
            pred_instances: List of predicted instance masks
            gt_instances: List of ground truth instance masks
            
        Returns:
            IoU matrix (n_pred, n_gt)
        """
        if len(pred_instances) == 0 or len(gt_instances) == 0:
            return np.zeros((len(pred_instances), len(gt_instances)))
        
        iou_matrix = np.zeros((len(pred_instances), len(gt_instances)))
        
        for i, pred_mask in enumerate(pred_instances):
            for j, gt_mask in enumerate(gt_instances):
                intersection = np.logical_and(pred_mask, gt_mask).sum()
                union = np.logical_or(pred_mask, gt_mask).sum()
                
                if union > 0:
                    iou_matrix[i, j] = intersection / union
        
        return iou_matrix

    def extract_instances_fast(self, mask):
        ids = np.unique(mask)
        ids = ids[ids != 0]
        instance_pixels = [np.flatnonzero(mask == iid) for iid in ids]
        return ids, instance_pixels

    def calculate_iou_matrix_fast(self, pred_pixels, gt_pixels):

        n_pred, n_gt = len(pred_pixels), len(gt_pixels)
        iou = np.zeros((n_pred, n_gt), dtype=np.float32)

        for i in range(n_pred):
            p_idx = pred_pixels[i]
            for j in range(n_gt):
                g_idx = gt_pixels[j]
                inter = np.intersect1d(p_idx, g_idx).size
                union = p_idx.size + g_idx.size - inter
                if union > 0:
                    iou[i, j] = inter / union

        return iou

    
    def _match_instances(
        self, 
        iou_matrix: np.ndarray, 
        threshold: float
    ) -> Tuple[int, int, int, List[Tuple[int, int]]]:
        """
        Match predicted and ground truth instances using Hungarian algorithm.
        
        Args:
            iou_matrix: IoU matrix between predictions and ground truth
            threshold: IoU threshold for positive matches
            
        Returns:
            Tuple of (true_positives, false_positives, false_negatives, matches)
        """
        if iou_matrix.size == 0:
            return 0, iou_matrix.shape[0], iou_matrix.shape[1], []
        
        # Use negative IoU for minimization in Hungarian algorithm
        cost_matrix = -iou_matrix
        
        # Apply Hungarian algorithm
        pred_indices, gt_indices = linear_sum_assignment(cost_matrix)
        
        # Filter matches by threshold - should be optimized
        matches = []
        for pred_idx, gt_idx in zip(pred_indices, gt_indices):
            if iou_matrix[pred_idx, gt_idx] >= threshold:
                matches.append((pred_idx, gt_idx))
        
        # Calculate TP, FP, FN
        tp = len(matches)
        fp = iou_matrix.shape[0] - tp  # Unmatched predictions
        fn = iou_matrix.shape[1] - tp  # Unmatched ground truth
        
        return tp, fp, fn, matches



    def _match_instances_optimized(
        self, 
        iou_matrix: np.ndarray, 
        threshold: float
    ) -> Tuple[int, int, int, List[Tuple[int, int]]]:
        """
        Match predicted and ground truth instances using Hungarian algorithm.
        
        Args:
            iou_matrix: IoU matrix between predictions and ground truth
            threshold: IoU threshold for positive matches
            
        Returns:
            Tuple of (true_positives, false_positives, false_negatives, matches)
        """
        if iou_matrix.size == 0:
            return 0, iou_matrix.shape[0], iou_matrix.shape[1], []
        n_pred, n_gt = iou_matrix.shape

        # Use negative IoU for minimization in Hungarian algorithm
        cost_matrix = -iou_matrix
        
        # Apply Hungarian algorithm
        pred_indices, gt_indices = linear_sum_assignment(cost_matrix)
        
        # Keep only matches above threshold
        matches = [
            (pi, gi) for pi, gi in zip(pred_indices, gt_indices)
            if iou_matrix[pi, gi] >= threshold
        ]

        matched_pred = {pi for pi, _ in matches}
        matched_gt = {gi for _, gi in matches}

        tp = len(matches)
        fp = n_pred - len(matched_pred)
        fn = n_gt - len(matched_gt)
        
        return tp, fp, fn, matches


    def _calculate_panoptic_quality(
        self,
        pred_instances: List[np.ndarray],
        gt_instances: List[np.ndarray],
        iou_matrix: np.ndarray,
        threshold: float = 0.5
    ) -> float:
        """
        Calculate Panoptic Quality (PQ).
        
        Args:
            pred_instances: List of predicted instance masks
            gt_instances: List of ground truth instance masks
            iou_matrix: IoU matrix between predictions and ground truth
            threshold: IoU threshold for matching
            
        Returns:
            Panoptic Quality score
        """
        if len(pred_instances) == 0 and len(gt_instances) == 0:
            return 1.0
        
        if len(pred_instances) == 0 or len(gt_instances) == 0:
            return 0.0
        
        # Find matches above threshold
        tp, fp, fn, matches = self._match_instances(iou_matrix, threshold)
        
        # Calculate IoU sum for matched pairs
        iou_sum = 0.0
        for pred_idx, gt_idx in matches:
            iou_sum += iou_matrix[pred_idx, gt_idx]
        
        # PQ = (IoU_sum) / (TP + 0.5 * FP + 0.5 * FN)
        denominator = tp + 0.5 * fp + 0.5 * fn
        pq = iou_sum / denominator if denominator > 0 else 0.0
        
        return pq

    def _calculate_panoptic_quality_optimized(
        self,
        pred_pixels: list[np.ndarray],
        gt_pixels: list[np.ndarray],
        iou_matrix: np.ndarray,
        threshold: float = 0.5
        ) -> float:
        """
        Panoptic Quality (PQ) using pixel-index representation.
        """
        n_pred, n_gt = len(pred_pixels), len(gt_pixels)
        if n_pred == 0 and n_gt == 0:
            return 1.0
        if n_pred == 0 or n_gt == 0:
            return 0.0
        
        tp, fp, fn, matches = self._match_instances_optimized(iou_matrix, threshold)

        iou_sum = np.sum([iou_matrix[p, q] for p, q in matches])

        # PQ = (IoU_sum) / (TP + 0.5 * FP + 0.5 * FN)
        denominator = tp + 0.5 * fp + 0.5 * fn
        pq = iou_sum / denominator if denominator > 0 else 0.0

        return pq


    def _calculate_aji(
        self,
        pred_instances: List[np.ndarray],
        gt_instances: List[np.ndarray], 
        iou_matrix: np.ndarray
    ) -> float:
        """
        Calculate Aggregated Jaccard Index (AJI).
        
        Args:
            pred_instances: List of predicted instance masks
            gt_instances: List of ground truth instance masks
            iou_matrix: IoU matrix between predictions and ground truth
            
        Returns:
            AJI score
        """
        if len(pred_instances) == 0 and len(gt_instances) == 0:
            return 1.0
        
        if len(pred_instances) == 0 or len(gt_instances) == 0:
            return 0.0
        
        # Find best matches (Hungarian algorithm)
        cost_matrix = -iou_matrix
        pred_indices, gt_indices = linear_sum_assignment(cost_matrix)
        
        # Calculate AJI components
        intersection_sum = 0.0
        union_sum = 0.0
        
        matched_pred = set()
        matched_gt = set()
        
        # Process matched pairs
        for pred_idx, gt_idx in zip(pred_indices, gt_indices):
            if iou_matrix[pred_idx, gt_idx] > 0:
                pred_mask = pred_instances[pred_idx]
                gt_mask = gt_instances[gt_idx]
                
                intersection = np.logical_and(pred_mask, gt_mask).sum()
                union = np.logical_or(pred_mask, gt_mask).sum()
                
                intersection_sum += intersection
                union_sum += union
                
                matched_pred.add(pred_idx)
                matched_gt.add(gt_idx)
        
        # Add unmatched instances to union
        for i, pred_mask in enumerate(pred_instances):
            if i not in matched_pred:
                union_sum += pred_mask.sum()
        
        for j, gt_mask in enumerate(gt_instances):
            if j not in matched_gt:
                union_sum += gt_mask.sum()
        
        return intersection_sum / union_sum if union_sum > 0 else 0.0

    def _calculate_aji_optimized(
            self,
            pred_pixels: list[np.ndarray],
            gt_pixels: list[np.ndarray],
            iou_matrix: np.ndarray
        ) -> float:
        """
        Aggregated Jaccard Index (AJI) using pixel-index representation.
        """
        n_pred, n_gt = len(pred_pixels), len(gt_pixels)
        if n_pred == 0 and n_gt == 0:
            return 1.0
        if n_pred == 0 or n_gt == 0:
            return 0.0

        # Precompute pixel counts
        pred_areas = np.array([len(p) for p in pred_pixels], dtype=np.float64)
        gt_areas = np.array([len(g) for g in gt_pixels], dtype=np.float64)

        # Hungarian matching (maximize IoU)
        cost_matrix = -iou_matrix
        pred_idx, gt_idx = linear_sum_assignment(cost_matrix)

        intersection_sum = 0.0
        union_sum = 0.0
        matched_pred = np.zeros(n_pred, dtype=bool)
        matched_gt = np.zeros(n_gt, dtype=bool)

        for pi, gi in zip(pred_idx, gt_idx):
            if iou_matrix[pi, gi] > 0:
                p = pred_pixels[pi]
                g = gt_pixels[gi]
                inter = np.intersect1d(p, g, assume_unique=True).size
                uni = np.union1d(p, g).size
                intersection_sum += inter
                union_sum += uni
                matched_pred[pi] = True
                matched_gt[gi] = True

        # Add unmatched instances to union
        union_sum += pred_areas[~matched_pred].sum()
        union_sum += gt_areas[~matched_gt].sum()

        return intersection_sum / union_sum if union_sum > 0 else 0.0

    def _accumulate_results(self, image_results: Dict[str, float]):
        """Accumulate results from a single image."""
        for threshold in self.iou_thresholds:
            threshold_key = f"{threshold:.2f}"
            
            if threshold_key not in self.results['tp_per_threshold']:
                self.results['tp_per_threshold'][threshold_key] = []
                self.results['fp_per_threshold'][threshold_key] = []
                self.results['fn_per_threshold'][threshold_key] = []
                self.results['precision_per_threshold'][threshold_key] = []
                self.results['recall_per_threshold'][threshold_key] = []
                self.results['f1_per_threshold'][threshold_key] = []
            
            self.results['tp_per_threshold'][threshold_key].append(image_results[f"tp_{threshold_key}"])
            self.results['fp_per_threshold'][threshold_key].append(image_results[f"fp_{threshold_key}"])
            self.results['fn_per_threshold'][threshold_key].append(image_results[f"fn_{threshold_key}"])
            self.results['precision_per_threshold'][threshold_key].append(image_results[f"precision_{threshold_key}"])
            self.results['recall_per_threshold'][threshold_key].append(image_results[f"recall_{threshold_key}"])
            self.results['f1_per_threshold'][threshold_key].append(image_results[f"f1_{threshold_key}"])
        
        self.results['aji_scores'].append(image_results['aji'])
        self.results['pq_scores'].append(image_results['pq'])
    
    def compute_final_metrics(self) -> Dict[str, float]:
        """
        Compute final aggregated metrics across all evaluated images.
        
        Returns:
            Dictionary containing final metrics
        """
        final_metrics = {}
        
        # Calculate mAP and other threshold-averaged metrics
        ap_scores = []
        
        for threshold in self.iou_thresholds:
            threshold_key = f"{threshold:.2f}"
            
            if threshold_key not in self.results['tp_per_threshold']:
                continue
            
            # Aggregate across images
            total_tp = sum(self.results['tp_per_threshold'][threshold_key])
            total_fp = sum(self.results['fp_per_threshold'][threshold_key])
            total_fn = sum(self.results['fn_per_threshold'][threshold_key])
            
            # Calculate aggregated metrics
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            final_metrics[f"precision_@{threshold_key}"] = precision
            final_metrics[f"recall_@{threshold_key}"] = recall
            final_metrics[f"f1_@{threshold_key}"] = f1
            final_metrics[f"ap_@{threshold_key}"] = precision  # Simplified AP as precision
            
            ap_scores.append(precision)
        
        # Calculate mAP
        final_metrics['mAP'] = np.mean(ap_scores) if ap_scores else 0.0
        
        # Calculate mAP@0.5 and mAP@0.75 specifically
        if '0.50' in self.results['tp_per_threshold']:
            final_metrics['mAP@0.5'] = final_metrics.get('ap_@0.50', 0.0)
        if '0.75' in self.results['tp_per_threshold']:
            final_metrics['mAP@0.75'] = final_metrics.get('ap_@0.75', 0.0)
        
        # Calculate mean F1 across thresholds
        f1_scores = [final_metrics[key] for key in final_metrics.keys() if key.startswith('f1_@')]
        final_metrics['mean_F1'] = np.mean(f1_scores) if f1_scores else 0.0
        
        # Calculate mean AJI and PQ
        final_metrics['mean_AJI'] = np.mean(self.results['aji_scores']) if self.results['aji_scores'] else 0.0
        final_metrics['mean_PQ'] = np.mean(self.results['pq_scores']) if self.results['pq_scores'] else 0.0
        
        return final_metrics
    
    def get_detailed_results(self) -> Dict[str, Union[Dict, List]]:
        """
        Get detailed results including per-image metrics.
        
        Returns:
            Dictionary containing all accumulated results
        """
        return self.results.copy()


def calculate_instance_f1_score(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    iou_threshold: float = 0.5
) -> float:
    """
    Calculate F1 score for instance segmentation at a specific IoU threshold.
    
    Args:
        pred_mask: Predicted instance segmentation mask
        gt_mask: Ground truth instance segmentation mask
        iou_threshold: IoU threshold for considering a match
        
    Returns:
        F1 score
    """
    metrics = InstanceSegmentationMetrics([iou_threshold])
    result = metrics.evaluate_image_pair(pred_mask, gt_mask)
    return result[f"f1_{iou_threshold:.2f}"]


def calculate_map(
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
    iou_thresholds: Optional[List[float]] = None
) -> float:
    """
    Calculate mean Average Precision (mAP) across multiple images.
    
    Args:
        pred_masks: List of predicted instance segmentation masks
        gt_masks: List of ground truth instance segmentation masks
        iou_thresholds: List of IoU thresholds for evaluation
        
    Returns:
        mAP score
    """
    if len(pred_masks) != len(gt_masks):
        raise ValueError("Number of prediction and ground truth masks must match")
    
    metrics = InstanceSegmentationMetrics(iou_thresholds)
    
    for pred_mask, gt_mask in zip(pred_masks, gt_masks):
        metrics.evaluate_image_pair(pred_mask, gt_mask)
    
    final_metrics = metrics.compute_final_metrics()
    return final_metrics['mAP']
