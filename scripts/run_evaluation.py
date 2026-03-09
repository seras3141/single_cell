#!/usr/bin/env python3
"""
Run evaluation script for single-cell segmentation.

This script provides a command-line interface to evaluate segmentation predictions
against ground truth labels using comprehensive instance and pixel-wise metrics.

Usage:
    python run_evaluation.py --predictions /path/to/predictions --labels /path/to/labels --output /path/to/output

The script supports various input formats:
- Directory of image files (PNG, TIFF, etc.)
- Single multi-page TIFF file
- NumPy array files (.npy, .npz)

Example:
    python run_evaluation.py \
        --predictions ./predictions/ \
        --labels ./ground_truth/ \
        --output ./evaluation_results/ \
        --format tiff \
        --iou-thresholds 0.5 0.75 0.9 \
        --pixel-spacing 0.65 \
        --compute-distances
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import List, Union, Optional, Dict, Any
import numpy as np
from tqdm import tqdm
import json

from src.evaluation.evaluation_pipeline import evaluate_segmentation_performance, EvaluationPipeline
from src.evaluation.instance_metrics import calculate_map
from src.evaluation.pixel_metrics import calculate_dice_coefficient
from src.evaluation.file_matching import get_matching_prediction_label_files, get_matching_prediction_label_file_paths


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_eval_summary(summary: Dict[str, Any]):
    """
    Print summary metrics to console.
    
    Args:
        summary: Summary metrics dictionary
    """
    # logger.info("Evaluation Summary:")
    # logger.info(f"Number of images evaluated: {summary.get('num_images_evaluated', 0)}")
    
    # instance_metrics = summary.get('instance_metrics', {})
    # pixel_metrics = summary.get('pixel_metrics', {})
    
    # if instance_metrics:
    #     logger.info("Instance-level Metrics:")
    #     for key, value in instance_metrics.items():
    #         logger.info(f"  {key}: {value:.4f}")
    
    # if pixel_metrics:
    #     logger.info("Pixel-level Metrics:")
    #     for key, value in pixel_metrics.items():
    #         logger.info(f"  {key}: {value:.4f}")    

    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Images evaluated: {summary['num_images_evaluated']}")
    # print(f"Output directory: {output_dir}")
    
    if summary['instance_metrics']:
        print(f"\nInstance Metrics:")
        print(f"  mAP: {summary['instance_metrics'].get('mAP', 0):.3f}")
        print(f"  mAP@0.5: {summary['instance_metrics'].get('mAP@0.5', 0):.3f}")
        print(f"  Mean F1: {summary['instance_metrics'].get('mean_F1', 0):.3f}")
        print(f"  Mean AJI: {summary['instance_metrics'].get('mean_AJI', 0):.3f}")
    
    if summary['pixel_metrics']:
        print(f"\nPixel Metrics:")
        print(f"  Mean Dice: {summary['pixel_metrics'].get('mean_dice', 0):.3f}")
        print(f"  Mean IoU: {summary['pixel_metrics'].get('mean_iou', 0):.3f}")
        print(f"  Mean Precision: {summary['pixel_metrics'].get('mean_precision', 0):.3f}")
        print(f"  Mean Recall: {summary['pixel_metrics'].get('mean_recall', 0):.3f}")
    
    print("\nFiles created:")
    print(f"  - evaluation_results.json")
    print(f"  - evaluation_results.csv")
    print(f"  - evaluation_results.xlsx (if available)")
    print(f"  - evaluation_config.json")
    print("="*60)


def run_evaluation(
    predictions_path: Union[str, Path],
    labels_path: Union[str, Path],
    output_dir: Union[str, Path],
    output_format: List[str] = ['csv'],
    iou_thresholds: Optional[List[float]] = None,
    pixel_spacing: Optional[float] = None,
    compute_distances: bool = False,
    prediction_file_pattern: str = "*",
    label_file_pattern: str = "*",
    create_plots: bool = True
) -> Dict[str, Any]:
    """
    Run comprehensive evaluation on prediction and label masks.
    
    Args:
        predictions_path: Path to predictions (directory or file)
        labels_path: Path to labels (directory or file)
        output_dir: Directory to save results
        iou_thresholds: List of IoU thresholds for instance evaluation
        pixel_spacing: Physical spacing between pixels
        compute_distances: Whether to compute distance-based metrics
        file_pattern: Pattern for matching files in directories
        create_plots: Whether to generate evaluation plots
        
    Returns:
        Dictionary containing evaluation results
    """

    matched_pairs = get_matching_prediction_label_file_paths(
        predictions_path,
        labels_path,
        prediction_file_pattern=prediction_file_pattern,
        label_file_pattern=label_file_pattern
    )

    # matched_pairs = get_matching_prediction_label_files(
    #     predictions_path,
    #     labels_path,
    #     prediction_file_pattern=prediction_file_pattern,
    #     label_file_pattern=label_file_pattern
    # )
    
    if not matched_pairs:
        raise ValueError("No valid prediction-label pairs found")
    
    # Extract arrays and image IDs
    # pred_masks = [pair[0] for pair in matched_pairs]
    # label_masks = [pair[1] for pair in matched_pairs]
    # image_ids = [pair[2] for pair in matched_pairs]
    pred_paths = [pair[0] for pair in matched_pairs]
    label_paths = [pair[1] for pair in matched_pairs]
    image_ids = [pair[2] for pair in matched_pairs]

    # Initialize evaluation pipeline
    logger.info("Initializing evaluation pipeline...")
    pipeline = EvaluationPipeline(
        iou_thresholds=iou_thresholds,
        pixel_spacing=pixel_spacing,
        compute_distances=compute_distances
    )
    
    # Run evaluation
    logger.info(f"Evaluating {len(matched_pairs)} image pairs...")
    batch_results = pipeline.evaluate_batch_path(pred_paths, label_paths, image_ids)
    # batch_results = pipeline.evaluate_batch(pred_masks, label_masks, image_ids)
    
    # Compute summary metrics
    logger.info("Computing summary metrics...")
    summary = pipeline.compute_summary_metrics()
    
    # Create output directory
    output_dir = Path(output_dir)
    logger.info(f"Output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export results
    logger.info("Exporting results...")
    if 'json' in output_format:
        pipeline.export_results(output_dir / 'evaluation_results.json', 'json')
    if 'csv' in output_format:
        pipeline.export_results(output_dir / 'evaluation_results.csv', 'csv')
    if 'xlsx' in output_format:    
        try:
            pipeline.export_results(output_dir / 'evaluation_results.xlsx', 'xlsx')
        except Exception as e:
            logger.warning(f"Failed to export Excel file: {e}")
    
    # Save configuration
    config = {
        'predictions_path': str(predictions_path),
        'labels_path': str(labels_path),
        'prediction_file_pattern': prediction_file_pattern,
        'label_file_pattern': label_file_pattern,
        'output_dir': str(output_dir),
        'iou_thresholds': iou_thresholds,
        'pixel_spacing': pixel_spacing,
        'compute_distances': compute_distances,
        'num_images_evaluated': len(matched_pairs),
        'image_ids': image_ids
    }
    
    with open(output_dir / 'evaluation_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Generate plots if requested
    if create_plots:
        try:
            logger.info("Generating evaluation plots...")
            plots = pipeline.plot_results(
                output_dir=output_dir,
                show=False,  # Don't show in CLI mode
                plot_types=['all']
            )
            
            if plots:
                print("\nPlots created:")
                for plot_name in plots.keys():
                    print(f"  - {plot_name}.png")
            
        except Exception as e:
            logger.warning(f"Failed to generate plots: {e}")
    
    # Print summary
    print_eval_summary(summary)
    
    return {
        'summary': summary,
        'detailed_results': batch_results,
        'config': config
    }


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Evaluate single-cell segmentation predictions against ground truth labels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate directory of images
  python run_evaluation.py --predictions ./pred/ --labels ./gt/ --output ./results/

  # With custom IoU thresholds and pixel spacing
  python run_evaluation.py --predictions ./pred/ --labels ./gt/ --output ./results/ \\
    --iou-thresholds 0.5 0.75 0.9 --pixel-spacing 0.65 --compute-distances

  # Evaluate single files
  python run_evaluation.py --predictions pred.tiff --labels gt.tiff --output ./results/

  # Custom file pattern
  python run_evaluation.py --predictions ./pred/ --labels ./gt/ --output ./results/ \\
    --pattern "*.png"
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--predictions', '-p',
        type=str,
        required=True,
        help='Path to predictions (directory or file)'
    )
    
    parser.add_argument(
        '--labels', '-l',
        type=str,
        required=True,
        help='Path to ground truth labels (directory or file)'
    )

    parser.add_argument('--pred-pattern', type=str, default="*", help='File pattern for matching prediction files (default: "*")')
    parser.add_argument('--label-pattern', type=str, default="*", help='File pattern for matching label files (default: "*")')

    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output directory for results'
    )
    
    # Optional arguments
    parser.add_argument(
        '--iou-thresholds',
        type=float,
        nargs='+',
        default=None,
        help='IoU thresholds for instance evaluation (default: 0.5 to 0.95 in steps of 0.05)'
    )
    
    parser.add_argument(
        '--pixel-spacing',
        type=float,
        default=None,
        help='Physical spacing between pixels (e.g., 0.65 for 0.65 μm/pixel)'
    )
    
    parser.add_argument(
        '--compute-distances',
        action='store_true',
        help='Compute distance-based metrics (Hausdorff, surface distance)'
    )
        
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Quiet mode (minimal output)'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating evaluation plots'
    )

    args = parser.parse_args()
    
    # Set logging level
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Run evaluation
        results = run_evaluation(
            predictions_path=args.predictions,
            labels_path=args.labels,
            output_dir=args.output,
            iou_thresholds=args.iou_thresholds,
            pixel_spacing=args.pixel_spacing,
            compute_distances=args.compute_distances,
            prediction_file_pattern=args.pred_pattern,
            label_file_pattern=args.label_pattern,
            create_plots=not args.no_plots
        )
        
        logger.info("Evaluation completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
