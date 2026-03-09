"""
Example usage of the 3D Cell Tracking Processor.

This script demonstrates how to use the complete 3D cell tracking postprocessing
pipeline with blur-based filtering. It shows both programmatic usage and
integration with the modular architecture.
"""

import logging
import numpy as np
import tifffile
from pathlib import Path

from src.postprocessing import (
    TrackingProcessor,
    TrackingProcessorConfig,
    TrackingConfig,
    FilterConfig,
    run_tracking_pipeline,
    main_compatible
)


def setup_logging():
    """Set up logging for the examples."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def example_1_basic_tracking():
    """
    Example 1: Basic 3D cell tracking with default parameters.
    
    This example shows the simplest way to use the tracking processor
    with default configuration settings.
    """
    print("\n=== Example 1: Basic 3D Cell Tracking ===")
    
    # Use default configuration
    processor = TrackingProcessor()
    
    # Create mock data for demonstration
    # In practice, you would have real segmentation masks
    segmentation_stack = np.zeros((5, 100, 100), dtype=int)
    
    # Add some mock cells that move across frames
    for z in range(5):
        # Cell 1: moves diagonally
        x1, y1 = 20 + z*2, 20 + z*2
        segmentation_stack[z, y1:y1+15, x1:x1+15] = 1
        
        # Cell 2: stationary
        segmentation_stack[z, 60:75, 60:75] = 2
        
        # Cell 3: appears in later frames
        if z >= 2:
            segmentation_stack[z, 30:45, 70:85] = 3
    
    # Perform tracking
    tracked_result = processor.track_3d_centers(segmentation_stack)
    
    print(f"Input shape: {segmentation_stack.shape}")
    print(f"Output shape: {tracked_result.shape}")
    print(f"Unique tracked IDs: {np.unique(tracked_result)}")
    print(f"Number of tracked particles: {len(np.unique(tracked_result)) - 1}")  # Exclude background


def example_2_custom_configuration():
    """
    Example 2: Custom configuration for specific experimental conditions.
    
    This example shows how to configure the tracking processor for
    specific experimental conditions with custom parameters.
    """
    print("\n=== Example 2: Custom Configuration ===")
    
    # Create custom tracking configuration
    tracking_config = TrackingConfig(
        search_range=10.0,  # Larger search range for fast-moving cells
        memory=2,           # Remember particles for 2 frames
        min_track_length=4, # Require longer tracks
        min_area=20,        # Larger minimum cell area
        max_area=3000       # Smaller maximum cell area
    )
    
    # Create custom filter configuration
    filter_config = FilterConfig(
        patch_size=64,      # Larger patches for blur measurement
        stride_size=16,     # Larger stride
        blur_threshold=0.3, # More stringent blur threshold
        invert_threshold=False
    )
    
    # Create processor configuration
    config = TrackingProcessorConfig(
        blur_threshold=0.3,
        invert_blur_threshold=False,
        tracking_config=tracking_config,
        filter_config=filter_config,
        save_tracking_data=True  # Save intermediate data
    )
    
    processor = TrackingProcessor(config)
    
    print(f"Search range: {config.tracking_config.search_range}")
    print(f"Memory: {config.tracking_config.memory}")
    print(f"Blur threshold: {config.blur_threshold}")
    print(f"Patch size: {config.filter_config.patch_size}")


def example_3_blur_filtering():
    """
    Example 3: Demonstrating blur-based filtering effects.
    
    This example shows how blur filtering affects cell detection
    and tracking quality.
    """
    print("\n=== Example 3: Blur-based Filtering ===")
    
    # Create segmentation with cells of different "sharpness"
    segmentation = np.zeros((3, 100, 100), dtype=int)
    segmentation[0, 20:35, 20:35] = 1  # Cell 1
    segmentation[0, 60:75, 60:75] = 2  # Cell 2
    segmentation[1, 22:37, 22:37] = 1  # Cell 1 moved
    segmentation[1, 62:77, 62:77] = 2  # Cell 2 moved
    
    # Create mock blur image (lower values = sharper)
    blur_image = np.ones((3, 100, 100)) * 0.8  # Most regions blurry
    blur_image[:, 20:40, 20:40] = 0.2  # Cell 1 area sharp
    blur_image[:, 60:80, 60:80] = 0.9  # Cell 2 area very blurry
    
    # Test without blur filtering
    processor_no_filter = TrackingProcessor(TrackingProcessorConfig(
        blur_threshold=1.0  # Accept all cells
    ))
    
    tracked_no_filter = processor_no_filter.track_3d_centers(segmentation)
    
    # Test with strict blur filtering
    processor_with_filter = TrackingProcessor(TrackingProcessorConfig(
        blur_threshold=0.5  # Only accept sharp cells
    ))
    
    tracked_with_filter = processor_with_filter.track_3d_centers(
        segmentation, sharpness_image=blur_image
    )
    
    print(f"Without filtering - unique IDs: {np.unique(tracked_no_filter)}")
    print(f"With filtering - unique IDs: {np.unique(tracked_with_filter)}")
    print("Blur filtering removed low-quality detections")


def example_4_batch_processing():
    """
    Example 4: Batch processing multiple files.
    
    This example demonstrates how to process multiple segmentation
    files in batch mode.
    """
    print("\n=== Example 4: Batch Processing ===")
    
    # For this example, we'll show the configuration for batch processing
    # In practice, you would have real directories with files
    
    config = TrackingProcessorConfig(
        mask_pattern="*_3d.tif",
        blur_heatmap_suffix="_blur_heatmap_32_8.tif",
        blur_threshold=0.5,
        create_output_dirs=True,
        overwrite_existing=False,  # Don't overwrite existing results
        save_tracking_data=True    # Save tracking data as CSV
    )
    
    print("Configuration for batch processing:")
    print(f"- Mask pattern: {config.mask_pattern}")
    print(f"- Blur threshold: {config.blur_threshold}")
    print(f"- Create output dirs: {config.create_output_dirs}")
    print(f"- Save tracking data: {config.save_tracking_data}")
    
    # Example of how to run batch processing:
    # results = run_tracking_pipeline(
    #     mask_directory="path/to/masks",
    #     image_directory="path/to/images",
    #     output_directory="path/to/output",
    #     blur_directory="path/to/blur_cache",
    #     config=config
    # )


def example_5_compatibility_mode():
    """
    Example 5: Using compatibility mode with original track_cells.py interface.
    
    This example shows how to use the new implementation while maintaining
    compatibility with the original interface.
    """
    print("\n=== Example 5: Compatibility Mode ===")
    
    # This function provides the same interface as the original main()
    # but uses the new modular architecture internally
    
    print("Compatibility function signature matches original track_cells.py:")
    print("main_compatible(")
    print("    image_directory='path/to/images',")
    print("    mask_directory='path/to/masks',") 
    print("    output_directory='path/to/output',")
    print("    blur_directory='path/to/blur',")
    print("    blur_thresh=0.5,")
    print("    inv=False")
    print(")")
    
    # Example usage (commented out since we don't have real files):
    # results = main_compatible(
    #     image_directory="data/BF+IF Experiments_3D_train_test_dataset/train",
    #     mask_directory="data/BF+IF Experiments_2D_train_test_dataset/predictions_test/pretrained_cell2d_cyto3_FlowT0.4/3d_view",
    #     output_directory="data/BF+IF Experiments_2D_train_test_dataset/predictions_test/pretrained_cell2d_cyto3_FlowT0.4/3d_view_tracked_SF",
    #     blur_directory="data/BF+IF Experiments_3D_train_test_dataset/blur_heatmaps",
    #     blur_thresh=0.5,
    #     inv=False
    # )


def example_6_error_handling():
    """
    Example 6: Error handling and robustness.
    
    This example demonstrates how the processor handles various error
    conditions and provides useful feedback.
    """
    print("\n=== Example 6: Error Handling ===")
    
    processor = TrackingProcessor(TrackingProcessorConfig(
        create_output_dirs=True,
        overwrite_existing=False
    ))
    
    # Example 1: Empty segmentation
    empty_segmentation = np.zeros((3, 50, 50), dtype=int)
    result = processor.track_3d_centers(empty_segmentation)
    print(f"Empty segmentation result shape: {result.shape}")
    print(f"Empty segmentation unique values: {np.unique(result)}")
    
    # Example 2: Single cell per frame
    single_cell_seg = np.zeros((3, 50, 50), dtype=int)
    single_cell_seg[:, 20:30, 20:30] = 1
    result = processor.track_3d_centers(single_cell_seg)
    print(f"Single cell tracking result: {np.unique(result)}")
    
    print("The processor gracefully handles edge cases and provides useful logging")


def main():
    """Run all examples."""
    setup_logging()
    
    print("3D Cell Tracking Processor Examples")
    print("="*50)
    
    try:
        example_1_basic_tracking()
        example_2_custom_configuration()
        example_3_blur_filtering()
        example_4_batch_processing()
        example_5_compatibility_mode()
        example_6_error_handling()
        
        print("\n" + "="*50)
        print("All examples completed successfully!")
        print("\nKey features demonstrated:")
        print("- Basic 3D cell tracking across z-stacks")
        print("- Custom configuration for different experimental conditions")
        print("- Blur-based filtering for quality improvement")
        print("- Batch processing of multiple files")
        print("- Compatibility with original interface")
        print("- Robust error handling")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        logging.exception("Example execution failed")


if __name__ == "__main__":
    main()
