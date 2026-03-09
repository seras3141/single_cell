#!/usr/bin/env python3
"""
Example script demonstrating the postprocessing pipeline usage.

This script shows how to use the postprocessing module for 3D cell tracking
with blur-based filtering.
"""

import os
import sys
import tempfile
import numpy as np
import tifffile
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.postprocessing.pipeline import CellTrackingPipeline, PipelineConfig
from src.postprocessing.cell_tracking import TrackingConfig
from src.postprocessing.blur_filtering import FilterConfig


def create_example_data(output_dir: Path):
    """Create example 3D segmentation and image data."""
    print("Creating example data...")
    
    # Create 3D segmentation stack (5 z-slices, 100x100 pixels)
    segmentation = np.zeros((5, 100, 100), dtype=np.int32)
    
    # Cell 1: Moving diagonally across z-stacks
    for z in range(5):
        y1 = 20 + z * 3  # Moving down
        x1 = 30 + z * 2  # Moving right
        segmentation[z, y1:y1+10, x1:x1+10] = 1
    
    # Cell 2: Moving vertically
    for z in range(5):
        y2 = 50 + z * 4
        x2 = 60
        segmentation[z, y2:y2+8, x2:x2+8] = 2
    
    # Cell 3: Stationary cell
    for z in range(5):
        y3 = 70
        x3 = 20
        segmentation[z, y3:y3+12, x3:x3+12] = 3
    
    # Cell 4: Cell that appears and disappears (simulating division/death)
    for z in range(2, 4):  # Only appears in middle z-slices
        y4 = 15
        x4 = 70
        segmentation[z, y4:y4+6, x4:x4+6] = 4
    
    # Create corresponding image data (random noise with some structure)
    image = np.random.randint(50, 200, (5, 100, 100), dtype=np.uint16)
    
    # Add some "signal" where cells are located
    for z in range(5):
        cell_mask = segmentation[z] > 0
        image[z][cell_mask] = np.random.randint(150, 255, np.sum(cell_mask))
    
    # Save files
    seg_path = output_dir / "example_segmentation_3d.tif"
    img_path = output_dir / "example_image_BF_3d.tif"
    
    tifffile.imwrite(str(seg_path), segmentation)
    tifffile.imwrite(str(img_path), image)
    
    print(f"Created segmentation: {seg_path}")
    print(f"Created image: {img_path}")
    
    return seg_path, img_path


def example_basic_usage():
    """Example 1: Basic pipeline usage with default settings."""
    print("\n" + "="*50)
    print("EXAMPLE 1: Basic Pipeline Usage")
    print("="*50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create example data
        seg_path, img_path = create_example_data(temp_path)
        output_dir = temp_path / "basic_output"
        
        # Run pipeline with default settings
        from src.postprocessing.pipeline import process_single_stack
        
        result = process_single_stack(
            segmentation_path=seg_path,
            image_path=img_path,
            output_dir=output_dir
        )
        
        print(f"\nProcessing completed!")
        print(f"Input shape: {result['input_shape']}")
        print(f"Final output: {result['final_output']}")
        print(f"Processing steps: {result['processing_steps']}")
        
        if 'tracking' in result:
            tracking_stats = result['tracking']
            print(f"Particles tracked: {tracking_stats['n_particles']}")
            print(f"Average track length: {tracking_stats['avg_track_length']:.1f}")


def example_custom_configuration():
    """Example 2: Custom configuration and detailed analysis."""
    print("\n" + "="*50)
    print("EXAMPLE 2: Custom Configuration")
    print("="*50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create example data
        seg_path, img_path = create_example_data(temp_path)
        output_dir = temp_path / "custom_output"
        
        # Create custom configuration
        tracking_config = TrackingConfig(
            search_range=8.0,      # Larger search range
            memory=2,              # Remember particles for 2 frames
            min_track_length=2,    # Shorter minimum track length
            min_area=5,            # Smaller minimum area
            max_area=1000          # Larger maximum area
        )
        
        filter_config = FilterConfig(
            patch_size=16,         # Smaller patches for faster computation
            stride_size=8,
            blur_threshold=0.3,    # More permissive blur threshold
            cache_blur_maps=True
        )
        
        pipeline_config = PipelineConfig(
            tracking_config=tracking_config,
            filter_config=filter_config,
            enable_blur_filtering=True,
            filter_before_tracking=True,
            save_intermediate_results=True
        )
        
        # Initialize pipeline
        pipeline = CellTrackingPipeline(pipeline_config)
        
        # Process the file
        result = pipeline.process_single_file(
            seg_path, img_path, output_dir,
            blur_cache_dir=temp_path / "blur_cache"
        )
        
        print(f"\nCustom processing completed!")
        print(f"Processing steps: {result['processing_steps']}")
        
        # Show detailed results
        if 'blur_filtering' in result:
            blur_stats = result['blur_filtering']
            print(f"Blur filtering: {blur_stats['total_cells_after']}/{blur_stats['total_cells_before']} cells passed")
        
        if 'tracking' in result:
            tracking_stats = result['tracking']
            print(f"Tracking: {tracking_stats['n_particles']} particles, "
                  f"avg length: {tracking_stats['avg_track_length']:.1f}")
        

def example_component_usage():
    """Example 3: Using individual components separately."""
    print("\n" + "="*50)
    print("EXAMPLE 3: Individual Component Usage")
    print("="*50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create example data
        seg_path, img_path = create_example_data(temp_path)
        
        # Load data
        segmentation = tifffile.imread(str(seg_path))
        
        print(f"Loaded segmentation with shape: {segmentation.shape}")
        
        # Step 1: Use blur filter
        from src.postprocessing.blur_filtering import BlurFilter
        
        blur_filter = BlurFilter()
        blur_heatmap = blur_filter.get_or_compute_blur_heatmap(img_path)
        
        print(f"Computed blur heatmap with shape: {blur_heatmap.shape}")
        
        # Filter cells by blur
        filtered_stack, quality_stats = blur_filter.filter_3d_stack(
            segmentation, [blur_heatmap] * segmentation.shape[0]
        )
        
        total_cells = sum(len(stats) for stats in quality_stats)
        passed_cells = sum(stats['passes_threshold'].sum() for stats in quality_stats)
        print(f"Blur filtering: {passed_cells}/{total_cells} cells passed")
        
        # Step 2: Use tracker
        from src.postprocessing.cell_tracking import CellTracker3D
        
        tracker = CellTracker3D()
        tracked_stack = tracker.track_cells(filtered_stack)
        
        tracking_stats = tracker.get_tracking_summary()
        print(f"Tracking: {tracking_stats['n_particles']} particles tracked")
        
                
            


def example_batch_processing():
    """Example 4: Batch processing multiple files."""
    print("\n" + "="*50)
    print("EXAMPLE 4: Batch Processing")
    print("="*50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_dir = temp_path / "input"
        input_dir.mkdir()
        
        # Create multiple example files
        for i in range(3):
            seg_path = input_dir / f"sample_{i}_3d.tif"
            img_path = input_dir / f"sample_{i}_BF_3d.tif"
            
            # Create slightly different data for each file
            segmentation = np.zeros((3, 50, 50), dtype=np.int32)
            image = np.random.randint(50, 200, (3, 50, 50), dtype=np.uint16)
            
            # Add some cells
            for z in range(3):
                # Moving cell
                y = 10 + z * 2 + i * 5  # Different starting position for each file
                x = 15 + z * 1
                segmentation[z, y:y+6, x:x+6] = 1
                image[z, y:y+6, x:x+6] = 255
                
                # Stationary cell
                y2 = 30 + i * 3
                x2 = 35
                segmentation[z, y2:y2+8, x2:x2+8] = 2
                image[z, y2:y2+8, x2:x2+8] = 200
            
            tifffile.imwrite(str(seg_path), segmentation)
            tifffile.imwrite(str(img_path), image)
        
        print(f"Created {len(list(input_dir.glob('*_3d.tif')))} example files")
        
        # Process batch
        from src.postprocessing.pipeline import process_batch_stacks
        
        output_dir = temp_path / "batch_output"
        
        results = process_batch_stacks(
            input_dir=input_dir,
            output_dir=output_dir
        )
        
        print(f"\nBatch processing completed!")
        print(f"Total files processed: {len(results)}")
        
        successful = [r for r in results if 'error' not in r]
        print(f"Successful: {len(successful)}")
        
        # Show summary for first file
        if successful:
            first_result = successful[0]
            print(f"\nExample result (first file):")
            print(f"  Input: {Path(first_result['input_segmentation']).name}")
            print(f"  Output: {Path(first_result['final_output']).name}")
            if 'tracking' in first_result:
                print(f"  Particles: {first_result['tracking']['n_particles']}")


def main():
    """Run all examples."""
    print("🔬 Cell Tracking Postprocessing Examples")
    print("========================================")
    
    try:
        # Set environment variable to avoid OpenMP issues
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        
        # Run examples
        example_basic_usage()
        example_custom_configuration()
        example_component_usage()
        example_batch_processing()
        
        print("\n" + "="*50)
        print("✅ All examples completed successfully!")
        print("="*50)
        print("\nFor more information, see:")
        print("- src/postprocessing/README.md")
        print("- CLI help: python scripts/run_tracking_pipeline.py --help")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
