"""
Tests for the postprocessing module.

This module contains tests for all postprocessing functionality including
cell tracking, and blur filtering.
"""

import unittest
import tempfile
import shutil
import os
import numpy as np
import pandas as pd
import tifffile
from pathlib import Path
from unittest.mock import patch
import glob

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.postprocessing.cell_tracking import CellTracker3D, TrackingConfig
from src.postprocessing.blur_filtering import BlurFilter, FilterConfig
from src.postprocessing.pipeline import CellTrackingPipeline, PipelineConfig
from src.postprocessing.tracking_processor import TrackingProcessor, TrackingProcessorConfig


class TestCellTracking3D(unittest.TestCase):
    """Tests for 3D cell tracking functionality."""
    
    def setUp(self):
        """Set up test data."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = TrackingConfig(
            search_range=5.0,
            memory=1,
            min_track_length=2,
            min_area=10,
            max_area=1000
        )
        self.tracker = CellTracker3D(self.config)
    
    def tearDown(self):
        """Clean up test data."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_mock_segmentation_stack(self, shape=(5, 100, 100)):
        """Create mock 3D segmentation data."""
        z, h, w = shape
        stack = np.zeros(shape, dtype=np.int32)
        
        # Create some cells that move slightly across z-stacks
        for z_idx in range(z):
            # Cell 1: moves diagonally
            y1 = 20 + z_idx * 2
            x1 = 30 + z_idx * 1
            stack[z_idx, y1:y1+10, x1:x1+10] = 1
            
            # Cell 2: moves vertically
            y2 = 50 + z_idx * 3
            x2 = 60
            stack[z_idx, y2:y2+8, x2:x2+8] = 2
            
            # Cell 3: stationary
            y3 = 70
            x3 = 20
            stack[z_idx, y3:y3+12, x3:x3+12] = 3
        
        return stack
    
    def test_extract_cell_properties(self):
        """Test cell property extraction from single mask."""
        # Create a simple 2D mask
        mask = np.zeros((100, 100), dtype=np.int32)
        mask[20:30, 30:40] = 1  # Cell 1
        mask[50:58, 60:68] = 2  # Cell 2
        
        properties = self.tracker.extract_cell_properties(mask)
        
        # Should find 2 cells
        self.assertEqual(len(properties), 2)
        self.assertIn('x', properties.columns)
        self.assertIn('y', properties.columns)
        self.assertIn('label', properties.columns)
        self.assertIn('area', properties.columns)
        
        # Check areas are reasonable
        self.assertTrue(all(properties['area'] >= self.config.min_area))
        self.assertTrue(all(properties['area'] <= self.config.max_area))
    
    def test_track_cells_3d(self):
        """Test 3D cell tracking."""
        segmentation_stack = self.create_mock_segmentation_stack()
        
        tracked_stack = self.tracker.track_cells(segmentation_stack)
        
        # Check output shape matches input
        self.assertEqual(tracked_stack.shape, segmentation_stack.shape)
        
        # Should have tracking data
        self.assertIsNotNone(self.tracker.last_tracking_data)
        self.assertGreater(len(self.tracker.last_tracking_data), 0)
        
        # Check tracking statistics
        stats = self.tracker.get_tracking_summary()
        self.assertIn('n_particles', stats)
        self.assertIn('n_detections', stats)
        self.assertGreater(stats['n_particles'], 0)
    
    def test_empty_segmentation(self):
        """Test tracking with empty segmentation."""
        empty_stack = np.zeros((3, 50, 50), dtype=np.int32)
        
        tracked_stack = self.tracker.track_cells(empty_stack)
        
        # Should return zeros
        self.assertTrue(np.all(tracked_stack == 0))
        
        # Should have empty tracking data
        stats = self.tracker.get_tracking_summary()
        self.assertEqual(stats.get('n_particles', 0), 0)


class TestBlurFiltering(unittest.TestCase):
    """Tests for blur-based filtering functionality."""
    
    def setUp(self):
        """Set up test data."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = FilterConfig(
            patch_size=16,
            stride_size=8,
            blur_threshold=0.5,
            cache_blur_maps=False
        )
        self.blur_filter = BlurFilter(self.config)
    
    def tearDown(self):
        """Clean up test data."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_mock_blur_heatmap(self, shape=(100, 100)):
        """Create mock blur heatmap data."""
        h, w = shape
        heatmap = np.random.rand(h, w).astype(np.float32)
        
        # Create some regions with high blur (should be filtered out)
        heatmap[20:40, 20:40] = 0.8  # High blur region
        heatmap[60:80, 60:80] = 0.2  # Low blur region (sharp)
        
        return heatmap
    
    def create_mock_segmentation(self, shape=(100, 100)):
        """Create mock segmentation mask."""
        mask = np.zeros(shape, dtype=np.int32)
        
        # Cell 1: in high blur region (should be filtered out)
        mask[25:35, 25:35] = 1
        
        # Cell 2: in low blur region (should pass)
        mask[65:75, 65:75] = 2
        
        # Cell 3: partially in blur region
        mask[45:55, 45:55] = 3
        
        return mask
    
    def test_filter_cells_by_blur(self):
        """Test blur-based cell filtering."""
        segmentation = self.create_mock_segmentation()
        blur_heatmap = self.create_mock_blur_heatmap()
        
        filtered_mask, quality_stats = self.blur_filter.filter_cells_by_blur(
            segmentation, blur_heatmap
        )
        
        # Should have same shape
        self.assertEqual(filtered_mask.shape, segmentation.shape)
        
        # Should have quality statistics
        self.assertIsInstance(quality_stats, pd.DataFrame)
        self.assertIn('passes_threshold', quality_stats.columns)
        self.assertIn('blur_intensity', quality_stats.columns)
        
        # Some cells should be filtered out
        n_original = len(np.unique(segmentation)) - 1  # Exclude background
        n_filtered = len(np.unique(filtered_mask)) - 1
        self.assertLessEqual(n_filtered, n_original)
    
    def test_filter_3d_stack(self):
        """Test filtering of 3D segmentation stack."""
        segmentation_stack = np.stack([
            self.create_mock_segmentation() for _ in range(3)
        ])
        blur_heatmaps = [
            self.create_mock_blur_heatmap() for _ in range(3)
        ]
        
        filtered_stack, quality_stats_list = self.blur_filter.filter_3d_stack(
            segmentation_stack, blur_heatmaps
        )
        
        # Should have same shape
        self.assertEqual(filtered_stack.shape, segmentation_stack.shape)
        
        # Should have quality stats for each z-slice
        self.assertEqual(len(quality_stats_list), 3)
        
        # Each should be a DataFrame
        for stats in quality_stats_list:
            self.assertIsInstance(stats, pd.DataFrame)
            self.assertIn('z', stats.columns)



class TestIntegratedPipeline(unittest.TestCase):
    """Tests for the complete integrated pipeline."""
    
    def setUp(self):
        """Set up test data."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = PipelineConfig(
            enable_blur_filtering=True,
            filter_before_tracking=True,
            save_intermediate_results=False
        )
        self.pipeline = CellTrackingPipeline(self.config)
    
    def tearDown(self):
        """Clean up test data."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_files(self):
        """Create test segmentation and image files."""
        # Create mock 3D segmentation
        segmentation = np.zeros((5, 60, 60), dtype=np.int32)
        for z in range(5):
            # Moving cell
            y, x = 20 + z * 2, 30 + z * 1
            segmentation[z, y:y+8, x:x+8] = 1
            
            # Stationary cell
            segmentation[z, 40:48, 40:48] = 2
        
        # Create mock image (just noise)
        image = np.random.randint(0, 255, (5, 60, 60), dtype=np.uint8)
        
        # Save files
        seg_path = Path(self.temp_dir) / "test_seg_3d.tif"
        img_path = Path(self.temp_dir) / "test_img_BF_3d.tif"
        
        tifffile.imwrite(str(seg_path), segmentation)
        tifffile.imwrite(str(img_path), image)
        
        return seg_path, img_path
    
    def test_process_single_file(self):
        """Test processing a single file through the pipeline."""
        seg_path, img_path = self.create_test_files()
        output_dir = Path(self.temp_dir) / "output"
        
        result = self.pipeline.process_single_file(
            seg_path, img_path, output_dir
        )
        
        # Should have processing results
        self.assertIn('input_segmentation', result)
        self.assertIn('final_output', result)
        self.assertIn('processing_steps', result)
        
        # Final output should exist
        final_output = Path(result['final_output'])
        self.assertTrue(final_output.exists())
        
        # Should have expected processing steps
        if self.config.enable_blur_filtering:
            self.assertIn('blur_filtering', result['processing_steps'])
        self.assertIn('tracking', result['processing_steps'])
    
    def test_batch_processing(self):
        """Test batch processing functionality."""
        # Clean up any existing files first
        existing_files = glob.glob(str(Path(self.temp_dir) / "test_*_3d.tif"))
        for f in existing_files:
            os.remove(f)
            
        # Create multiple test files
        test_files = []
        for i in range(3):
            seg, img = self.create_test_files()
            # Rename with unique names that follow the expected pattern
            new_seg = seg.parent / f"test_{i}_seg_3d.tif"
            new_img = img.parent / f"test_{i}_BF_3d.tif"
            seg.rename(new_seg)
            img.rename(new_img)
            test_files.append((new_seg, new_img))
        
        output_dir = Path(self.temp_dir) / "batch_output"
                
        results = self.pipeline.process_batch(
            self.temp_dir, output_dir, segmentation_pattern="*_seg_3d.tif", image_pattern="*_BF_3d.tif"
        )
        
        # Should process all files
        self.assertEqual(len(results), 3)
        
        # All should be successful
        successful = [r for r in results if 'error' not in r]
        self.assertEqual(len(successful), 3)
        
        # Summary file should exist
        summary_file = output_dir / "batch_processing_summary.json"
        self.assertTrue(summary_file.exists())


class TestTrackingProcessor(unittest.TestCase):
    """Tests for the complete tracking processor."""
    
    def setUp(self):
        """Set up test data."""
        self.temp_dir = tempfile.mkdtemp()
        self.processor = TrackingProcessor(TrackingProcessorConfig(
            blur_threshold=0.5,
            create_output_dirs=True,
            overwrite_existing=True
        ))
    
    def tearDown(self):
        """Clean up test data."""
        if hasattr(self, 'temp_dir') and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def create_test_segmentation(self, shape=(5, 100, 100)):
        """Create a test 3D segmentation with moving cells."""
        segmentation = np.zeros(shape, dtype=int)
        
        # Add cells that move across frames
        for z in range(shape[0]):
            # Cell 1: moves diagonally
            x1, y1 = 20 + z*3, 20 + z*2
            if x1 < shape[2]-15 and y1 < shape[1]-15:
                segmentation[z, y1:y1+10, x1:x1+10] = 1
            
            # Cell 2: moves in opposite direction
            x2, y2 = 70 - z*2, 70 - z*3
            if x2 > 15 and y2 > 15:
                segmentation[z, y2:y2+12, x2:x2+12] = 2
        
        return segmentation
    
    def test_get_label_centers_with_blur_filtering(self):
        """Test extraction of label centers with blur filtering."""
        # Create simple 2D mask
        mask = np.zeros((50, 50), dtype=int)
        mask[10:20, 10:20] = 1
        mask[30:40, 30:40] = 2
        
        # Test without blur filtering
        centers = self.processor.get_label_centers_with_blur_filtering(mask)
        self.assertEqual(len(centers), 2)
        self.assertIn('label', centers.columns)
        self.assertIn('x', centers.columns)
        self.assertIn('y', centers.columns)
        
        # Create sharpness image (note: lower values = sharper)
        sharpness = np.ones((50, 50)) * 0.8  # Blurry
        sharpness[10:20, 10:20] = 0.3  # Sharp
        
        # Test with blur filtering (should keep cell 1 with low blur value)
        centers_filtered = self.processor.get_label_centers_with_blur_filtering(
            mask, sharpness, blur_thresh=0.5, inv=False
        )
        self.assertEqual(len(centers_filtered), 1)
        # Check that we kept the sharp cell (label 1)
        kept_labels = set(centers_filtered['label'])
        self.assertIn(1, kept_labels)
    
    def test_track_3d_centers(self):
        """Test 3D cell tracking."""
        segmentation = self.create_test_segmentation()
        tracked_stack = self.processor.track_3d_centers(segmentation)
        
        self.assertEqual(tracked_stack.shape, segmentation.shape)
        self.assertEqual(tracked_stack.dtype, np.int32)
        
        # Check that we have tracked cells
        unique_ids = np.unique(tracked_stack)
        self.assertIn(0, unique_ids)  # Background
        self.assertGreater(len(unique_ids), 1)  # At least one tracked cell
    
    def test_process_single_file_mock(self):
        """Test processing a single file with mocked dependencies."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set up directories
            dirs = {
                'masks': Path(temp_dir) / "masks",
                'images': Path(temp_dir) / "images",
                'output': Path(temp_dir) / "output", 
                'blur': Path(temp_dir) / "blur"
            }
            
            for d in dirs.values():
                d.mkdir()
            
            mask_path = dirs['masks'] / "test_3d.tif"
            
            # Create and save test data
            segmentation = self.create_test_segmentation((3, 50, 50))
            tifffile.imwrite(str(mask_path), segmentation)
            
            # Mock blur heatmap computation
            with patch.object(self.processor, 'get_or_create_blur_heatmap') as mock_blur:
                mock_blur.return_value = np.random.rand(3, 50, 50)
                
                result = self.processor.process_single_file(
                    mask_path, dirs['images'], dirs['output'], dirs['blur']
                )
            
            self.assertEqual(result['status'], 'success')
            self.assertIn('output_path', result)
            self.assertEqual(result['input_shape'], (3, 50, 50))
            
            # Check output file was created
            output_path = Path(result['output_path'])
            self.assertTrue(output_path.exists())


if __name__ == '__main__':
    unittest.main()
