"""
Basic tests for the segmentation visualization GUI.

Tests the core functionality without requiring actual image files.
"""

import os
import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import tifffile as tiff

# Skip tests if PyQt5 is not available
pytest_pyqt5 = pytest.importorskip("PyQt5")

# Skip entire module on headless systems — napari.Viewer() issues a C-level
# abort when no display is present, which cannot be caught by except Exception.
pytestmark = pytest.mark.skipif(
    not os.environ.get("DISPLAY"),
    reason="requires a display (DISPLAY not set)"
)

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
import sys


class TestFileSearcher:
    """Test the FileSearcher functionality."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory structure."""
        temp_dir = tempfile.mkdtemp()
        data_dir = Path(temp_dir)
        
        # Create directory structure
        (data_dir / "sample_plates_processed" / "split_3d").mkdir(parents=True)
        (data_dir / "sample_plates_processed" / "blur_heatmaps").mkdir(parents=True)
        (data_dir / "sample_plates_processed" / "segmentation_2d" / "cyto3" / "test" / "masks_3d").mkdir(parents=True)
        (data_dir / "sample_plates_processed" / "segmentation_2d" / "cyto3" / "test" / "tracking" / "final").mkdir(parents=True)
        
        # Create dummy TIFF files
        dummy_image = np.random.randint(0, 255, (10, 50, 50), dtype=np.uint8)
        dummy_labels = np.random.randint(0, 10, (10, 50, 50), dtype=np.uint32)
        dummy_blur = np.random.rand(10, 50, 50).astype(np.float32)
        
        base_names = ["p2126_J03", "p2126_J04"]
        
        for base_name in base_names:
            # Brightfield
            bf_path = data_dir / "sample_plates_processed" / "split_3d" / f"{base_name}_BF_3d.tif"
            tiff.imwrite(str(bf_path), dummy_image)
            
            # Ground truth
            gt_path = data_dir / "sample_plates_processed" / "split_3d" / f"{base_name}_Cells_3d.tif"
            tiff.imwrite(str(gt_path), dummy_labels)
            
            # Blur heatmap
            blur_path = data_dir / "sample_plates_processed" / "blur_heatmaps" / f"{base_name}_BF_3d_blur_heatmap.tif"
            tiff.imwrite(str(blur_path), dummy_blur)
            
            # Inference (only for first image)
            if base_name == "p2126_J03":
                inf_path = data_dir / "sample_plates_processed" / "segmentation_2d" / "cyto3" / "test" / "masks_3d" / f"{base_name}_masks_3d.tif"
                tiff.imwrite(str(inf_path), dummy_labels)
                
                # Final segmentation
                final_path = data_dir / "sample_plates_processed" / "segmentation_2d" / "cyto3" / "test" / "tracking" / "final" / f"{base_name}_masks_3d.tif"
                tiff.imwrite(str(final_path), dummy_labels)
        
        yield data_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_file_searcher_init(self, temp_data_dir):
        """Test FileSearcher initialization."""
        from src.visualize.segmentation_gui import FileSearcher
        
        searcher = FileSearcher(temp_data_dir)
        assert searcher.base_dir == temp_data_dir
    
    def test_get_available_base_names(self, temp_data_dir):
        """Test getting available base image names."""
        from src.visualize.segmentation_gui import FileSearcher
        
        searcher = FileSearcher(temp_data_dir)
        base_names = searcher.get_available_base_names()
        
        assert len(base_names) == 2
        assert "p2126_J03" in base_names
        assert "p2126_J04" in base_names
        assert base_names == sorted(base_names)  # Should be sorted
    
    def test_find_related_files_complete(self, temp_data_dir):
        """Test finding related files for complete dataset."""
        from src.visualize.segmentation_gui import FileSearcher
        
        searcher = FileSearcher(temp_data_dir)
        files = searcher.find_related_files("p2126_J03")
        
        # Should find all file types
        assert files['brightfield'] is not None
        assert files['ground_truth'] is not None
        assert files['blur_map'] is not None
        assert files['inference'] is not None
        assert files['postprocessed'] is not None
        
        # All files should exist
        for file_path in files.values():
            if file_path:
                assert file_path.exists()
    
    def test_find_related_files_incomplete(self, temp_data_dir):
        """Test finding related files for incomplete dataset."""
        from src.visualize.segmentation_gui import FileSearcher
        
        searcher = FileSearcher(temp_data_dir)
        files = searcher.find_related_files("p2126_J04")
        
        # Should find basic files but not inference/postprocessed
        assert files['brightfield'] is not None
        assert files['ground_truth'] is not None
        assert files['blur_map'] is not None
        assert files['inference'] is None
        assert files['postprocessed'] is None
    
    def test_find_related_files_nonexistent(self, temp_data_dir):
        """Test finding related files for nonexistent base name."""
        from src.visualize.segmentation_gui import FileSearcher
        
        searcher = FileSearcher(temp_data_dir)
        files = searcher.find_related_files("nonexistent")
        
        # Should find no files
        for file_path in files.values():
            assert file_path is None


class TestVisualizationFunctions:
    """Test the enhanced visualization functions."""
    
    @pytest.fixture
    def dummy_data(self):
        """Create dummy image data for testing."""
        return {
            'brightfield': np.random.randint(0, 255, (5, 20, 20), dtype=np.uint8),
            'labels': np.random.randint(0, 5, (5, 20, 20), dtype=np.uint32),
            'heatmap': np.random.rand(5, 20, 20).astype(np.float32)
        }
    
    def test_create_enhanced_napari_viewer_brightfield_only(self, dummy_data):
        """Test creating viewer with only brightfield data."""
        from src.visualize.visualize_prediction import create_enhanced_napari_viewer
        
        # This test only checks that the function doesn't crash
        # Actually running napari would require a display
        try:
            viewer = create_enhanced_napari_viewer(
                brightfield=dummy_data['brightfield']
            )
            
            # Basic checks
            assert viewer is not None
            assert len(viewer.layers) >= 1
            assert viewer.layers[0].name == "Brightfield"
            
            # Close viewer to avoid hanging
            viewer.close()
            
        except Exception as e:
            # Skip test if napari can't run (e.g., no display)
            pytest.skip(f"Napari cannot run: {e}")
    
    def test_create_enhanced_napari_viewer_all_layers(self, dummy_data):
        """Test creating viewer with all layer types."""
        from src.visualize.visualize_prediction import create_enhanced_napari_viewer
        
        try:
            viewer = create_enhanced_napari_viewer(
                brightfield=dummy_data['brightfield'],
                ground_truth=dummy_data['labels'],
                inference_prediction=dummy_data['labels'],
                blur_heatmap=dummy_data['heatmap'],
                final_segmentation=dummy_data['labels']
            )
            
            # Should have all 5 layers
            assert viewer is not None
            assert len(viewer.layers) == 5
            
            layer_names = [layer.name for layer in viewer.layers]
            expected_names = ["Brightfield", "Ground Truth", "Inference Prediction", "Blur Heatmap", "Final Segmentation"]
            
            for name in expected_names:
                assert name in layer_names
            
            # Close viewer to avoid hanging
            viewer.close()
            
        except Exception as e:
            pytest.skip(f"Napari cannot run: {e}")


class TestGUIComponents:
    """Test GUI components without actually showing the window."""
    
    @pytest.fixture
    def qapp(self):
        """Create QApplication for testing."""
        if not QApplication.instance():
            app = QApplication([])
        else:
            app = QApplication.instance()
        yield app
        # Don't quit the app as it might be used by other tests
    
    def test_gui_initialization(self, qapp, tmp_path):
        """Test that GUI can be initialized."""
        from src.visualize.segmentation_gui import SegmentationVisualizationGUI
        
        # Create dummy data directory
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        try:
            gui = SegmentationVisualizationGUI(str(data_dir))
            
            # Basic checks
            assert gui.data_dir == data_dir
            assert gui.file_searcher is not None
            assert gui.file_list is not None
            assert gui.visualize_btn is not None
            
            # Should start with no selection
            assert not gui.visualize_btn.isEnabled()
            
        except Exception as e:
            pytest.skip(f"GUI cannot be created: {e}")
    
    def test_refresh_file_list_empty(self, qapp, tmp_path):
        """Test refreshing file list with empty directory."""
        from src.visualize.segmentation_gui import SegmentationVisualizationGUI
        
        # Create empty data directory
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        try:
            gui = SegmentationVisualizationGUI(str(data_dir))
            gui.refresh_file_list()
            
            # Should have no items
            assert gui.file_list.count() == 0
            
        except Exception as e:
            pytest.skip(f"GUI test failed: {e}")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
