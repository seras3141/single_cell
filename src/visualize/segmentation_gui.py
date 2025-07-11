"""
Segmentation Visualization GUI

A PyQt5-based GUI for visualizing segmentation results including:
- Brightfield images
- Ground truth segmentation 
- Blur heatmaps
- Inference predictions
- Final postprocessed segmentation

All visualizations are displayed in 2D using napari with z-slider navigation.
"""

# Import Qt configuration first to avoid conflicts
from .qt_config import configure_qt_backend, create_qt_application

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QListWidget, QListWidgetItem, QPushButton, QLabel, QGroupBox,
    QCheckBox, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit,
    QSplitter, QMessageBox, QProgressBar, QFrame, QGridLayout,
    QFileDialog
)
from PyQt5.QtCore import Qt, QThread, QObject, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QIcon

import numpy as np
import tifffile as tiff
import napari


class FileSearcher:
    """Helper class to search for related files based on base image name."""
    
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        
    def find_related_files(self, base_name: str) -> Dict[str, Optional[Path]]:
        """
        Find all related files for a given base name (e.g., 'p2126_J03').
        
        Returns:
            Dictionary with keys: 'brightfield', 'ground_truth', 'blur_map', 
                                'inference', 'postprocessed'
        """
        files = {
            'brightfield': None,
            'ground_truth': None,
            'blur_map': None,
            'inference': None,
            'postprocessed': None
        }
        
        # Brightfield image
        bf_path = self.base_dir / "sample_plates_processed" / "split_3d" / f"{base_name}_BF_3d.tif"
        if bf_path.exists():
            files['brightfield'] = bf_path
            
        # Ground truth segmentation
        gt_path = self.base_dir / "sample_plates_processed" / "split_3d" / f"{base_name}_Cells_3d.tif"
        if gt_path.exists():
            files['ground_truth'] = gt_path
            
        # Blur heatmap
        blur_path = self.base_dir / "sample_plates_processed" / "blur_heatmaps" / f"{base_name}_BF_3d_blur_heatmap.tif"
        if blur_path.exists():
            files['blur_map'] = blur_path
            
        # Inference prediction
        inference_path = self.base_dir / "sample_plates_processed" / "segmentation_2d" / "cyto3" / "test" / "masks_3d" / f"{base_name}_masks_3d.tif"
        if inference_path.exists():
            files['inference'] = inference_path
            
        # Final postprocessed segmentation
        final_path = self.base_dir / "sample_plates_processed" / "segmentation_2d" / "cyto3" / "test" / "tracking" / "final" / f"{base_name}_masks_3d.tif"
        if final_path.exists():
            files['postprocessed'] = final_path
            
        return files
    
    def get_available_base_names(self) -> List[str]:
        """Get list of available base image names."""
        base_names = set()
        
        # Look in split_3d for BF images
        split_3d_dir = self.base_dir / "sample_plates_processed" / "split_3d"
        if split_3d_dir.exists():
            for file_path in split_3d_dir.glob("*_BF_3d.tif"):
                base_name = file_path.stem.replace("_BF_3d", "")
                base_names.add(base_name)
                
        return sorted(list(base_names))


class NapariLauncher(QObject):
    """
    Napari launcher that runs in the main thread to avoid Qt threading issues.
    
    Instead of using a separate thread for napari (which causes Qt issues),
    this class launches napari in the main thread using QTimer.
    """
    
    finished = pyqtSignal()
    error = pyqtSignal(str)
    
    def __init__(self, files: Dict[str, Path], visualization_config: Dict):
        super().__init__()
        self.files = files
        self.config = visualization_config
        
    def launch_napari(self):
        """Launch napari visualization in the main thread."""
        try:
            # Import the enhanced napari visualization function
            from .visualize_prediction import create_enhanced_napari_viewer
            
            # Prepare file paths
            file_args = {}
            if self.files.get('brightfield'):
                file_args['brightfield'] = str(self.files['brightfield'])
            if self.files.get('ground_truth') and self.config.get('show_ground_truth', True):
                file_args['ground_truth'] = str(self.files['ground_truth'])
            if self.files.get('inference') and self.config.get('show_inference', True):
                file_args['inference_prediction'] = str(self.files['inference'])
            if self.files.get('blur_map') and self.config.get('show_blur_map', True):
                file_args['blur_heatmap'] = str(self.files['blur_map'])
            if self.files.get('postprocessed') and self.config.get('show_postprocessed', True):
                file_args['final_segmentation'] = str(self.files['postprocessed'])
            
            # Prepare layer configuration
            layer_config = {
                'ground_truth': {'opacity': self.config.get('ground_truth_opacity', 0.6)},
                'inference': {'opacity': self.config.get('inference_opacity', 0.5)},
                'blur_heatmap': {'opacity': self.config.get('blur_opacity', 0.3)},
                'final_segmentation': {'opacity': self.config.get('final_opacity', 0.7)}
            }
            
            # Create the napari viewer
            viewer = create_enhanced_napari_viewer(
                layer_config=layer_config,
                viewer_title="2D Segmentation Visualization",
                **file_args
            )
            
            if viewer is None:
                raise ValueError("Failed to create napari viewer")
            
            self.finished.emit()
            
        except Exception as e:
            self.error.emit(str(e))


class SegmentationVisualizationGUI(QMainWindow):
    """Main GUI window for segmentation visualization."""
    
    def __init__(self, data_dir: str = None):
        super().__init__()
        
        # Set data directory
        if data_dir is None:
            # Default to the data directory relative to this script
            current_dir = Path(__file__).parent.parent.parent
            self.data_dir = current_dir / "data"
        else:
            self.data_dir = Path(data_dir)
            
        self.file_searcher = FileSearcher(self.data_dir)
        self.current_files = {}
        
        self.init_ui()
        self.refresh_file_list()
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("2D Segmentation Visualization Tool")
        self.setGeometry(100, 100, 1000, 700)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - File selection and controls
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        # Right panel - File information and status
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)
        
        # Set splitter proportions
        splitter.setSizes([400, 600])
        
    def create_left_panel(self) -> QWidget:
        """Create the left panel with file list and controls."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Data directory selection
        dir_group = QGroupBox("Data Directory")
        dir_layout = QVBoxLayout(dir_group)
        
        self.dir_label = QLabel(str(self.data_dir))
        self.dir_label.setWordWrap(True)
        dir_layout.addWidget(self.dir_label)
        
        change_dir_btn = QPushButton("Change Directory")
        change_dir_btn.clicked.connect(self.change_data_directory)
        dir_layout.addWidget(change_dir_btn)
        
        layout.addWidget(dir_group)
        
        # File list
        file_group = QGroupBox("Available Images")
        file_layout = QVBoxLayout(file_group)
        
        self.file_list = QListWidget()
        self.file_list.itemSelectionChanged.connect(self.on_file_selected)
        file_layout.addWidget(self.file_list)
        
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_file_list)
        file_layout.addWidget(refresh_btn)
        
        layout.addWidget(file_group)
        
        # Visualization options
        vis_group = QGroupBox("Visualization Options")
        vis_layout = QGridLayout(vis_group)
        
        self.show_ground_truth = QCheckBox("Ground Truth")
        self.show_ground_truth.setChecked(True)
        vis_layout.addWidget(self.show_ground_truth, 0, 0)
        
        self.show_blur_map = QCheckBox("Blur Heatmap")
        self.show_blur_map.setChecked(True)
        vis_layout.addWidget(self.show_blur_map, 0, 1)
        
        self.show_inference = QCheckBox("Inference")
        self.show_inference.setChecked(True)
        vis_layout.addWidget(self.show_inference, 1, 0)
        
        self.show_postprocessed = QCheckBox("Final Result")
        self.show_postprocessed.setChecked(True)
        vis_layout.addWidget(self.show_postprocessed, 1, 1)
        
        # Opacity controls
        vis_layout.addWidget(QLabel("Opacity Controls:"), 2, 0, 1, 2)
        
        vis_layout.addWidget(QLabel("Ground Truth:"), 3, 0)
        self.gt_opacity = QDoubleSpinBox()
        self.gt_opacity.setRange(0.0, 1.0)
        self.gt_opacity.setSingleStep(0.1)
        self.gt_opacity.setValue(0.6)
        vis_layout.addWidget(self.gt_opacity, 3, 1)
        
        vis_layout.addWidget(QLabel("Blur Map:"), 4, 0)
        self.blur_opacity = QDoubleSpinBox()
        self.blur_opacity.setRange(0.0, 1.0)
        self.blur_opacity.setSingleStep(0.1)
        self.blur_opacity.setValue(0.3)
        vis_layout.addWidget(self.blur_opacity, 4, 1)
        
        vis_layout.addWidget(QLabel("Inference:"), 5, 0)
        self.inference_opacity = QDoubleSpinBox()
        self.inference_opacity.setRange(0.0, 1.0)
        self.inference_opacity.setSingleStep(0.1)
        self.inference_opacity.setValue(0.5)
        vis_layout.addWidget(self.inference_opacity, 5, 1)
        
        vis_layout.addWidget(QLabel("Final Result:"), 6, 0)
        self.final_opacity = QDoubleSpinBox()
        self.final_opacity.setRange(0.0, 1.0)
        self.final_opacity.setSingleStep(0.1)
        self.final_opacity.setValue(0.7)
        vis_layout.addWidget(self.final_opacity, 6, 1)
        
        layout.addWidget(vis_group)
        
        # Visualize button
        self.visualize_btn = QPushButton("Visualize in Napari")
        self.visualize_btn.setEnabled(False)
        self.visualize_btn.clicked.connect(self.visualize_selected)
        layout.addWidget(self.visualize_btn)
        
        return panel
        
    def create_right_panel(self) -> QWidget:
        """Create the right panel with file information."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Selected file info
        info_group = QGroupBox("Selected Image Information")
        info_layout = QVBoxLayout(info_group)
        
        self.selected_label = QLabel("No image selected")
        self.selected_label.setFont(QFont("Arial", 12, QFont.Bold))
        info_layout.addWidget(self.selected_label)
        
        # File availability table
        self.file_info = QTextEdit()
        self.file_info.setMaximumHeight(200)
        self.file_info.setReadOnly(True)
        info_layout.addWidget(self.file_info)
        
        layout.addWidget(info_group)
        
        # Status and log
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout(status_group)
        
        self.status_label = QLabel("Ready")
        status_layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        status_layout.addWidget(self.progress_bar)
        
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(150)
        self.log_text.setReadOnly(True)
        status_layout.addWidget(self.log_text)
        
        layout.addWidget(status_group)
        
        return panel
    
    def change_data_directory(self):
        """Change the data directory."""
        new_dir = QFileDialog.getExistingDirectory(
            self, 
            "Select Data Directory",
            str(self.data_dir)
        )
        
        if new_dir:
            self.data_dir = Path(new_dir)
            self.dir_label.setText(str(self.data_dir))
            self.file_searcher = FileSearcher(self.data_dir)
            self.refresh_file_list()
    
    def refresh_file_list(self):
        """Refresh the list of available base images."""
        self.file_list.clear()
        self.log_message("Scanning for available images...")
        
        try:
            base_names = self.file_searcher.get_available_base_names()
            
            if not base_names:
                self.log_message("No images found in the specified directory.")
                self.status_label.setText("No images found")
                return
            
            for base_name in base_names:
                item = QListWidgetItem(base_name)
                
                # Check file availability
                files = self.file_searcher.find_related_files(base_name)
                available_count = sum(1 for f in files.values() if f is not None)
                
                item.setToolTip(f"Available files: {available_count}/5")
                self.file_list.addItem(item)
            
            self.log_message(f"Found {len(base_names)} images.")
            self.status_label.setText(f"Found {len(base_names)} images")
            
        except Exception as e:
            self.log_message(f"Error scanning directory: {str(e)}")
            self.status_label.setText("Error scanning directory")
    
    def on_file_selected(self):
        """Handle file selection."""
        current_item = self.file_list.currentItem()
        if not current_item:
            self.visualize_btn.setEnabled(False)
            self.selected_label.setText("No image selected")
            self.file_info.clear()
            return
        
        base_name = current_item.text()
        self.selected_label.setText(f"Selected: {base_name}")
        
        # Find related files
        self.current_files = self.file_searcher.find_related_files(base_name)
        
        # Update file info display
        info_text = "File Availability:\n\n"
        
        file_types = {
            'brightfield': 'Brightfield Image',
            'ground_truth': 'Ground Truth Segmentation',
            'blur_map': 'Blur Heatmap',
            'inference': 'Inference Prediction',
            'postprocessed': 'Final Segmentation'
        }
        
        available_files = 0
        for key, description in file_types.items():
            file_path = self.current_files.get(key)
            if file_path and file_path.exists():
                info_text += f"✓ {description}\n   {file_path.name}\n\n"
                available_files += 1
            else:
                info_text += f"✗ {description}\n   Not found\n\n"
        
        self.file_info.setText(info_text)
        
        # Enable visualize button if at least brightfield is available
        self.visualize_btn.setEnabled(
            self.current_files.get('brightfield') is not None
        )
        
        self.log_message(f"Selected {base_name}: {available_files}/5 files available")
    
    def visualize_selected(self):
        """Launch napari visualization for selected image."""
        if not self.current_files.get('brightfield'):
            QMessageBox.warning(
                self, 
                "No Data", 
                "No brightfield image available for visualization."
            )
            return
        
        # Prepare visualization configuration
        config = {
            'show_ground_truth': self.show_ground_truth.isChecked(),
            'show_blur_map': self.show_blur_map.isChecked(),
            'show_inference': self.show_inference.isChecked(),
            'show_postprocessed': self.show_postprocessed.isChecked(),
            'ground_truth_opacity': self.gt_opacity.value(),
            'blur_opacity': self.blur_opacity.value(),
            'inference_opacity': self.inference_opacity.value(),
            'final_opacity': self.final_opacity.value()
        }
        
        self.status_label.setText("Launching napari...")
        self.log_message("Starting 2D visualization in napari...")
        
        # Create napari launcher and run it in main thread using QTimer
        self.napari_launcher = NapariLauncher(self.current_files, config)
        self.napari_launcher.finished.connect(self.on_visualization_finished)
        self.napari_launcher.error.connect(self.on_visualization_error)
        
        # Use QTimer to delay the napari launch to avoid blocking
        QTimer.singleShot(100, self.napari_launcher.launch_napari)
    
    def on_visualization_finished(self):
        """Handle visualization completion."""
        self.status_label.setText("Visualization completed")
        self.log_message("Napari visualization completed.")
    
    def on_visualization_error(self, error_msg: str):
        """Handle visualization error."""
        self.status_label.setText("Visualization failed")
        self.log_message(f"Visualization error: {error_msg}")
        QMessageBox.critical(
            self,
            "Visualization Error",
            f"Failed to create visualization:\n{error_msg}"
        )
    
    def log_message(self, message: str):
        """Add message to log."""
        self.log_text.append(f"[{QTimer().remainingTime()}] {message}")


def main():
    """Main function to run the GUI."""
    # Create QApplication with proper configuration
    app = create_qt_application()
    
    # Create and show main window
    window = SegmentationVisualizationGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
