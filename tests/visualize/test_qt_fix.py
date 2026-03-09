#!/usr/bin/env python3
"""
Quick test to verify the Qt backend fix.

This script tests that PyQt5 and napari work together without threading issues.
"""

import os
import sys

# Set Qt backend before any imports
os.environ['QT_API'] = 'pyqt5'

def test_qt_backend():
    """Test Qt backend configuration."""
    print("=== Qt Backend Test ===")
    
    try:
        # Test PyQt5 import
        from PyQt5.QtWidgets import QApplication
        from PyQt5 import QtCore
        print(f"✓ PyQt5 version: {QtCore.QT_VERSION_STR}")
        
        # Test qtpy backend
        import qtpy
        print(f"✓ QtPy backend: {qtpy.API_NAME}")
        
        if qtpy.API_NAME != 'pyqt5':
            print(f"⚠️  Warning: Expected PyQt5, got {qtpy.API_NAME}")
        
        # Test napari import
        import napari
        print(f"✓ Napari version: {napari.__version__}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_gui_creation():
    """Test basic GUI creation without showing it."""
    print("\n=== GUI Creation Test ===")
    
    try:
        from PyQt5.QtWidgets import QApplication
        import sys
        
        # Create application
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        print("✓ QApplication created successfully")
        
        # Test napari viewer creation (but don't show it)
        import napari
        viewer = napari.Viewer(show=False)
        print("✓ Napari viewer created successfully")
        
        # Close viewer
        viewer.close()
        print("✓ Napari viewer closed successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ GUI creation error: {e}")
        return False

def test_visualization_gui():
    """Test the segmentation GUI initialization."""
    print("\n=== Segmentation GUI Test ===")
    
    try:
        # Add src to path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.join(current_dir, "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        
        from visualize.segmentation_gui import SegmentationVisualizationGUI
        from PyQt5.QtWidgets import QApplication
        
        # Create application
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        # Create GUI (but don't show it)
        gui = SegmentationVisualizationGUI()
        print("✓ SegmentationVisualizationGUI created successfully")
        
        # Test basic functionality
        assert gui.file_list is not None
        assert gui.visualize_btn is not None
        print("✓ GUI components initialized correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Segmentation GUI error: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing Qt backend and GUI fixes...\n")
    
    success = True
    
    # Test 1: Qt backend
    if not test_qt_backend():
        success = False
    
    # Test 2: Basic GUI creation
    if not test_gui_creation():
        success = False
    
    # Test 3: Segmentation GUI
    if not test_visualization_gui():
        success = False
    
    print("\n" + "="*50)
    if success:
        print("🎉 All tests passed! The Qt backend fix is working.")
        print("\nYou can now run:")
        print("  python launch_gui_safe.py")
    else:
        print("❌ Some tests failed. Check the error messages above.")
        print("\nTroubleshooting:")
        print("1. pip install PyQt5 napari[all]")
        print("2. pip uninstall PySide6 (if installed)")
        print("3. Check TROUBLESHOOTING_QT.md for more help")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
