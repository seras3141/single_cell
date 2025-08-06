#!/usr/bin/env python3
"""
Launcher for the 2D Segmentation Visualization GUI.

This launcher handles Qt backend configuration and provides comprehensive
dependency checking and error handling.
"""

import sys
import os
from pathlib import Path

def setup_environment():
    """Set up environment variables to avoid Qt conflicts."""
    # Force PyQt5 backend
    os.environ['QT_API'] = 'pyqt5'
    os.environ['NAPARI_QT_BACKEND'] = 'pyqt5'
    
    # Disable PySide6 if it's installed
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = ''
    
    # Add src to path
    current_dir = Path(__file__).parent
    src_dir = current_dir / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []
    
    try:
        import PyQt5
        from PyQt5 import QtCore
        print(f"✓ PyQt5 {QtCore.QT_VERSION_STR} found")
    except ImportError:
        missing.append("PyQt5")
    
    try:
        import napari
        print(f"✓ napari {napari.__version__} found")
    except ImportError:
        missing.append("napari[all]")
    
    try:
        import tifffile
        print(f"✓ tifffile {tifffile.__version__} found")
    except ImportError:
        missing.append("tifffile")
    
    if missing:
        print(f"\n❌ Missing dependencies: {', '.join(missing)}")
        print("Install with:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    return True

import argparse
from src.utils.config import ConfigManager, get_paths_from_config

def parse_args():
    """Parse command-line arguments for configuration."""
    parser = argparse.ArgumentParser(description="2D Segmentation Visualization GUI Launcher")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a configuration file (optional)"
    )
    return parser.parse_args()


def main():
    """Main launcher function."""
    print("=== 2D Segmentation Visualization GUI ===")
    print("Setting up Qt backend configuration...")
    
    # Set up environment first
    setup_environment()
    
    # Check dependencies
    if not check_dependencies():
        input("Press Enter to exit...")
        sys.exit(1)
    
    print("✓ Dependencies OK")
    print("✓ Qt backend configured")
    
    try:
        # Import and run GUI
        print("Loading GUI components...")
        from src.visualize.segmentation_gui import SegmentationVisualizationGUI
        from PyQt5.QtWidgets import QApplication
        
        # Create application
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
            app.setApplicationName("2D Segmentation Visualizer")
            app.setApplicationVersion("1.0")

        args = parse_args()
        if args.config:
            print(f"Using configuration file: {args.config}")
            # Load configuration logic here if needed
            config = ConfigManager(args.config)
            paths = get_paths_from_config(config)
        else:
            print("No configuration file provided, using defaults.")
            paths = {}
        
        # Check data directory
        current_dir = Path(__file__).parent
        data_dir = current_dir.parent / "data"
        if not data_dir.exists():
            print(f"⚠️  Warning: Data directory not found at {data_dir}")
            print("   You can change the directory from within the GUI.")
        else:
            print(f"✓ Data directory found: {data_dir}")
        
        # Create and show GUI
        print("Launching GUI...")
        window = SegmentationVisualizationGUI(str(data_dir) if data_dir.exists() else None, custom_paths=paths)
        window.show()
        
        print("✓ GUI launched successfully!")
        print("   - Select an image from the list")
        print("   - Configure visualization options")
        print("   - Click 'Visualize in Napari' to view in 2D")
        
        sys.exit(app.exec())
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you're in the correct directory")
        print("2. Install missing dependencies")
        print("3. Try: pip install -r requirements_gui.txt")
        input("Press Enter to exit...")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Error launching GUI: {e}")
        print(f"Error type: {type(e).__name__}")
        
        # Additional troubleshooting for Qt issues
        if "Qt" in str(e) or "qtpy" in str(e):
            print("\nQt Backend Issue Detected:")
            print("1. Try uninstalling PySide6: pip uninstall PySide6")
            print("2. Reinstall napari: pip install --force-reinstall napari[all]")
            print("3. Make sure only PyQt5 is installed for Qt")
        
        input("Press Enter to exit...")
        sys.exit(1)

if __name__ == "__main__":
    main()
