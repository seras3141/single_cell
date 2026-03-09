"""
Qt Backend Configuration for Visualization

This module ensures consistent Qt backend usage across the visualization tools.
Import this before importing any other visualization modules.
"""

import os
import sys
import warnings

def configure_qt_backend():
    """
    Configure Qt backend to use PyQt5 and avoid conflicts.
    
    This function should be called before importing napari or any Qt-based libraries.
    """
    # Set environment variables for Qt backend
    os.environ['QT_API'] = 'pyqt5'
    os.environ['NAPARI_QT_BACKEND'] = 'pyqt5'
    
    # Suppress Qt-related warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='qtpy')
    
    # Ensure PyQt5 is available
    try:
        import PyQt5
        from PyQt5 import QtCore
        print(f"Using PyQt5 version: {QtCore.QT_VERSION_STR}")
    except ImportError:
        print("Error: PyQt5 not found. Please install it with: pip install PyQt5")
        sys.exit(1)
    
    # Configure napari backend
    try:
        # Try to set napari backend before first import
        import qtpy
        print(f"QtPy using backend: {qtpy.API_NAME}")
        
        if qtpy.API_NAME != 'pyqt5':
            print(f"Warning: QtPy is using {qtpy.API_NAME} instead of PyQt5")
            print("This might cause conflicts. Consider reinstalling napari with PyQt5 support.")
            
    except ImportError:
        pass
    
    return True

def create_qt_application():
    """
    Create a QApplication instance with proper configuration.
    
    Returns:
        QApplication: Configured application instance
    """
    from PyQt5.QtWidgets import QApplication
    
    # Check if QApplication already exists
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
        app.setApplicationName("2D Segmentation Visualizer")
        app.setApplicationVersion("1.0")
    
    return app

# Configure backend on import
configure_qt_backend()
