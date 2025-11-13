"""Single Cell Analysis Pipeline - Core Package.

A modular pipeline for single-cell segmentation, tracking, and feature extraction
using Cellpose models.

The library provides:
- Cellpose 2D segmentation (training and prediction)
- Configurable model training with presets for different cell types
- Evaluation and performance metrics
- 3D cell tracking with blur-based quality filtering
- Utility functions for image processing and quality assessment
"""

__version__ = "0.1.0"

from . import utils
# from . import inference
from . import preprocessing
# from . import postprocessing

__all__ = ["utils", "inference", "preprocessing", "postprocessing"]
