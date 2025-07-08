# Single Cell Analysis Pipeline

A comprehensive, modular pipeline for single-cell segmentation, tracking, and feature extraction in brightfield microscopy images using Cellpose models.


<!-- ### Training a Model
```python
from models.config_presets import create_training_config_from_preset
from models.cellpose_trainer import CellposeTrainer

# Create configuration using preset
config = create_training_config_from_preset(
    "mammalian_cells_cellpose",
    train_dir="path/to/training/data",
    test_dir="path/to/test/data", 
    output_dir="path/to/save/models"
)

# Train the model
trainer = CellposeTrainer(config)
trainer.train()
```

### Evaluating a Model
```python
from models.config_presets import create_evaluation_config_from_preset
from models.cellpose_trainer import CellposeEvaluator

# Create evaluation configuration
config = create_evaluation_config_from_preset(
    "default",
    test_dir="path/to/test/data",
    model_path="cyto2",  # or path to custom model
    output_dir="path/to/results"
)

# Run evaluation
evaluator = CellposeEvaluator(config)
results = evaluator.evaluate()
``` -->

## Overview

This project provides a complete, modular workflow for:
- **🎯 2D/3D Cell Segmentation**: Using Cellpose models with preset configurations
- **📊 Modular Training System**: Easy-to-use trainers with configuration presets
- **🔍 Model Evaluation**: Comprehensive evaluation with multiple preset configurations  
- **🔄 Cell Tracking**: Tracking cells across z-stacks using trackpy
- **📈 Feature Extraction**: PyRadiomics-based feature extraction for cell analysis
- **📁 Data Management**: Standardized file naming and dataset organization
- **✅ Quality Control**: Blur detection and filtering for improved segmentation
- **👁️ Visualization**: 3D/4D visualization tools using Napari

## Project Structure

```
single_cell/
├── src/    # source files
│   ├── models/ #  Model implementations
│   ├── preprocessing/  # Data preprocessing pipeline
│   ├── utils/  # Other utility modules
│   ├── data/                          #  │   └── visualize/                    # 
├── examples/   # Usage examples and tutorials
├── docs/   # Documentation
├── tests/  # Test suite
├── config/ # Configuration files
├── scripts/    # Standalone scripts
└── github/ # 📦 Git submodules (cellpose, omnipose)
```

## Installation

### Prerequisites
- Python 3.10+ (tested with Python 3.10.18)
- CUDA-capable GPU (optional, for faster processing)
- **Verified Environment**: Windows 10/11, Anaconda/Miniconda

### Environment Setup (Tested & Validated ✅)

1. **Create conda environment:**
```bash
conda create -n cellpose-env python=3.10
conda activate cellpose-env
```

2. **Install PyTorch with CUDA support:**
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

3. **Clone and setup the repository:**
```bash
git clone <repository-url>
cd single_cell
git submodule update --init --recursive  # Initialize Cellpose submodule
```

4. **Install Cellpose:**
```bash
# Install Cellpose in development mode
pip install -e ./github/cellpose

# Install additional dependencies
pip install -r requirements.txt
```

5. **Verify installation:**
```bash
# Run the test suite to verify everything works
python tests/test_training_system.py
python tests/test_integration.py
```


## 🚀 Quick Start

### 1. Preprocess Raw Data

> **Note:** For detailed CLI and API usage for preprocessing, see [`preprocessing/README.md`](preprocessing/README.md).


### 2. Train Models

```python
from models.config_presets import create_training_config_from_preset
from models.cellpose_trainer import CellposeTrainer

# Create configuration using preset
config = create_training_config_from_preset(
    "mammalian_cells_cellpose",
    train_dir="data/processed/split/train",
    test_dir="data/processed/split/test", 
    output_dir="models/cellpose"
)

# Train the model
trainer = CellposeTrainer(config)
trainer.train()
```

### 3. Evaluate Models

```python
from models.config_presets import create_evaluation_config_from_preset
from models.cellpose_trainer import CellposeEvaluator

# Create evaluation configuration
config = create_evaluation_config_from_preset(
    "default",
    test_dir="data/processed/split/test",
    model_path="cyto2",  # or path to custom model
    output_dir="results/evaluation"
)

# Run evaluation
evaluator = CellposeEvaluator(config)
results = evaluator.evaluate()
```

### 4. Run Predictions

#### 2D Cell Segmentation
```bash
python src/cellpose_2D_prediction.py --flow-threshold 0.4
```

#### 3D Cell Segmentation
```bash
python src/models/cellpose/cellpose_3D_train.py --input-dir data/test --output-dir results/3d_predictions
```

### 5. Cell Tracking
```python
from src.util.track_cells import track_3d_centers
import tifffile

# Load 3D segmentation
segmentation = tifffile.imread("results/3d_predictions/sample.tif")

# Track cells across z-stacks
tracked = track_3d_centers(segmentation)

# Save results
tifffile.imwrite("results/tracked/sample_tracked.tif", tracked)
```

### 6. Feature Extraction
```python
from src.util.feature_extractor import extract_features

# Extract PyRadiomics features
features = extract_features("image.tif", "mask.tif")
```

### 7. Visualization
```python
from src.visualize.view_3d_tiff import view_3d_data

# Visualize 3D data in Napari
view_3d_data("results/3d_predictions/")
```

## Configuration

Edit `config/config.yaml` to customize pipeline parameters:
```yaml
segmentation:
  model_type: "cyto3"
  flow_threshold: 0.4
  min_cell_size: 30

tracking:
  search_range: 5
  memory: 1
  blur_threshold: 0.5
```

## Supported File Formats

- **Input**: Multi-well plate TIFF files with standardized naming
- **Output**: Segmentation masks, tracked labels, feature tables (CSV)
- **Visualization**: 3D/4D TIFF stacks compatible with Napari

## Examples

### 📚 Complete Examples Available
- **`examples/complete_usage_examples.py`**: Comprehensive usage demonstrations
- **`examples/training_examples.py`**: Basic training workflow examples  
- **`docs/TRAINING_GUIDE.md`**: Detailed step-by-step training guide

### Quick Examples

#### Modular Training with Presets
```python
# Train a Cellpose model for mammalian cells
from models.config_presets import create_training_config_from_preset
from models.cellpose_trainer import CellposeTrainer

config = create_training_config_from_preset(
    "mammalian_cells_cellpose",
    train_dir="data/training",
    test_dir="data/test",
    output_dir="models/cellpose"
)
trainer = CellposeTrainer(config)
trainer.train()
```

#### Modular Evaluation
```python
# Evaluate model performance
from models.config_presets import create_evaluation_config_from_preset  
from models.cellpose_trainer import CellposeEvaluator

config = create_evaluation_config_from_preset(
    "default",
    test_dir="data/test",
    model_path="cyto2",  # or path to custom model
    output_dir="results/evaluation"
)
evaluator = CellposeEvaluator(config)
results = evaluator.evaluate()
```

## 🧪 Testing & Validation

The system includes comprehensive testing to ensure reliability:

### Test Suite
```bash
# Run all modular system tests (19 tests)
python tests/test_training_system.py

# Run integration tests
python tests/test_integration.py

# Run preprocessing tests
python -m pytest tests/preprocessing/ -v  # 11 tests total

# Run utility tests  
python -m pytest tests/utils/ -v          # 19 tests total

# Verify GPU/CUDA functionality
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### Test Coverage
- **Configuration Management**: Preset creation, validation, parameter overrides
- **Trainer Initialization**: Cellpose trainer setup and configuration
- **Evaluator Setup**: Model evaluation configuration
- **Preprocessing Pipeline**: Blur analysis and dataset splitting (11 tests)
- **Utility Functions**: File handling, blur measurement (19 tests)
- **Data Loading**: File discovery and data validation
- **GPU Support**: CUDA availability and PyTorch compatibility
- **End-to-End Workflows**: Complete training and evaluation pipelines

### Validation Results ✅
- **All 19 unit tests pass** (training system)
- **All 11 preprocessing tests pass** (blur analysis + dataset splitting)
- **All 19 utility tests pass** (blur measurement)
- **All integration tests pass** 
- **Environment verified**: Python 3.10.18, PyTorch 1.13.1+cu117, CUDA 11.7
- **Models verified**: Cellpose 4.0.6 installed and tested
- **GPU acceleration confirmed**: CUDA support working

## 🏆 Project Status

### ✅ **COMPLETED & TESTED**
- **Modular Architecture**: Complete refactoring with abstract base classes
- **Preprocessing Pipeline**: Blur analysis & group-aware dataset splitting (NEW)
- **Utility Consolidation**: Unified file handling, blur measurement refactoring (NEW)
- **Environment Setup**: PyTorch (CUDA 11.7), Cellpose 4.0.6 installed & verified
- **Configuration System**: 7 training + 6 evaluation presets implemented
- **Testing Suite**: 49 total tests (19 training + 11 preprocessing + 19 utilities), all passing
- **Documentation**: Comprehensive guides and examples with preprocessing workflows
- **GPU Support**: CUDA acceleration verified and working

### 🚀 **READY FOR PRODUCTION**
The system is fully operational for:
- **Data Preprocessing**: Quality assessment with blur heatmaps + group-aware splitting
- Training custom Cellpose models with preprocessed data
- Evaluating model performance with standardized metrics  
- Processing large datasets with GPU acceleration
- Research workflows with reproducible configurations

### 📋 **Future Enhancements** (Optional)
- 3D training workflow modularization
- Advanced cell tracking integration
- Additional cell-type presets
- GUI interface development

---

*Last updated: July 3, 2025 - Cellpose-focused system verified and production-ready* ✅

