# Single Cell Analysis Pipeline

A comprehensive, modular pipeline for single-cell segmentation, tracking, and feature extraction in brightfield microscopy images using Cellpose models.

## 🚀 Quick Start

### Training a Model
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
```

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
├── src/
│   ├── models/                        # 🤖 Model implementations (REFACTORED)
│   │   ├── base_trainer.py            # Abstract base classes for training/evaluation
│   │   ├── cellpose_trainer.py        # Cellpose trainer and evaluator
│   │   ├── train_eval_2d.py           # Unified 2D training interface
│   │   ├── config_presets.py          # Configuration presets (7 training, 6 eval)
│   │   └── cellpose/                  # Cellpose-specific scripts
│   │       ├── cellpose_2D_prediction.py  # 2D Cellpose prediction
│   │       ├── cellpose_2D_train.py       # 2D Cellpose training
│   │       └── cellpose_3D_train.py       # 3D Cellpose training
│   ├── preprocessing/                 # 🔄 Data preprocessing pipeline (NEW)
│   │   ├── blur_analysis.py           # Blur heatmap generation for quality control
│   │   ├── dataset_split.py           # Group-aware train/test splitting
│   │   └── __init__.py                # Preprocessing module interface
│   ├── utils/                         # 🔧 Core utilities (CONSOLIDATED)
│   │   ├── blur_measure.py            # Image blur/sharpness detection
│   │   ├── file_utils.py              # Unified file handling (replaces file_handler.py + file_naming.py)
│   │   ├── conversion.py              # 2D/3D format conversion utilities
│   │   └── ...                        # Other utility modules
│   ├── data/                          # 📁 Data processing and management
│   │   ├── analyze_tif_labels.py      # Label analysis and validation
│   │   ├── make_train_3D.py           # 3D training data preparation  
│   │   └── reformat_data.py           # Data reformatting utilities
│   ├── util/                         # 🔧 Legacy utilities (to be migrated)
│   │   ├── converter_2d_3d.py        # Legacy 2D/3D conversion
│   │   ├── feature_extractor.py      # PyRadiomics feature extraction
│   │   ├── file_renamer.py           # File naming standardization
│   │   ├── track_cells.py            # Cell tracking algorithms
│   │   ├── train_test_split.py       # Legacy dataset splitting
│   │   └── tif_to_png.py             # TIFF to PNG conversion
│   └── visualize/                    # 👁️ Visualization tools
│       ├── view_3d_tiff.py           # 3D TIFF viewer
│       ├── view_4d_tiff.py           # 4D TIFF viewer
│       └── visualize_predcition.py   # Prediction visualization
├── examples/                         # 📚 Usage examples and tutorials
│   ├── training_examples.py          # Basic training examples
│   └── complete_usage_examples.py    # Comprehensive usage guide
├── docs/                            # 📖 Documentation
│   └── TRAINING_GUIDE.md             # Detailed training guide
├── tests/                           # ✅ Test suite
│   ├── preprocessing/                # Preprocessing module tests
│   │   ├── test_blur_analysis.py     # Blur analysis tests (6 tests)
│   │   └── test_dataset_split.py     # Dataset splitting tests
│   ├── utils/                       # Utility tests
│   │   └── test_blur_measure.py      # Blur measurement tests (19 tests)
│   ├── test_training_system.py       # Modular system tests (19 tests)
│   └── test_integration.py           # Integration tests
├── config/                          # ⚙️ Configuration files
├── scripts/                         # 🔧 Standalone scripts
└── github/                          # 📦 Git submodules (cellpose, omnipose)
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

### ✅ **Installation Status**
- **Environment Tested**: cellpose-env conda environment
- **PyTorch**: v1.13.1+cu117 with CUDA 11.7 support verified
- **Cellpose**: v4.0.6 installed and tested
- **All Tests**: 19 modular system tests + integration tests passing

## Quick Start

### 1. Preprocess Raw Data

Before training or analysis, preprocess your raw dataset:

```python
# Generate blur heatmaps for quality assessment
from src.preprocessing.blur_analysis import measure_dataset_blur_heatmaps

blur_results = measure_dataset_blur_heatmaps(
    input_dir="data/raw/images",
    output_dir="data/processed/blur_heatmaps"
)

# Create train/test split with group awareness
from src.preprocessing.dataset_split import split_dataset
from src.utils.file_utils import BF_IF_FileHandler

file_patterns = {
    'images': 'data/raw/*/t1_*_w1_*.tif',
    'masks': 'data/raw/*/Cells_*.tif'
}

split_results = split_dataset(
    file_patterns=file_patterns,
    output_dir="data/processed/split",
    test_fraction=0.2,
    file_handler=BF_IF_FileHandler()
)
```

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

## Preprocessing Pipeline

Before training models or running analysis, raw datasets need to be preprocessed. This involves two key steps:

### Step 1: Generate Blur Heatmaps

Generate blur quality heatmaps for all images to enable quality assessment and filtering:

```python
from src.preprocessing.blur_analysis import measure_dataset_blur_heatmaps

# Generate blur heatmaps for the entire dataset
results = measure_dataset_blur_heatmaps(
    input_dir="data/raw/images",
    output_dir="data/processed/blur_heatmaps",
    pattern="*.tif",
    patch_size=32,           # Size of patches for blur detection
    stride_size=16,          # Stride between patches
    normalize=True,          # Normalize values to [0,1] range
    center_values=True,      # Center values on patches
    overwrite=False          # Skip existing files
)

print(f"Generated blur heatmaps for {len(results)} images")
```

**Command Line Usage:**
```bash
python -m src.preprocessing.blur_analysis \
    --input data/raw/images \
    --output data/processed/blur_heatmaps \
    --patch-size 32 \
    --stride-size 16 \
    --normalize
```

### Step 2: Split Dataset into Train/Test

Create group-aware train/test splits ensuring all images from the same experimental group stay together:

```python
from src.preprocessing.dataset_split import split_dataset
from src.utils.file_utils import BF_IF_FileHandler

# Define file patterns for images and masks
file_patterns = {
    'images': 'data/raw/*/t1_*_w1_*.tif',
    'masks': 'data/raw/*/Cells_*.tif'
}

# Split dataset with group awareness
results = split_dataset(
    file_patterns=file_patterns,
    output_dir="data/processed/split",
    test_fraction=0.2,        # 20% for testing
    random_seed=42,           # For reproducibility
    file_handler=BF_IF_FileHandler(),  # Handles standardized naming
    copy_files=True           # Copy files (Windows compatible)
)

print(f"Train set: {len(results['train']['images'])} images")
print(f"Test set: {len(results['test']['images'])} images")
```

**Alternative Directory-based Split:**
```python
from src.preprocessing.dataset_split import train_test_split_directory

# Split an existing directory structure
train_test_split_directory(
    input_dir="data/raw",
    output_dir="data/processed/split",
    test_fraction=0.2,
    file_handler=BF_IF_FileHandler(),
    random_seed=42
)
```

### Complete Preprocessing Workflow

```python
from pathlib import Path
from src.preprocessing.blur_analysis import measure_dataset_blur_heatmaps  
from src.preprocessing.dataset_split import split_dataset
from src.utils.file_utils import BF_IF_FileHandler

# Setup paths
raw_data_dir = Path("data/raw")
processed_dir = Path("data/processed")

# Step 1: Generate blur heatmaps
print("Step 1: Generating blur heatmaps...")
blur_results = measure_dataset_blur_heatmaps(
    input_dir=raw_data_dir / "images",
    output_dir=processed_dir / "blur_heatmaps",
    pattern="*.tif",
    normalize=True,
    overwrite=False
)

# Step 2: Split dataset
print("Step 2: Creating train/test split...")
file_patterns = {
    'images': str(raw_data_dir / '*' / 't1_*_w1_*.tif'),
    'masks': str(raw_data_dir / '*' / 'Cells_*.tif')
}

split_results = split_dataset(
    file_patterns=file_patterns,
    output_dir=processed_dir / "split",
    test_fraction=0.2,
    random_seed=42,
    file_handler=BF_IF_FileHandler(),
    copy_files=True
)

print(f"✅ Preprocessing complete!")
print(f"   📊 Blur heatmaps: {len(blur_results)} images")
print(f"   🎯 Train set: {len(split_results['train']['images'])} images") 
print(f"   🧪 Test set: {len(split_results['test']['images'])} images")
```

### Output Structure

After preprocessing, your data will be organized as:

```
data/
├── raw/                              # Original data
│   ├── plate1/
│   ├── plate2/
│   └── ...
└── processed/
    ├── blur_heatmaps/               # Quality assessment maps
    │   ├── image1_blur_heatmap.tif
    │   ├── image2_blur_heatmap.tif
    │   └── ...
    └── split/                       # Train/test split
        ├── train/
        │   ├── images/
        │   └── masks/ 
        └── test/
            ├── images/
            └── masks/
```

### Preprocessing Benefits

- **🎯 Quality Control**: Blur heatmaps identify low-quality regions for filtering
- **🔄 Group Awareness**: Train/test split maintains experimental groups
- **📁 Organization**: Standardized structure for downstream analysis
- **🔧 Reproducibility**: Configurable parameters with seeded random splits
- **💾 Efficiency**: Skip existing files to avoid reprocessing

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

