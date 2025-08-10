# Single Cell Analysis Pipeline

A comprehensive, modular pipeline for single-cell segmentation, tracking, and feature extraction in brightfield microscopy images using Cellpose models.

## Quick Start

### Using Scripts (Recommended)

```bash
# 1. Preprocess data
python scripts/run_preprocessing.py data/raw data/processed

# 2. Run inference
python scripts/run_inference.py --input-dir data/test --output-dir results

# 3. Postprocess results
python scripts/run_postprocessing.py results/segmentation results/tracked

# 4. Extract features from segmented cells
python scripts/run_feature_extraction.py --config config/feature_extraction_config.yaml

# 5. Launch GUI for visualization
python scripts/launch_gui_safe.py
```

### Using Python API

```python
from src.inference.inference_pipeline import InferencePipeline
from src.inference.cellpose_predictor import CellposePredictor
from src.inference.output_manager import OutputManager

# Create inference pipeline
predictor = CellposePredictor(model_type="cyto3", gpu=True)
output_manager = OutputManager(base_output_dir="results")
pipeline = InferencePipeline(predictor, output_manager)


# Run inference
results = pipeline.run_inference(input_dir="data/test")
```

## Overview

This project provides a complete, modular workflow for:
- **2D/3D Cell Segmentation**: Using Cellpose models with preset configurations
- **Cell Tracking**: Tracking cells across z-stacks using trackpy
- **Feature Extraction**: Comprehensive morphological, intensity, spatial, and texture feature extraction from segmented cells
- **Data Management**: Standardized file naming and dataset organization
- **Quality Control**: Blur detection and filtering for improved segmentation
- **Visualization**: 3D/4D visualization tools using Napari

## Project Structure

```
single_cell/
├── src/                               # Core modules
│   ├── core/                          # Core functionality and base classes
│   ├── preprocessing/                 # Data preprocessing pipeline
│   ├── inference/                     # Inference pipeline
│   ├── features/                      # Feature extraction
│   ├── postprocessing/               # Post-inference processing
│   ├── utils/                         # Core utilities
│   │   ├── file_utils.py             # Unified file handling
│   │   ├── blur_measure.py           # Image blur/sharpness detection
│   │   ├── conversion.py             # 2D/3D format conversion utilities
│   │   └── config.py                 # Configuration management
│   └── visualize/                     # Visualization tools
├── scripts/                           # Command-line interfaces
│   ├── run_preprocessing.py          # Data preprocessing script
│   ├── run_inference.py              # Inference execution script
│   ├── run_postprocessing.py         # Postprocessing script
│   ├── run_feature_extraction.py     # Feature extraction script
│   ├── run_pipeline.py               # Complete pipeline script
│   └── launch_gui_safe.py            # GUI launcher with dependency checking
├── examples/                          # Usage examples and tutorials
│   └── complete_usage_examples.py    # Comprehensive usage guide
├── tests/                            # Test suite
├── config/                           # Configuration files
├── data/                             # Data directories
    ├── sample_plates/                # Raw sample data
    └── sample_plates_processed/      # Processed sample data
```

## Installation

### Prerequisites
- Python 3.10+ (tested with Python 3.10.18)
- CUDA-capable GPU (optional, for faster processing)
- **Verified Environment**: Windows 10/11, Anaconda/Miniconda

### Environment Setup (Tested & Validated)

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
python -m pytest tests/ -v
```

### Installation Status
- **Environment Tested**: cellpose-env conda environment
- **PyTorch**: v1.13.1+cu117 with CUDA 11.7 support verified
- **Cellpose**: v4.0.6 installed and tested
- **All Tests**: 49 total tests passing

## Usage

### Command Line Interface (Recommended)

#### 1. Data Preprocessing
```bash
# Full preprocessing pipeline
python scripts/run_preprocessing.py data/raw data/processed \
    --test-size 0.2 \
    --random-seed 42 \
    --patch-size 32 \
    --stride-size 16

# Options:
#   --test-size: Fraction for test set (default: 0.2)
#   --random-seed: Random seed for reproducibility
#   --patch-size: Patch size for blur detection
#   --stride-size: Stride size for blur detection
#   --overwrite: Overwrite existing files
```

#### 2. Inference
```bash
# Run inference on test data
python scripts/run_inference.py \
    --input-dir data/processed/split/test \
    --output-dir results/inference \
    --model-name cyto3 \
    --flow-threshold 0.4 \
    --gpu

# Batch inference with config file
python scripts/run_inference.py \
    --config config/inference_config.yaml
```

#### 3. Postprocessing
```bash
# 3D cell tracking and blur filtering
python scripts/run_postprocessing.py \
    --input-dir results/inference \
    --output-dir results/tracked \
    --search-range 5 \
    --memory 1 \
    --blur-threshold 0.5

# Single file processing
python scripts/run_postprocessing.py \
    --single-file results/inference/sample_3d.tif \
    --output-dir results/tracked \
    --image-file data/test/sample_BF_3d.tif
```

#### 4. Feature Extraction
```bash
# Extract features from segmented cells
python scripts/run_feature_extraction.py \
    --config config/feature_extraction_config.yaml

# Custom patterns and options
python scripts/run_feature_extraction.py \
    --input-dir data/segmented \
    --output-dir results/features \
    --image-pattern "*_BF.tif" \
    --mask-pattern "Cells_*.tif" \
    --n-jobs 8

# Process multiple datasets
python scripts/run_feature_extraction.py \
    --config config/multi_dataset_config.yaml
```

#### 5. Complete Pipeline
```bash
# Run entire pipeline from raw data to results
python scripts/run_pipeline.py \
    --input-dir data/raw \
    --output-dir results \
    --config config/config.yaml \
    --preprocess \
    --inference \
    --postprocess
```

#### 6. Visualization GUI
```bash
# Launch safe GUI with dependency checking
python scripts/launch_gui_safe.py

# Direct visualization
python scripts/visualize_results.py \
    --input results/tracked \
    --type 3d
```

### Python API Usage

For detailed Python API usage, see the module-specific README files:

- **Preprocessing**: `src/preprocessing/README.md`
- **Inference**: `src/inference/README.md`
- **Postprocessing**: `src/postprocessing/README.md`

## Core Modules

### Preprocessing (`src/preprocessing/`)
Data preprocessing pipeline with blur analysis and group-aware dataset splitting.

**Key Features:**
- Blur heatmap generation for quality assessment
- Group-aware train/test splitting
- 2D to 3D stack combination
- Standardized file naming and organization

**Usage:** `python scripts/run_preprocessing.py`  
**Details:** See `src/preprocessing/README.md`

### Inference (`src/inference/`)
Organized inference pipeline for running predictions on test datasets.

**Key Features:**
- Modular predictor architecture
- Structured output management
- Z-stack and batch processing support
- Visualization and logging

**Usage:** `python scripts/run_inference.py`  
**Details:** See `src/inference/README.md`

### Features (`src/features/`)
Comprehensive feature extraction from 2D instance segmentations.

**Key Features:**
- 23 morphological, intensity, spatial, and texture features per cell
- Parallel processing with configurable batch sizes

**Extracted Features:**
- **Morphology**: Area, perimeter, elongation, compactness, circularity, Feret diameter, etc.
- **Intensity**: Mean, std, CV, total intensity
- **Spatial**: Centroids, center of mass, mass displacement
- **Texture**: Gabor filters, skewness, kurtosis, entropy

**Usage:** `python scripts/run_feature_extraction.py`  
**Details:** See `docs/FEATURE_EXTRACTION.md`

### 📊 Postprocessing (`src/postprocessing/`)
3D cell tracking and quality filtering pipeline.

**Key Features:**
- 3D cell tracking across z-stacks
- Blur-based quality filtering
- Configurable processing workflows
- Comprehensive result reporting

**Usage:** `python scripts/run_postprocessing.py`  
**Details:** See `src/postprocessing/README.md`

### � Utils (`src/utils/`)
Core utilities for file handling, format conversion, and configuration management.

**Key Features:**
- Unified file handling and naming
- Blur measurement and assessment
- 2D/3D format conversion
- Configuration loading and validation

### 👁️ Visualization (`src/visualize/`)
3D/4D visualization tools using Napari.

**Key Features:**
- 3D/4D TIFF stack visualization
- Prediction overlay visualization
- Interactive GUI with dependency checking

**Usage:** `python scripts/launch_gui_safe.py`

## Configuration

The pipeline uses YAML configuration files for reproducible experiments:

### Main Configuration (`config/config.yaml`)
```yaml
segmentation:
  model_type: "cyto3"
  flow_threshold: 0.4
  min_cell_size: 30

tracking:
  search_range: 5
  memory: 1
  blur_threshold: 0.5

preprocessing:
  test_fraction: 0.2
  random_seed: 42
  patch_size: 32
  stride_size: 16
```

### Inference Configuration (`config/inference_config.yaml`)
```yaml
model:
  name: "cyto3"
  flow_threshold: 0.4
  gpu: true

input:
  directory: "data/processed/split/test"
  pattern: "*_BF.tif"

output:
  directory: "results/inference"
  save_overlays: true
  save_metadata: true
```

### Feature Extraction Configuration (`config/feature_extraction_config.yaml`)
```yaml
paths:
  input_dir: "data/segmented"
  output_dir: "data/features_output"

feature_extraction:
  n_jobs: -1
  batch_size: 50
  features:
    morphology: true
    intensity: true
    spatial: true
    texture: true
    file_patterns:
    images: ["*.tif", "*_BF*.tif"]
    masks: ["Cells_*.tif", "*_mask*.tif"]
```

## Supported File Formats

- **Input**: Multi-well plate TIFF files with standardized naming
- **Output**: Segmentation masks, tracked labels, feature tables (CSV)
- **Visualization**: 3D/4D TIFF stacks compatible with Napari
- **Configuration**: YAML files for reproducible experiments

## Examples

### Complete Examples Available
- **`examples/complete_usage_examples.py`**: Comprehensive usage demonstrations
- **`examples/inference_example.py`**: Basic inference workflow examples  
- **`examples/feature_extraction_examples.py`**: Feature extraction workflows and analysis

### Quick Command Line Examples

#### Complete Pipeline
```bash
# Process raw data through entire pipeline
python scripts/run_pipeline.py \
    --input-dir data/raw \
    --output-dir results \
    --preprocess \
    --inference \
    --postprocess
```

#### Individual Steps
```bash
# 1. Preprocess data
python scripts/run_preprocessing.py data/raw data/processed

# 2. Run inference
python scripts/run_inference.py \
    --input-dir data/processed/split/test \
    --output-dir results/inference \
    --model-name cyto3

# 3. Postprocess results
python scripts/run_postprocessing.py \
    --input-dir results/inference \
    --output-dir results/tracked
```

### Quick Python API Examples

#### Modular Training with Presets
```python
from src.models.config_presets import create_training_config_from_preset
from src.models.cellpose_trainer import CellposeTrainer

config = create_training_config_from_preset(
    "mammalian_cells_cellpose",
    train_dir="data/processed/split/train",
    test_dir="data/processed/split/test",
    output_dir="models/cellpose"
)
trainer = CellposeTrainer(config)
trainer.train()
```

#### Modular Evaluation
```python
from src.models.config_presets import create_evaluation_config_from_preset
from src.models.cellpose_trainer import CellposeEvaluator

config = create_evaluation_config_from_preset(
    "default",
    test_dir="data/processed/split/test",
    model_path="cyto2",
    output_dir="results/evaluation"
)
evaluator = CellposeEvaluator(config)
results = evaluator.evaluate()
```

## Testing & Validation

### Test Suite
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/preprocessing/ -v     # Preprocessing tests
python -m pytest tests/utils/ -v            # Utility tests  
python -m pytest tests/test_training_system.py -v  # Training system tests

# Verify GPU/CUDA functionality
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## Project Status

### **READY FOR PRODUCTION**
The system is fully operational for:
- **Data Preprocessing**: Quality assessment with blur heatmaps + group-aware splitting
- **Model Training**: Custom Cellpose models with preprocessed data
- **Inference**: Large-scale prediction with organized output structure
- **Postprocessing**: 3D cell tracking and quality filtering
- **Evaluation**: Model performance assessment with standardized metrics
- **Visualization**: Interactive 3D/4D visualization with Napari
- **Research Workflows**: Reproducible configurations and command-line automation

---

